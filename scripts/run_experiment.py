"""Resumable Tier A runner with per-cell atomic checkpoints.

Workers own separate caches. The parent atomically rebuilds the aggregate
CSV after each completed cell. Manifests include source/data/config hashes.
One numerical-library thread per worker; attack gradient batches <= 8.
"""
import argparse
import itertools
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# ── project imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gse2034 import load_gse2034
from src.data.preprocessing import QKPreprocessor
from src.data.split import get_train_test_split
from src.data.synthetic import load_synthetic
from src.data.wbcd import load_wbcd
from src.attacks.a0_margin import a0_linearized_margin
from src.attacks.a1_transfer import a1_transfer_attack
from src.attacks.a2_pgd import a2_pgd_attack
from src.attacks.base import median_pairwise_l2
from src.attacks.masking import run_masking_diagnostics
from src.kernel.fidelity import compute_kernel_matrix, fixed_reference_kernel
from src.kernel.psd_repair import make_psd
from src.kernel.gram import get_gram_matrix
from src.metrics.concentration import compute_concentration_metrics
from src.metrics.margin import compute_kta
from src.utils.hashing import stable_hash
from src.utils.paths import get_cache_dir, get_results_dir, get_tables_dir
from src.utils.seed import seed_everything

# ── constants ────────────────────────────────────────────────────────────────
RESULTS_VERSION = "v2_per_gate"


# ── data loading ─────────────────────────────────────────────────────────────
def load_dataset(name: str, n_qubits: int):
    if name == "wbcd":
        return load_wbcd()
    elif name == "synthetic":
        return load_synthetic("classification", n_samples=300, n_features=n_qubits)
    elif name == "gse2034":
        return load_gse2034()
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ── kernel function factory ───────────────────────────────────────────────────
def make_kernel_fn(reps, c, p):
    def fn(X1, X2):
        return compute_kernel_matrix(X1, X2, reps=reps, c=c, p=p)
    return fn


# ── SVM fitting ──────────────────────────────────────────────────────────────
def fit_svm_precomputed(K_train: np.ndarray, y_train: np.ndarray):
    """Fit SVC with precomputed kernel + CV for C. Returns (svm, C_chosen)."""
    param_grid = {"C": [0.1, 1, 10, 100]}
    clf = GridSearchCV(SVC(kernel="precomputed"), param_grid, cv=3)
    clf.fit(K_train, y_train)
    return clf.best_estimator_, clf.best_params_["C"]


# ── margin computation ────────────────────────────────────────────────────────
def compute_geometric_margin(svm, K_train: np.ndarray, sv_indices: np.ndarray) -> float:
    a_y = svm.dual_coef_[0]
    K_sv = K_train[np.ix_(sv_indices, sv_indices)]
    w_sq = float(np.dot(a_y, K_sv @ a_y))
    return 1.0 / math.sqrt(w_sq) if w_sq > 0 else 0.0


# ── single cell runner ────────────────────────────────────────────────────────
def run_cell(cfg: dict, dataset_name: str, seed: int,
             reps: int, p: float, c: float,
             results_path: Path, cache_dir: Path,
             pgd_T: int, pgd_R: int,
             eps_rels: list, eps_large_rel: float) -> dict | None:

    cell_id = f"{dataset_name}_s{seed}_r{reps}_p{p:.4f}_c{c:.2f}"
    torch.set_num_threads(1)

    cell_config = dict(dataset=dataset_name, seed=seed, reps=reps, p=p, c=c,
                       config=cfg, version=RESULTS_VERSION)
    cell_hash = stable_hash(cell_config)
    cache_dir = cache_dir / cell_hash
    cache_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── Load & preprocess ───────────────────────────────────────────────
    seed_everything(seed)
    n_qubits = cfg["n_qubits"]
    X_raw, y_raw = load_dataset(dataset_name, n_qubits)

    X_tr, X_te, y_tr, y_te = get_train_test_split(
        X_raw, y_raw, train_size=cfg["train_size"], test_size=cfg["test_size"], seed=seed
    )
    prep = QKPreprocessor(n_qubits=n_qubits, select_k=cfg.get('omics_select_k', 1000) if dataset_name=='gse2034' else None)
    X_tr_q = prep.fit_transform(X_tr, y_tr)   # torch.Tensor (N_tr, n)
    X_te_q = prep.transform(X_te)

    y_tr_pm = np.where(y_tr <= 0, -1, 1)
    y_te_pm = np.where(y_te <= 0, -1, 1)

    # ── Gram matrices ────────────────────────────────────────────────────
    K_tr = get_gram_matrix(X_tr_q, reps=reps, c=c, p=p,
                           cache_dir=cache_dir, config_dict=cell_config, repair=False).cpu().numpy()
    K_te_tr = compute_kernel_matrix(X_te_q, X_tr_q, reps=reps, c=c, p=p).detach().numpy()

    t_gram = time.time() - t_start

    # ── PSD repair flag ───────────────────────────────────────────────────
    min_eig = float(np.linalg.eigvalsh(K_tr).min())
    psd_repaired = min_eig < -1e-8

    # ── Fit SVM ──────────────────────────────────────────────────────────
    K_fit = make_psd(torch.tensor(K_tr), epsilon=0).numpy() if psd_repaired else K_tr
    svm, C_chosen = fit_svm_precomputed(K_fit, y_tr_pm)
    sv_indices = svm.support_
    X_sv = X_tr_q[sv_indices]
    alpha_y = torch.tensor(svm.dual_coef_[0], dtype=torch.float64)
    intercept = float(svm.intercept_[0])

    # Clean accuracy
    K_te_sv = K_te_tr[:, sv_indices]
    f_te = K_te_sv @ svm.dual_coef_[0] + intercept
    clean_acc = float(np.mean((f_te > 0).astype(int) == (y_te_pm > 0).astype(int)))

    # ── Concentration metrics ─────────────────────────────────────────────
    conc = compute_concentration_metrics(K_tr, n_qubits)
    kta = compute_kta(K_tr, y_tr_pm)
    geom_margin = compute_geometric_margin(svm, K_fit, sv_indices)

    # ── Attack setup ──────────────────────────────────────────────────────
    d_med = median_pairwise_l2(X_tr_q)
    kfn_noisy = fixed_reference_kernel(X_sv, reps, c, p)


    # Noiseless SVM for A1 surrogate
    K_tr_nl = get_gram_matrix(X_tr_q, reps=reps, c=c, p=0.0,
                              cache_dir=cache_dir,
                              config_dict={**cell_config, "p": 0.0}).cpu().numpy()
    svm_nl, _ = fit_svm_precomputed(K_tr_nl, y_tr_pm)
    X_sv_nl = X_tr_q[svm_nl.support_]
    alpha_y_nl = torch.tensor(svm_nl.dual_coef_[0], dtype=torch.float64)
    intercept_nl = float(svm_nl.intercept_[0])

    kfn_noiseless = fixed_reference_kernel(X_sv_nl, reps, c, 0.0)
    rows = []

    a0_result = a0_linearized_margin(X_te_q, y_te_pm, X_sv, alpha_y, intercept, kfn_noisy)
    a0_median_margin = float(np.median(a0_result['margins'])) if len(a0_result['margins']) else float('nan')
    a2_large = a2_pgd_attack(X_te_q, y_te_pm, X_sv, alpha_y, intercept, kfn_noisy,
                             eps=eps_large_rel*d_med, T=pgd_T, R=pgd_R)
    point_records = []
    for eps_rel in eps_rels:
        eps = eps_rel * d_med

        a2_result = a2_pgd_attack(
            X_te_q, y_te_pm, X_sv, alpha_y, intercept, kfn_noisy,
            eps=eps, T=pgd_T, R=pgd_R
        )

        # ── A2 single-step (for M1) ────────────────────────────────────
        a2_single_result = a2_pgd_attack(
            X_te_q, y_te_pm, X_sv, alpha_y, intercept, kfn_noisy,
            eps=eps, T=1, R=1
        )

        # ── A2 no-restarts (for M5) ────────────────────────────────────
        a2_norestart_result = a2_pgd_attack(
            X_te_q, y_te_pm, X_sv, alpha_y, intercept, kfn_noisy,
            eps=eps, T=pgd_T, R=1
        )

        # ── A1: transfer attack ────────────────────────────────────────
        a1_result = a1_transfer_attack(
            X_te_q, y_te_pm,
            X_sv_nl, alpha_y_nl, intercept_nl, kfn_noiseless,
            X_sv, alpha_y, intercept, kfn_noisy,
            eps=eps, T=pgd_T, R=pgd_R
        )

        # ── M2: large eps ──────────────────────────────────────────────
        masking = run_masking_diagnostics(
            robust_acc_a1=a1_result["robust_acc"],
            robust_acc_a2=a2_result["robust_acc"],
            robust_acc_a2_singlestep=a2_single_result["robust_acc"],
            robust_acc_a2_no_restarts=a2_norestart_result["robust_acc"],
            robust_acc_large_eps=a2_large["robust_acc"],
            grad_norms=a0_result["grad_norms"],
        )
        m_diag = {m["diagnostic"]: m.get("passed", None) for m in masking if "diagnostic" in m}
        masking_gap = next((m["masking_gap"] for m in masking if "masking_gap" in m), None)

        row = {
            "cell_hash": cell_hash,
            "dataset": dataset_name,
            "seed": seed,
            "n_qubits": n_qubits,
            "reps": reps,
            "noise_p": p,
            "bandwidth": c,
            "C_chosen": C_chosen,
            "d_med": d_med,
            "eps_rel": eps_rel,
            "eps": eps,
            # Concentration
            "mu_K": conc["mu_K"],
            "var_K": conc["var_K"],
            "log10_var_K": conc["log10_var_K"],
            "cv_K": conc["cv_K"],
            "mu_diff": conc["mu_diff"],
            "spectral_ratio": conc["spectral_ratio"],
            "eff_rank": conc["eff_rank"],
            "kta": kta,
            "geom_margin": geom_margin,
            # PSD
            "min_eigenvalue": min_eig,
            "psd_repaired": psd_repaired,
            # Accuracy
            "clean_acc": clean_acc,
            # A0
            "a0_median_margin": a0_median_margin,
            "a0_median_grad_norm": float(np.median(a0_result["grad_norms"])) if len(a0_result["grad_norms"]) > 0 else float("nan"),
            # A1
            "a1_robust_acc": a1_result["robust_acc"],
            "a1_asr": a1_result["attack_success_rate"],
            # A2
            "a2_robust_acc": a2_result["robust_acc"],
            "a2_asr": a2_result["attack_success_rate"],
            # A2 single-step
            "a2_single_robust_acc": a2_single_result["robust_acc"],
            # A2 large eps
            "a2_large_robust_acc": a2_large["robust_acc"],
            # Masking diagnostics
            "M1_passed": m_diag.get("M1"),
            "M2_passed": m_diag.get("M2"),
            "M4_passed": m_diag.get("M4"),
            "M5_passed": m_diag.get("M5"),
            "masking_gap": masking_gap,
            # Timing
            "t_gram_s": t_gram,
            "t_total_s": time.time() - t_start,
            "version": RESULTS_VERSION,
            "source_hash": cfg.get('_source_hash', ''),
            "train_size": cfg['train_size'], "test_size": cfg['test_size'],
            "pgd_T": pgd_T, "pgd_R": pgd_R,
            "a2_norestart_robust_acc": a2_norestart_result['robust_acc'],
            "M3_passed": None,
            "grad_p10": float(np.percentile(a0_result['grad_norms'],10)) if len(a0_result['grad_norms']) else float('nan'),
            "grad_p90": float(np.percentile(a0_result['grad_norms'],90)) if len(a0_result['grad_norms']) else float('nan'),
            "psd_repair_frobenius": float(np.linalg.norm(K_fit-K_tr)),
        }
        rows.append(row)
        for i in range(len(y_te_pm)):
            point_records.append(dict(cell_hash=cell_hash, test_index=i, eps_rel=eps_rel,
                initially_correct=bool(a2_result['initially_correct'][i]),
                a1_correct=bool(a1_result['after_attack_correct'][i]),
                a2_correct=bool(a2_result['after_attack_correct'][i])))

    pd.DataFrame(point_records).to_csv(cache_dir / 'attack_points.csv', index=False)
    np.savez(cache_dir / 'a0_points.npz', **a0_result)
    return rows


# ── orchestrator ──────────────────────────────────────────────────────────────

def source_fingerprint():
    import hashlib
    root = Path(__file__).resolve().parent.parent
    files = sorted(p for p in (root / 'src').rglob('*.py') if 'analysis' not in p.parts) + [Path(__file__).resolve()]
    h = hashlib.sha256()
    for path in files:
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def atomic_csv(frame, path):
    import uuid
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def valid_cell(path, cfg):
    if not path.exists(): return False
    try:
        df = pd.read_csv(path)
        return (len(df) == len(cfg['eps_rel']) and
                set(df.eps_rel) == set(cfg['eps_rel']) and
                df.version.eq(RESULTS_VERSION).all() and
                df.source_hash.eq(cfg['_source_hash']).all() and
                df[['clean_acc','a1_robust_acc','a2_robust_acc','var_K']].notna().all().all())
    except (OSError, ValueError, KeyError):
        return False


def run_checkpointed(cfg, cell, run_dir):
    dataset,seed,reps,p,c = cell
    cell_key = stable_hash(dict(dataset=dataset,seed=seed,reps=reps,p=p,c=c))
    path = run_dir/'cells'/f'{cell_key}.csv'
    if valid_cell(path,cfg):
        return str(path), 'cached'
    try:
        rows = run_cell(cfg,dataset,seed,reps,p,c,path,run_dir/'cache',
                        cfg.get('pgd_T',40),cfg.get('pgd_R',3),cfg['eps_rel'],cfg.get('eps_large_rel',5.))
        atomic_csv(pd.DataFrame(rows),path)
        return str(path),'computed'
    except Exception:
        error = dict(cell=cell,traceback=traceback.format_exc(),time=time.time())
        (run_dir/'failures'/f'{cell_key}_{time.time_ns()}.json').write_text(json.dumps(error,indent=2))
        return str(path),'failed'


def main():
    import hashlib
    import platform
    import shutil
    from joblib import parallel_config
    from filelock import FileLock
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/tier_a_main.yaml')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8-sig'))
    cfg['_source_hash']=source_fingerprint()
    cfg['_data_hashes']={}
    for name in cfg['datasets']:
        X,y=load_dataset(name,cfg['n_qubits'])
        cfg['_data_hashes'][name]=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest()
    run_id=stable_hash(cfg)
    run_dir=get_results_dir()/'runs'/cfg['experiment']/run_id
    for folder in ('cells','cache','failures','source'):
        (run_dir/folder).mkdir(parents=True,exist_ok=True)
    manifest=dict(config=cfg,python=sys.version,platform=platform.platform(),run_id=run_id)
    (run_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
    root=Path(__file__).resolve().parent.parent
    for source in list((root/'src').rglob('*.py'))+[Path(__file__).resolve()]:
        target=run_dir/'source'/source.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copy2(source,target)
    grid=list(itertools.product(cfg['datasets'],cfg['seeds'],cfg['reps'],cfg['noise_p'],cfg['bandwidth']))
    output=get_tables_dir()/cfg.get('results_file',cfg['experiment']+'_results.csv')
    if output.exists():
        archive=run_dir/('previous_'+output.name)
        if not archive.exists(): shutil.copy2(output,archive)
    print(f'Run {run_id}: {len(grid)} cells; checkpoints {run_dir}',flush=True)
    paths=[]
    failures=0
    with FileLock(str(run_dir / '.run.lock'), timeout=0), parallel_config(backend='loky',inner_max_num_threads=1):
        results=Parallel(n_jobs=cfg.get('n_jobs',1),return_as='generator_unordered')(
            delayed(run_checkpointed)(cfg,cell,run_dir) for cell in grid)
        for path,status in results:
            if status=='failed':
                failures+=1
            else:
                paths.append(path)
                atomic_csv(pd.concat([pd.read_csv(p) for p in paths],ignore_index=True),output)
            print(f'{len(paths)}/{len(grid)} complete, {failures} failed ({status})',flush=True)
    status=dict(expected_cells=len(grid),completed_cells=len(paths),failures=failures,
                complete=len(paths)==len(grid),results=str(output),run_id=run_id)
    (run_dir/'status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status),flush=True)
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
