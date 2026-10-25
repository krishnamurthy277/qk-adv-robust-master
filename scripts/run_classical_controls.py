"""Train-only variance-matched and CV-selected RBF controls for saved QSVM cells."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import brentq, minimize_scalar
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from scripts.run_experiment import load_dataset, atomic_csv
from src.data.split import get_train_test_split
from src.data.preprocessing import QKPreprocessor
from src.attacks.a0_margin import a0_linearized_margin
from src.attacks.a2_pgd import a2_pgd_attack
from src.utils.hashing import stable_hash
from src.utils.seed import seed_everything


def matched_gamma(X,target):
    """Smallest gamma root; deterministic broad-kernel branch, train data only.

    RBF off-diagonal variance is not monotone in gamma. Return residual for
    auditing unattainable targets; never silently claim an exact match.
    """
    mask=~np.eye(len(X),dtype=bool)
    def variance(log_gamma):
        return float(rbf_kernel(X,gamma=np.exp(log_gamma))[mask].var())
    def objective(log_gamma): return variance(log_gamma)-target
    grid=np.linspace(-16,12,281)
    values=[objective(x) for x in grid]
    roots=[]
    for a,b,fa,fb in zip(grid[:-1],grid[1:],values[:-1],values[1:]):
        if fa==0: roots.append(a)
        elif fa*fb<0: roots.append(brentq(objective,a,b,xtol=1e-12))
    if roots:
        chosen=min(roots)
    else:
        best=int(np.argmin(np.abs(values)))
        result=minimize_scalar(lambda z:objective(z)**2,
            bounds=(grid[max(0,best-1)],grid[min(len(grid)-1,best+1)]),method='bounded')
        chosen=result.x
    return float(np.exp(chosen)),variance(chosen),len(roots)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/tier_a_main.yaml')
    parser.add_argument('--results',default='results/tables/tier_a_results.csv')
    parser.add_argument('--output',default='results/tables/classical_controls.csv')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    table=pd.read_csv(args.results)
    models=table.drop_duplicates('cell_hash')
    expected=np.prod([len(cfg[k]) for k in ('datasets','seeds','reps','noise_p','bandwidth')])
    if len(models)!=expected or len(table)!=expected*len(cfg['eps_rel']):
        raise ValueError('Complete Tier A table required before Phase 6 controls')
    torch.set_num_threads(1)
    fingerprint=hashlib.sha256(Path(__file__).read_bytes()+Path(args.results).read_bytes()).hexdigest()[:16]
    cache=Path('results/runs/classical_controls')/fingerprint
    cache.mkdir(parents=True,exist_ok=True)
    rows=[]
    for (dataset,seed),group in models.groupby(['dataset','seed']):
        seed_everything(int(seed))
        raw,y=load_dataset(dataset,cfg['n_qubits'])
        tr,te,ytr,yte=get_train_test_split(raw,y,cfg['train_size'],cfg['test_size'],int(seed))
        prep=QKPreprocessor(cfg['n_qubits'],cfg.get('omics_select_k',1000) if dataset=='gse2034' else None)
        X=prep.fit_transform(tr,ytr); Y=prep.transform(te)
        ytr=np.where(ytr<=0,-1,1); yte=np.where(yte<=0,-1,1)
        cv=GridSearchCV(SVC(kernel='rbf'),{'C':[.1,1,10,100],
                  'gamma':['scale','auto',.01,.1,1.]},cv=3).fit(X.numpy(),ytr)
        cv_model=cv.best_estimator_
        for model in group.itertuples():
            for mode in ('variance_matched','cv'):
                key=stable_hash(dict(qcell=model.cell_hash,mode=mode))
                path=cache/f'{key}.csv'
                if path.exists():
                    rows.extend(pd.read_csv(path).to_dict('records')); continue
                seed_everything(int(seed))
                if mode=='variance_matched':
                    gamma,variance,roots=matched_gamma(X.numpy(),model.var_K)
                    fit=GridSearchCV(SVC(kernel='rbf',gamma=gamma),{'C':[.1,1,10,100]},cv=3).fit(X.numpy(),ytr)
                    svm=fit.best_estimator_
                else:
                    svm=cv_model; gamma=float(svm._gamma); roots=0
                    gram=rbf_kernel(X.numpy(),gamma=gamma)
                    variance=float(gram[~np.eye(len(X),dtype=bool)].var())
                sv=X[svm.support_]
                coefficients=torch.tensor(svm.dual_coef_[0],dtype=torch.float64)
                bias=float(svm.intercept_[0])
                def kernel(a,b):
                    return torch.exp(-gamma*((a[:,None,:]-b[None,:,:])**2).sum(-1))
                a0=a0_linearized_margin(Y,yte,sv,coefficients,bias,kernel)
                batch=[]
                for eps_rel in cfg['eps_rel']:
                    attack=a2_pgd_attack(Y,yte,sv,coefficients,bias,kernel,
                        eps_rel*model.d_med,cfg['pgd_T'],cfg['pgd_R'])
                    batch.append(dict(qcell_hash=model.cell_hash,dataset=dataset,seed=seed,
                        reps=model.reps,noise_p=model.noise_p,bandwidth=model.bandwidth,
                        mode=mode,gamma=gamma,gamma_roots=roots,C_chosen=svm.C,
                        qsvm_var_K=model.var_K,rbf_var_K=variance,
                        variance_relative_error=abs(variance-model.var_K)/max(model.var_K,1e-30),
                        clean_acc=float(np.mean(svm.predict(Y.numpy())==yte)),
                        a0_median_margin=float(np.median(a0['margins'])) if len(a0['margins']) else np.nan,
                        eps_rel=eps_rel,a2_robust_acc=attack['robust_acc'],
                        a2_asr=attack['attack_success_rate'],source_hash=fingerprint))
                atomic_csv(pd.DataFrame(batch),path)
                rows.extend(batch)
            print(f'Controls completed: {dataset} seed={seed} reps={model.reps} p={model.noise_p} c={model.bandwidth}',flush=True)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_csv(pd.DataFrame(rows),output)
    (cache/'manifest.json').write_text(json.dumps(dict(config=cfg,input=args.results,
          fingerprint=fingerprint,rows=len(rows),gamma_branch='smallest positive root'),indent=2))


if __name__=='__main__': main()
