"""T8: Attack sanity checks.

Pass criteria:
- Robust accuracy is non-increasing in eps.
- At very large eps it approaches 0 on RBF-SVM and noiseless QSVM.
- PGD (multi-step) >= single-step attack strength.
"""
import math
import pytest
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.data.synthetic import load_synthetic
from src.data.preprocessing import QKPreprocessor
from src.data.split import get_train_test_split
from src.kernel.fidelity import compute_kernel_matrix
from src.kernel.gram import get_gram_matrix
from src.attacks.a2_pgd import a2_pgd_attack
from src.attacks.base import median_pairwise_l2


def _build_kernel_fn(X_tr_q, reps, c, p):
    """Return a closure: kernel_fn(X1, X2) -> K tensor."""
    def kernel_fn(X1, X2):
        return compute_kernel_matrix(X1, X2, reps=reps, c=c, p=p)
    return kernel_fn


def _fit_qsvm(X_tr_q, y_tr_pm, reps=2, c=1.0, p=0.0):
    """Fit a precomputed-kernel SVC and return (svm, X_sv, alpha_y, intercept)."""
    K_tr = get_gram_matrix(X_tr_q, reps=reps, c=c, p=p).cpu().numpy()
    param_grid = {"C": [0.1, 1, 10, 100]}
    clf = GridSearchCV(SVC(kernel="precomputed"), param_grid, cv=3)
    clf.fit(K_tr, y_tr_pm)
    svm = clf.best_estimator_
    X_sv = X_tr_q[svm.support_]
    alpha_y = torch.tensor(svm.dual_coef_[0], dtype=torch.float64)
    intercept = float(svm.intercept_[0])
    return svm, X_sv, alpha_y, intercept


def test_t8_attack_sanity():
    """T8: Attack sanity — non-increasing robust acc in eps; large-eps -> ~0; PGD >= 1-step."""
    torch.manual_seed(0)
    np.random.seed(0)

    # Use a tiny 2-qubit moons dataset for speed
    n_qubits = 2
    X, y_raw = load_synthetic("moons", n_samples=100, n_features=2)
    X_tr, X_te, y_tr, y_te = get_train_test_split(X, y_raw, train_size=50, test_size=50, seed=0)

    prep = QKPreprocessor(n_qubits=n_qubits)
    X_tr_q = prep.fit_transform(X_tr)
    X_te_q = prep.transform(X_te)

    # Labels in {-1, +1}
    y_tr_pm = np.where(y_tr == 0, -1, 1)
    y_te_pm = np.where(y_te == 0, -1, 1)

    # Fit noiseless QSVM
    svm, X_sv, alpha_y, intercept = _fit_qsvm(X_tr_q, y_tr_pm, reps=2, c=1.0, p=0.0)
    kfn = _build_kernel_fn(X_tr_q, reps=2, c=1.0, p=0.0)

    d_med = median_pairwise_l2(X_tr_q)
    eps_values = [0.05 * d_med, 0.1 * d_med, 0.2 * d_med, 0.4 * d_med]

    # --- Check 1: robust accuracy non-increasing in eps ---
    robust_accs = []
    for eps in eps_values:
        res = a2_pgd_attack(
            X_te_q, y_te_pm, X_sv, alpha_y, intercept, kfn,
            eps=eps, T=15, R=2,
        )
        robust_accs.append(res["robust_acc"])

    for i in range(len(robust_accs) - 1):
        assert robust_accs[i] >= robust_accs[i + 1] - 0.05, (
            f"Robust acc NOT non-increasing at eps index {i}: "
            f"{robust_accs[i]:.3f} < {robust_accs[i+1]:.3f}"
        )

    # --- Check 2: very large eps -> robust acc near 0 ---
    eps_large = 5.0 * d_med
    res_large = a2_pgd_attack(
        X_te_q, y_te_pm, X_sv, alpha_y, intercept, kfn,
        eps=eps_large, T=20, R=2,
    )
    assert res_large["robust_acc"] < 0.30, (
        f"Large-eps robust acc = {res_large['robust_acc']:.3f} is too high (expected < 0.30)"
    )

    # --- Check 3: multi-step PGD >= single-step (T=1, R=1) ---
    eps_mid = 0.2 * d_med
    res_multi = a2_pgd_attack(
        X_te_q, y_te_pm, X_sv, alpha_y, intercept, kfn,
        eps=eps_mid, T=20, R=2,
    )
    res_single = a2_pgd_attack(
        X_te_q, y_te_pm, X_sv, alpha_y, intercept, kfn,
        eps=eps_mid, T=1, R=1,
    )
    # Multi-step should be at least as strong (lower or equal robust acc)
    assert res_multi["robust_acc"] <= res_single["robust_acc"] + 0.05, (
        f"Multi-step robust acc {res_multi['robust_acc']:.3f} > "
        f"single-step {res_single['robust_acc']:.3f} + tolerance"
    )
