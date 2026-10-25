"""T7: Baseline sanity."""
import numpy as np
import torch
from src.data.wbcd import load_wbcd
from src.data.synthetic import load_synthetic
from src.data.preprocessing import QKPreprocessor
from src.data.split import get_train_test_split
from src.models.rbf import RBFSVM
from src.models.qsvm import QSVM

def test_t7_baseline_sanity():
    # RBF-SVM on WBCD (PCA to 6 dims) >= 0.93
    X, y = load_wbcd()
    X_tr, X_te, y_tr, y_te = get_train_test_split(X, y, train_size=100, test_size=100, seed=42)
    prep = QKPreprocessor(n_qubits=6)
    X_tr_trans = prep.fit_transform(X_tr).numpy()
    X_te_trans = prep.transform(X_te).numpy()
    
    rbf = RBFSVM().fit(X_tr_trans, y_tr)
    acc_rbf = np.mean(rbf.predict(X_te_trans) == y_te)
    assert acc_rbf >= 0.93
    
    # Noiseless QSVM on synthetic moons >= 0.80
    X, y = load_synthetic("moons", n_samples=200, n_features=2)
    X_tr, X_te, y_tr, y_te = get_train_test_split(X, y, train_size=100, test_size=100, seed=42)
    prep_q = QKPreprocessor(n_qubits=2)
    X_tr_q = prep_q.fit_transform(X_tr)
    X_te_q = prep_q.transform(X_te)
    
    qsvm = QSVM(reps=2, c=1.0, p=0.0).fit(X_tr_q, y_tr)
    acc_qsvm = np.mean(qsvm.predict(X_te_q) == y_te)
    assert acc_qsvm >= 0.80
