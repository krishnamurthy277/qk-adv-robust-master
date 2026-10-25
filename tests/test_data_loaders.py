"""Test dataset pipeline sanity checks."""
import pytest
from src.data.wbcd import load_wbcd
from src.data.synthetic import load_synthetic
from src.data.gse2034 import load_gse2034
from src.data.preprocessing import QKPreprocessor

def test_wbcd_loader():
    X, y = load_wbcd()
    assert X.shape[0] == 569
    assert X.shape[1] == 30
    assert set(y) == {0, 1}

def test_synthetic_loader():
    X, y = load_synthetic("classification", n_samples=100, n_features=4)
    assert X.shape == (100, 4)
    assert set(y) == {0, 1}
    
def test_gse2034_loader():
    X, y = load_gse2034()
    assert X.shape[0] == 286
    assert X.shape[1] > 20000
    assert len(y) == 286
    assert set(y) == {0, 1}
    
def test_preprocessor():
    X, y = load_synthetic("classification", n_samples=50, n_features=10)
    prep = QKPreprocessor(n_qubits=4)
    X_trans = prep.fit_transform(X)
    assert X_trans.shape == (50, 4)
    import torch
    import math
    assert torch.all(X_trans >= 0.0)
    assert torch.all(X_trans <= math.pi + 1e-5)
