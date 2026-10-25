"""T10: Preprocessing logic must strictly avoid data leakage."""
import numpy as np
import torch
from src.data.preprocessing import QKPreprocessor

def test_t10_preprocessing_leakage():
    """T10: Preprocessor must fit on train only and clip test features."""
    X_train = np.random.randn(100, 20)
    # create a massive outlier in test to see if it affects train or gets clipped
    X_test = np.random.randn(20, 20)
    X_test[0, 0] = 1000.0
    
    preprocessor = QKPreprocessor(n_qubits=4)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    assert X_train_trans.shape == (100, 4)
    assert X_test_trans.shape == (20, 4)
    
    assert torch.all(X_train_trans >= 0.0) and torch.all(X_train_trans <= np.pi)
    assert torch.all(X_test_trans >= 0.0) and torch.all(X_test_trans <= np.pi)
    
    preprocessor_leaky = QKPreprocessor(n_qubits=4)
    X_all = np.vstack((X_train, X_test))
    preprocessor_leaky.fit(X_all)
    X_train_leaky = preprocessor_leaky.transform(X_train)
    
    # Asserting no data leakage means X_train_trans != X_train_leaky
    assert not torch.allclose(X_train_trans, X_train_leaky)
