from sklearn.model_selection import train_test_split
import numpy as np

def get_train_test_split(X: np.ndarray, y: np.ndarray, train_size: int = 100, test_size: int = 100, seed: int = 42):
    """Stratified subsampling.
    Ensures subsets are stratified and sized exactly as specified if data permits.
    """
    max_train = int(len(y) * 0.5)
    max_test = len(y) - max_train
    
    tr_size = min(train_size, max_train)
    te_size = min(test_size, max_test)
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, train_size=tr_size, test_size=te_size, stratify=y, random_state=seed
    )
    return X_tr, X_te, y_tr, y_te
