from sklearn.datasets import load_breast_cancer
import numpy as np

def load_wbcd() -> tuple[np.ndarray, np.ndarray]:
    """Load WBCD dataset.
    
    Returns:
        X: Feature matrix.
        y: Labels (1 = Malignant, 0 = Benign).
    """
    data = load_breast_cancer()
    # sklearn breast_cancer: 0 = malignant, 1 = benign.
    # We invert it so that 1 = malignant (the positive class)
    y = 1 - data.target
    return data.data, y
