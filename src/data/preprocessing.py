import numpy as np
import torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

class QKPreprocessor:
    """Preprocesses data for quantum kernel methods.
    
    Pipeline: StandardScaler -> PCA -> MinMaxScaler(0, pi).
    STRICT REQUIREMENT: test features MUST be clipped to [0, pi] if they exceed.
    """
    def __init__(self, n_qubits: int, select_k: int = None):
        self.n_qubits = n_qubits
        selection = [('selection', SelectKBest(f_classif, k=select_k))] if select_k else []
        self.pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))] + selection + [
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_qubits, svd_solver='full')),
            ('minmax', MinMaxScaler(feature_range=(0, np.pi)))
        ])
        
    def fit(self, X_train: np.ndarray, y_train=None):
        self.pipeline.fit(X_train, y_train)
        return self
        
    def transform(self, X: np.ndarray) -> torch.Tensor:
        X_trans = self.pipeline.transform(X)
        # Strict clipping to [0, pi]
        X_trans = np.clip(X_trans, 0.0, np.pi)
        return torch.tensor(X_trans, dtype=torch.float64)
        
    def fit_transform(self, X_train: np.ndarray, y_train=None) -> torch.Tensor:
        self.fit(X_train, y_train)
        return self.transform(X_train)
