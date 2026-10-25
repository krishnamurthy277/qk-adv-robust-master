import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.kernel.gram import get_gram_matrix

class QSVM:
    def __init__(self, reps: int=2, c: float=1.0, p: float=0.0):
        self.reps = reps
        self.c = c
        self.p = p
        self.svm = None
        self.X_train = None
        
    def fit(self, X_train: torch.Tensor, y_train: np.ndarray):
        """Fit the QSVM using CV for C parameter on the precomputed gram matrix."""
        self.X_train = X_train
        K_train = get_gram_matrix(X_train, reps=self.reps, c=self.c, p=self.p).cpu().numpy()
        
        param_grid = {'C': [0.1, 1, 10, 100]}
        svc = SVC(kernel="precomputed")
        clf = GridSearchCV(svc, param_grid, cv=3)
        clf.fit(K_train, y_train)
        
        self.svm = clf.best_estimator_
        return self
        
    def predict(self, X_test: torch.Tensor) -> np.ndarray:
        K_test = get_gram_matrix(X_test, self.X_train, reps=self.reps, c=self.c, p=self.p).cpu().numpy()
        return self.svm.predict(K_test)
        
    def decision_function(self, X_test: torch.Tensor) -> np.ndarray:
        K_test = get_gram_matrix(X_test, self.X_train, reps=self.reps, c=self.c, p=self.p).cpu().numpy()
        return self.svm.decision_function(K_test)
