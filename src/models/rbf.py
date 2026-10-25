import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class RBFSVM:
    def __init__(self):
        self.svm = None
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1, 1.0]}
        svc = SVC(kernel="rbf")
        clf = GridSearchCV(svc, param_grid, cv=3)
        clf.fit(X_train, y_train)
        
        self.svm = clf.best_estimator_
        return self
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.svm.predict(X_test)
        
    def decision_function(self, X_test: np.ndarray) -> np.ndarray:
        return self.svm.decision_function(X_test)
