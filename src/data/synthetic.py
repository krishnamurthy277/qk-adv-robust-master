import numpy as np
from sklearn.datasets import make_classification, make_moons

def load_synthetic(name: str = "classification", n_samples: int = 200, n_features: int = 10, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic datasets.
    
    Args:
        name: "classification" or "moons"
        n_samples: Number of samples.
        n_features: Number of features (only for classification).
        random_state: Seed.
    """
    if name == "classification":
        from sklearn.datasets import make_blobs
        c1 = [2.0] * n_features
        c2 = [-2.0] * n_features
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=[c1, c2], cluster_std=1.0, random_state=random_state)
    elif name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=random_state)
    else:
        raise ValueError(f"Unknown synthetic dataset: {name}")
        
    return X, y
