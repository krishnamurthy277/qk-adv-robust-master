import numpy as np
from typing import Dict, Any

def compute_concentration_metrics(K: np.ndarray, n_qubits: int) -> Dict[str, float]:
    """Compute concentration and spectral metrics for a kernel matrix.
    
    Args:
        K: Gram matrix (N x N)
        n_qubits: Number of qubits
    Returns:
        Dictionary of metrics.
    """
    N = K.shape[0]
    # Off-diagonal entries
    off_mask = ~np.eye(N, dtype=bool)
    off_entries = K[off_mask]
    
    mu_K = np.mean(off_entries)
    var_K = np.var(off_entries)
    cv_K = np.sqrt(var_K) / (mu_K + 1e-12)
    
    ref = 1.0 / (2**n_qubits)
    mu_diff = mu_K - ref
    
    # Spectral metrics
    eigenvalues = np.linalg.eigvalsh(K)
    # Clip negative eigenvalues for trace and rank
    pos_eigvals = np.clip(eigenvalues, 0.0, None)
    tr_K = np.sum(pos_eigvals) + 1e-12
    
    lambda_max = np.max(pos_eigvals)
    spectral_ratio = lambda_max / tr_K
    
    q = pos_eigvals / tr_K
    # Filter q > 0 to avoid log(0)
    q_pos = q[q > 1e-12]
    eff_rank = np.exp(-np.sum(q_pos * np.log(q_pos)))
    
    return {
        "mu_K": float(mu_K),
        "var_K": float(var_K),
        "log10_var_K": float(np.log10(var_K + 1e-15)),
        "cv_K": float(cv_K),
        "mu_diff": float(mu_diff),
        "spectral_ratio": float(spectral_ratio),
        "eff_rank": float(eff_rank)
    }
