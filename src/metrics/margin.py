import numpy as np

def compute_margin(qsvm, K_train: np.ndarray, y_train: np.ndarray) -> float:
    """Compute SVM geometric margin.
    
    Args:
        qsvm: Fitted QSVM or RBFSVM wrapper object.
        K_train: Training Gram matrix (N x N) (for QSVM) or None (for RBFSVM).
                 If RBFSVM, we can't easily extract K_train, but for this project
                 we always have precomputed kernels for QSVM.
        y_train: Labels in {0, 1} or {-1, 1}.
    """
    svm = qsvm.svm
    if not hasattr(svm, "dual_coef_"):
        return 0.0
        
    # SVM dual_coef_ is shape (1, n_SV). It contains alpha_i * y_i where y_i in {-1, 1}
    a_y = svm.dual_coef_[0]
    sv_indices = svm.support_
    
    # Extract the submatrix of the kernel for the support vectors
    if K_train is not None:
        K_sv = K_train[np.ix_(sv_indices, sv_indices)]
    else:
        # Fallback for RBF: compute kernel explicitly if needed, but the plan
        # mostly asks for margin of QSVM. Wait, geometric margin is 1 / ||w||.
        # ||w||^2 = sum_ij a_i y_i a_j y_j K_ij = a_y @ K_sv @ a_y.T
        raise ValueError("K_train must be provided for margin computation")
        
    w_sq = np.dot(a_y, np.dot(K_sv, a_y))
    if w_sq <= 0:
        return 0.0
    return 1.0 / np.sqrt(w_sq)

def compute_kta(K: np.ndarray, y: np.ndarray) -> float:
    """Compute Kernel-Target Alignment.
    
    Args:
        K: Gram matrix (N x N)
        y: Labels in {0, 1} or {-1, 1}.
    """
    y_scaled = np.where(y <= 0, -1, 1)
    Y = np.outer(y_scaled, y_scaled)
    
    inner_prod = np.sum(K * Y)
    norm_K = np.linalg.norm(K, ord='fro')
    norm_Y = np.linalg.norm(Y, ord='fro')
    
    return float(inner_prod / (norm_K * norm_Y + 1e-12))
