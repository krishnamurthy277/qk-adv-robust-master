import torch

def make_psd(K: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """Project a symmetric matrix to the nearest PSD matrix.
    
    Performs eigenvalue decomposition, clips negative eigenvalues to zero,
    and reconstructs the matrix.
    
    Args:
        K: Symmetric tensor of shape (N, N).
        epsilon: Small positive value to add to diagonal for strict positive definiteness.
        
    Returns:
        PSD tensor of shape (N, N).
    """
    K_sym = (K + K.T) / 2.0
    
    eigenvalues, eigenvectors = torch.linalg.eigh(K_sym)
    eigenvalues_clipped = torch.clamp(eigenvalues, min=0.0)
    
    K_psd = eigenvectors @ torch.diag(eigenvalues_clipped) @ eigenvectors.T
    K_psd = (K_psd + K_psd.T) / 2.0
    
    if epsilon > 0.0:
        K_psd += epsilon * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
        
    return K_psd
