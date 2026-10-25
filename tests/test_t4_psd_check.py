"""T4: Gram-matrix PSD check."""

import torch
from src.kernel.psd_repair import make_psd

def test_t4_psd_repair():
    """T4: PSD projection removes negative eigenvalues and maintains symmetry."""
    N = 5
    A = torch.randn(N, N, dtype=torch.float64)
    A = (A + A.T) / 2.0
    A[0, 0] = -10.0 # Force negative eigenvalue
    
    A_psd = make_psd(A, epsilon=0.0)
    
    assert torch.allclose(A_psd, A_psd.T)
    eigenvalues = torch.linalg.eigvalsh(A_psd)
    assert torch.all(eigenvalues >= -1e-10)
