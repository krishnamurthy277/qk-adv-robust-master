"""T1: Noiseless kernel basics."""

import torch
import math
from src.kernel.fidelity import compute_kernel_matrix

def test_t1_diagonal_is_one():
    """T1.1: K(x, x) == 1 for all x in [0, pi]^n."""
    n = 4
    N = 10
    X = torch.rand((N, n), dtype=torch.float64) * math.pi
    
    K = compute_kernel_matrix(X)
    diag = torch.diag(K)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)

def test_t1_bounds():
    """T1.2: 0 <= K(x, y) <= 1."""
    n = 4
    N1, N2 = 5, 5
    X1 = torch.rand((N1, n), dtype=torch.float64) * math.pi
    X2 = torch.rand((N2, n), dtype=torch.float64) * math.pi
    
    K = compute_kernel_matrix(X1, X2)
    assert torch.all(K >= -1e-5)
    assert torch.all(K <= 1.0 + 1e-5)
