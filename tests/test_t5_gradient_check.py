"""T5: Gradient check — autograd vs central finite differences."""
import torch
import math
import numpy as np
import pytest
from src.kernel.fidelity import compute_kernel_matrix

@pytest.mark.parametrize('p', [0.0, 0.03])
def test_t5_gradient_check(p):
    """T5: Autograd gradient of decision function vs central finite differences.
    
    Relative error must be < 1e-5 on 10 random points.
    """
    torch.manual_seed(42)
    n = 4
    n_sv = 5
    h = 1e-5
    
    # Random support vectors (fixed) and dual coefficients
    X_sv = torch.rand(n_sv, n, dtype=torch.float64)
    alpha_y = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0], dtype=torch.float64)

    def f(x: torch.Tensor) -> torch.Tensor:
        """Decision function f(x) = alpha_y @ k(SV, x)."""
        K = compute_kernel_matrix(x.unsqueeze(0), X_sv, reps=2, c=1.0, p=p)
        return (alpha_y @ K[0])
    
    max_rel_err = 0.0
    for _ in range(10):
        x0 = torch.rand(n, dtype=torch.float64)
        
        # Autograd gradient
        x_ad = x0.clone().detach().requires_grad_(True)
        fx = f(x_ad)
        fx.backward()
        grad_ad = x_ad.grad.clone()
        
        # Central finite differences
        grad_fd = torch.zeros_like(x0)
        for i in range(n):
            xp = x0.clone(); xp[i] += h
            xm = x0.clone(); xm[i] -= h
            grad_fd[i] = (f(xp) - f(xm)) / (2.0 * h)
        
        # Relative error
        norm_ad = grad_ad.norm()
        norm_diff = (grad_ad - grad_fd).norm()
        rel_err = (norm_diff / (norm_ad + 1e-12)).item()
        max_rel_err = max(max_rel_err, rel_err)
    
    assert max_rel_err < 1e-5, f"Max relative gradient error {max_rel_err:.2e} >= 1e-5"
