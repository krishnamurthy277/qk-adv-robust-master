"""T6: Noise behaviour — concentration toward 1/2^n and variance reduction as p increases."""
import torch
import numpy as np
from src.kernel.fidelity import compute_kernel_matrix

def test_t6_noise_behaviour():
    """T6: Off-diagonal mean moves toward 1/2^n and variance shrinks as p increases.
    Diagonal < 1 under noise.
    """
    torch.manual_seed(0)
    n = 4
    N = 10
    reps = 2
    c = 1.0
    ref = 1.0 / (2**n)  # = 0.0625
    
    X = torch.rand(N, n, dtype=torch.float64)
    
    off_mask = ~np.eye(N, dtype=bool)
    
    # Noiseless
    K0 = compute_kernel_matrix(X, None, reps=reps, c=c, p=0.0).detach().numpy()
    off0 = K0[off_mask]
    mu0 = off0.mean()
    var0 = off0.var()
    diag0 = np.diag(K0)
    
    # T1 sanity: diagonal should be 1 noiseless
    assert np.allclose(diag0, 1.0, atol=1e-8), f"Noiseless diagonal not 1: {diag0}"
    
    prev_mu_diff = abs(mu0 - ref)
    prev_var = var0
    
    for p in [0.01, 0.05, 0.1, 0.2]:
        Kp = compute_kernel_matrix(X, None, reps=reps, c=c, p=p).detach().numpy()
        off_p = Kp[off_mask]
        mu_p = off_p.mean()
        var_p = off_p.var()
        diag_p = np.diag(Kp)
        
        # Off-diagonal mean moves closer to 1/2^n
        mu_diff = abs(mu_p - ref)
        assert mu_diff <= prev_mu_diff + 1e-6, (
            f"p={p}: |mu - ref| = {mu_diff:.4f} did not decrease (was {prev_mu_diff:.4f})"
        )
        
        # Variance shrinks
        assert var_p <= prev_var + 1e-8, (
            f"p={p}: var = {var_p:.6f} did not decrease (was {prev_var:.6f})"
        )
        
        # Diagonal < 1 under noise
        assert np.all(diag_p < 1.0), f"p={p}: Some diagonal entries >= 1: {diag_p}"
        
        prev_mu_diff = mu_diff
        prev_var = var_p
