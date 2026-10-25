"""T9: Reproducibility — same config + seed gives identical results."""
import torch
import numpy as np
from src.kernel.fidelity import compute_kernel_matrix
from src.utils.seed import seed_everything

def test_t9_reproducibility():
    """T9: Two runs with the same seed must produce byte-identical kernel matrices."""
    n = 4
    N = 8
    reps = 2
    c = 1.0
    p = 0.01
    
    def run():
        seed_everything(42)
        X = torch.rand(N, n, dtype=torch.float64)
        K = compute_kernel_matrix(X, None, reps=reps, c=c, p=p)
        return K.detach().numpy()
    
    K1 = run()
    K2 = run()
    
    assert np.array_equal(K1, K2), "Identical seeds produced different kernel matrices"
