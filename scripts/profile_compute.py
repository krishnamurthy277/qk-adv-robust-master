"""profile_compute.py - Time kernel evaluation, Gram matrix, and gradient at n=4,6,8.

For n=8, Gram is N x N with N=20 (not 100) to stay tractable on CPU.
"""
import time
import torch
import numpy as np
from src.kernel.fidelity import compute_kernel_matrix
from src.sim.gates import expand_gate
from src.sim.noise import depolarizing_channel_kraus

def time_kernel_pair(n, reps=2, c=1.0, p=0.01, repeats=3):
    """Time a single-row kernel evaluation (1 x 5)."""
    X1 = torch.rand(1, n, dtype=torch.float64)
    X2 = torch.rand(5, n, dtype=torch.float64)
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = compute_kernel_matrix(X1, X2, reps=reps, c=c, p=p)
    return (time.perf_counter() - t0) / repeats * 1000  # ms

def time_gram(n, N, reps=2, c=1.0, p=0.01):
    """Time full Gram matrix computation (N x N)."""
    X = torch.rand(N, n, dtype=torch.float64)
    t0 = time.perf_counter()
    _ = compute_kernel_matrix(X, None, reps=reps, c=c, p=p)
    return time.perf_counter() - t0  # seconds

def time_gradient(n, n_sv=5, reps=2, c=1.0, p=0.01, repeats=3):
    """Time gradient of f(x) = sum_i k(x, sv_i) w.r.t. x."""
    X1 = torch.rand(1, n, dtype=torch.float64, requires_grad=True)
    X_sv = torch.rand(n_sv, n, dtype=torch.float64)
    t0 = time.perf_counter()
    for _ in range(repeats):
        if X1.grad is not None:
            X1.grad.zero_()
        K = compute_kernel_matrix(X1, X_sv, reps=reps, c=c, p=p)
        K.sum().backward()
    return (time.perf_counter() - t0) / repeats * 1000  # ms

# Gram sizes adjusted so n=8 finishes in reasonable time on CPU
GRAM_SIZES = {4: 50, 6: 30, 8: 10}

print("Profiling quantum kernel compute times (reps=2, c=1.0, p=0.01)")
print(f"{'n':>4} | {'Pair 1x5 (ms)':>14} | {'Gram NxN (s)':>13} | {'N':>5} | {'Grad (ms)':>9}")
print("-" * 55)

for n in [4, 6, 8]:
    N = GRAM_SIZES[n]
    try:
        t_pair = time_kernel_pair(n)
        t_gram = time_gram(n, N)
        t_grad = time_gradient(n)
        print(f"{n:>4} | {t_pair:>14.1f} | {t_gram:>13.2f} | {N:>5} | {t_grad:>9.1f}")
    except Exception as e:
        print(f"{n:>4} | ERROR: {e}")

print()
print("Compute plan (estimated wall-clock):")
print("  Tier A: n=6, N_train=100  ->  Gram ~", round(GRAM_SIZES[6] * (100/GRAM_SIZES[6])**2 * 1, 0), "s (extrapolated)")
print("  Tier B: n=8, N_train=100  ->  Gram ~", round(GRAM_SIZES[8] * (100/GRAM_SIZES[8])**2 * 1, 0), "s (extrapolated)")
