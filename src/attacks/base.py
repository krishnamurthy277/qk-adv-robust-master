"""attacks/base.py — Shared utilities for adversarial attacks."""
import math
import torch
import numpy as np
from typing import Callable


def project_l2_ball(x: torch.Tensor, x0: torch.Tensor, eps: float) -> torch.Tensor:
    """Project x back into the L2 ball of radius eps centred at x0."""
    delta = x - x0
    norm = delta.norm()
    if norm > eps:
        delta = delta * (eps / (norm + 1e-12))
    return x0 + delta


def clip_to_domain(x: torch.Tensor) -> torch.Tensor:
    """Clip features to [0, pi]."""
    return x.clamp(0.0, math.pi)


def median_pairwise_l2(X: torch.Tensor) -> float:
    """Compute median pairwise L2 distance between training points."""
    N = X.shape[0]
    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            dists.append((X[i] - X[j]).norm().item())
    return float(np.median(dists)) if dists else 1.0


def decision_function(
    x: torch.Tensor,
    X_sv: torch.Tensor,
    alpha_y: torch.Tensor,
    intercept: float,
    kernel_fn: Callable,
) -> torch.Tensor:
    """Compute f(x) = alpha_y @ k(SV, x) + b using the provided kernel function.

    Args:
        x: Query point, shape (n,) or (1, n).
        X_sv: Support vectors, shape (n_sv, n).
        alpha_y: dual_coef_ * y_i, shape (n_sv,).
        intercept: SVM bias term b.
        kernel_fn: callable(X1: (N1, n), X2: (N2, n)) -> K: (N1, N2).

    Returns:
        Scalar tensor representing f(x).
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    K = kernel_fn(x, X_sv)  # (1, n_sv)
    return (alpha_y @ K[0]) + intercept
