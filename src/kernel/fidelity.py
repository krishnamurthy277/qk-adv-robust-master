"""Exact per-gate noisy compute-uncompute kernel.

Depolarization on a gate's full support is self-adjoint and commutes with
that gate. The noisy inverse half is thus the adjoint of the forward
channel: k(x,y) = Tr(rho_x rho_y). This simulates each input once while
preserving noise after every gate in BOTH halves. Tested independently
against an explicit Qiskit noisy compute-uncompute circuit.
"""
import math
import torch
from src.sim.gates import H, CNOT, RZ

KERNEL_VERSION = "per_gate_depolarizing_v2"


def local_unitary(rho, gate, targets, n):
    """Apply a local gate without expanding full-system unitaries."""
    batch = rho.shape[0]
    state = rho.reshape((batch,) + (2,) * (2*n))
    for axes, matrix in (([1+q for q in targets], gate),
                         ([1+n+q for q in targets], gate.conj())):
        rest = [i for i in range(1, 2*n+1) if i not in axes]
        perm = [0] + axes + rest
        inv = [perm.index(i) for i in range(2*n+1)]
        moved = state.permute(perm).reshape(batch, 2**len(targets), -1)
        moved = torch.matmul(matrix, moved)
        state = moved.reshape((batch,) + (2,)*(2*n)).permute(inv)
    return state.reshape(batch, 2**n, 2**n)


def local_depolarize(rho, targets, n, probability):
    """Exact Pauli channel via partial trace, lambda=p*4^k/(4^k-1)."""
    if probability == 0:
        return rho
    batch = rho.shape[0]
    other = [q for q in range(n) if q not in targets]
    perm = [0] + [1+q for q in targets] + [1+q for q in other]
    perm += [1+n+q for q in targets] + [1+n+q for q in other]
    inv = [perm.index(i) for i in range(2*n+1)]
    d, rest = 2**len(targets), 2**len(other)
    grouped = rho.reshape((batch,) + (2,)*(2*n)).permute(perm).reshape(batch,d,rest,d,rest)
    reduced = grouped.diagonal(dim1=1, dim2=3).sum(-1)
    identity = torch.eye(d, dtype=rho.dtype, device=rho.device) / d
    mixed = torch.einsum("ac,brs->barcs", identity, reduced)
    mixed = mixed.reshape((batch,) + (2,)*(2*n)).permute(inv).reshape_as(rho)
    lam = probability * d*d / (d*d-1)
    return (1-lam)*rho + lam*mixed


def feature_density(X, reps=2, c=1.0, p=0.0):
    """Forward ZZ density; RZ angles match Qiskit's P(2*angle)."""
    if X.ndim != 2 or X.dtype != torch.float64:
        raise ValueError("Expected float64 inputs of shape (batch, n_qubits)")
    if not 0 <= p <= 1 or reps < 1:
        raise ValueError("Require 0 <= p <= 1 and reps >= 1")
    batch,n = X.shape
    rho = torch.zeros((batch,2**n,2**n), dtype=torch.complex128, device=X.device)
    rho[:,0,0] = 1
    def apply(state, gate, targets):
        state = local_unitary(state, gate, targets, n)
        return local_depolarize(state, targets, n, p if len(targets)==2 else p/10)
    for _ in range(reps):
        for q in range(n):
            rho = apply(rho, H.to(X.device), [q])
        for q in range(n):
            rho = apply(rho, RZ(2*c*X[:,q]), [q])
        for q in range(n-1):
            rho = apply(rho, CNOT.to(X.device), [q,q+1])
            rho = apply(rho, RZ(2*(math.pi-c*X[:,q])*(math.pi-c*X[:,q+1])), [q+1])
            rho = apply(rho, CNOT.to(X.device), [q,q+1])
    return rho


def density_overlap(left, right):
    return (left.flatten(1) @ right.flatten(1).conj().T).real


def compute_kernel_matrix(X1, X2=None, reps=2, c=1.0, p=0.0):
    left = feature_density(X1, reps, c, p)
    right = left if X2 is None else feature_density(X2, reps, c, p)
    return density_overlap(left, right)


def fixed_reference_kernel(reference, reps=2, c=1.0, p=0.0):
    """Cache only fixed support features; retain full input gradients."""
    with torch.no_grad():
        right = feature_density(reference, reps, c, p)
    def kernel(X, X_ref):
        if X_ref is not reference:
            raise ValueError("Kernel closure must use its original fixed reference")
        return density_overlap(feature_density(X, reps, c, p), right)
    return kernel
