import torch
import math
from src.sim.gates import H, RZ, expand_gate, expand_two_qubit_gate

def zz_feature_map_circuit(x: torch.Tensor, reps: int, c: float = 1.0) -> torch.Tensor:
    """Construct the batched unitary for the ZZ Feature Map.
    
    Args:
        x: Tensor of shape (B, n_qubits).
        reps: Number of repetitions.
        c: Scaling factor for data.
        
    Returns:
        Tensor of shape (B, 2^n, 2^n) representing the full unitary.
    """
    B, n = x.shape
    device = x.device
    dim = 2**n
    
    U_total = torch.eye(dim, dtype=torch.complex128, device=device).unsqueeze(0).expand(B, -1, -1)
    
    def apply_layer(U: torch.Tensor, gate_layer: torch.Tensor) -> torch.Tensor:
        return torch.bmm(gate_layer, U)
        
    H_all = H
    for _ in range(n - 1):
        H_all = torch.kron(H_all, H)
    H_all = H_all.to(device).unsqueeze(0).expand(B, -1, -1)
    
    for r in range(reps):
        U_total = apply_layer(U_total, H_all)
        
        for i in range(n):
            angle = 2.0 * c * x[:, i]
            rz_gate = RZ(angle)
            U_rz = expand_gate(rz_gate, n, i)
            U_total = apply_layer(U_total, U_rz)
            
        for i in range(n - 1):
            angle = 2.0 * (math.pi - c * x[:, i]) * (math.pi - c * x[:, i+1])
            half = angle / 2.0
            zz_gate = torch.zeros((B, 4, 4), dtype=torch.complex128, device=device)
            zz_gate[:, 0, 0] = torch.exp(-1j * half)
            zz_gate[:, 1, 1] = torch.exp(1j * half)
            zz_gate[:, 2, 2] = torch.exp(1j * half)
            zz_gate[:, 3, 3] = torch.exp(-1j * half)
            
            U_zz = expand_two_qubit_gate(zz_gate, n, i, i+1)
            U_total = apply_layer(U_total, U_zz)
            
    return U_total
