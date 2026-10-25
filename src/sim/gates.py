import torch

# Standard single-qubit gates
I = torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128)
H = (2.0 ** -0.5) * torch.tensor([[1, 1], [1, -1]], dtype=torch.complex128)
X = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
Y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)

# CNOT gate
CNOT = torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=torch.complex128)

def RZ(theta: torch.Tensor) -> torch.Tensor:
    """Return a batched RZ gate.
    
    Args:
        theta: Tensor of shape (B,) representing angles.
        
    Returns:
        Tensor of shape (B, 2, 2) representing the RZ gates.
    """
    # RZ(theta) = diag(e^{-i theta/2}, e^{i theta/2})
    half_theta = theta / 2.0
    B = theta.shape[0]
    gate = torch.zeros((B, 2, 2), dtype=torch.complex128, device=theta.device)
    gate[:, 0, 0] = torch.exp(-1j * half_theta)
    gate[:, 1, 1] = torch.exp(1j * half_theta)
    return gate

def batched_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched kronecker product of two 3D tensors.
    
    Args:
        a: Tensor of shape (B, N, M)
        b: Tensor of shape (B, P, Q)
        
    Returns:
        Tensor of shape (B, N*P, M*Q)
    """
    B, N, M = a.shape
    _, P, Q = b.shape
    return torch.einsum('bij,bkl->bikjl', a, b).reshape(B, N*P, M*Q)

def expand_gate(gate: torch.Tensor, n_qubits: int, target: int, batch_size: int = None) -> torch.Tensor:
    """Expand a single qubit gate to the full n_qubits Hilbert space for a batch.
    
    Args:
        gate: Gate tensor, shape (2, 2) or (B, 2, 2).
        n_qubits: Total number of qubits.
        target: Target qubit index (0-indexed).
        batch_size: If gate is unbatched, specify the batch size to broadcast to.
        
    Returns:
        Expanded gate of shape (B, 2^n, 2^n).
    """
    if gate.dim() == 2:
        if batch_size is None:
            batch_size = 1
        gate = gate.unsqueeze(0).expand(batch_size, -1, -1)
    
    B = gate.shape[0]
    device = gate.device
    I_batched = I.unsqueeze(0).expand(B, -1, -1).to(device)
    
    mats = [I_batched] * n_qubits
    mats[target] = gate
    
    res = mats[0]
    for m in mats[1:]:
        res = batched_kron(res, m)
    return res

def expand_two_qubit_gate(gate: torch.Tensor, n_qubits: int, control: int, target: int, batch_size: int = None) -> torch.Tensor:
    """Expand a two-qubit gate to the full n_qubits Hilbert space.
    
    Currently assumes control and target are adjacent (control + 1 == target).
    """
    if target != control + 1:
        raise NotImplementedError("Only adjacent qubits (i, i+1) supported for two-qubit gates.")
        
    if gate.dim() == 2:
        if batch_size is None:
            batch_size = 1
        gate = gate.unsqueeze(0).expand(batch_size, -1, -1)
        
    B = gate.shape[0]
    device = gate.device
    I_batched = I.unsqueeze(0).expand(B, -1, -1).to(device)
    
    mats = [I_batched] * n_qubits
    mats[control] = gate
    del mats[control+1] # Gate covers both control and target
    
    res = mats[0]
    for m in mats[1:]:
        res = batched_kron(res, m)
    return res
