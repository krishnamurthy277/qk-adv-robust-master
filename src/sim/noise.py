import torch
from .gates import I, X, Y, Z

def depolarizing_channel_kraus(p: float, num_qubits: int = 1) -> torch.Tensor:
    """Return Kraus operators for the depolarizing channel on num_qubits.
    
    The channel is E(rho) = (1 - p) * rho + (p / (4^k - 1)) * sum_{i=1}^{4^k-1} P_i rho P_i
    where P_i are Pauli strings excluding the all-identity string.
    
    Returns:
        Tensor of shape (4^num_qubits, 2^num_qubits, 2^num_qubits) containing Kraus operators.
    """
    paulis = [I, X, Y, Z]
    
    def generate_strings(k):
        if k == 1:
            return paulis
        smaller = generate_strings(k - 1)
        res = []
        for p1 in paulis:
            for p2 in smaller:
                res.append(torch.kron(p1, p2))
        return res
        
    strings = generate_strings(num_qubits)
    
    dim = 4**num_qubits
    kraus = torch.zeros((dim, 2**num_qubits, 2**num_qubits), dtype=torch.complex128)
    
    kraus[0] = torch.sqrt(torch.tensor(1.0 - p, dtype=torch.float64)) * strings[0]
    coef = torch.sqrt(torch.tensor(p / (dim - 1.0), dtype=torch.float64))
    
    for i in range(1, dim):
        kraus[i] = coef * strings[i]
        
    return kraus
