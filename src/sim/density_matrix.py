import torch
from .gates import expand_gate

class BatchedDensityMatrixSimulator:
    """Batched exact density-matrix simulator in PyTorch."""
    
    def __init__(self, n_qubits: int, batch_size: int, device: str = "cpu"):
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.dim = 2**n_qubits
        self.rho = None
        self.reset()
        
    def reset(self):
        """Reset to the |0...0><0...0| state for the whole batch."""
        state = torch.zeros((self.batch_size, self.dim, self.dim), dtype=torch.complex128, device=self.device)
        state[:, 0, 0] = 1.0
        self.rho = state
        
    def apply_unitary(self, u: torch.Tensor):
        """Apply a batched unitary u to the density matrices.
        
        Args:
            u: Tensor of shape (B, 2^n, 2^n)
        """
        u_dagger = u.conj().transpose(-2, -1)
        # u @ rho @ u^dagger
        self.rho = torch.bmm(torch.bmm(u, self.rho), u_dagger)
        
    def apply_expanded_channel(self, expanded_kraus: list[torch.Tensor]):
        """Apply a quantum channel using pre-expanded Kraus operators for speed."""
        new_rho = torch.zeros_like(self.rho)
        for E_i in expanded_kraus:
            E_i_dagger = E_i.conj().transpose(-2, -1)
            temp = torch.matmul(E_i, self.rho)
            new_rho += torch.matmul(temp, E_i_dagger)
        self.rho = new_rho
        
    def apply_channel(self, kraus: torch.Tensor, target_qubits: list[int]):
        """Apply a quantum channel defined by Kraus operators to specific qubits."""
        K = kraus.shape[0]
        
        if len(target_qubits) == 1:
            t = target_qubits[0]
            expanded_kraus = []
            for i in range(K):
                E_i = expand_gate(kraus[i], self.n_qubits, t, batch_size=1).squeeze(0).to(self.device)
                expanded_kraus.append(E_i)
        else:
            raise NotImplementedError("Only single-qubit channels supported for target_qubits expanding.")
            
        self.apply_expanded_channel(expanded_kraus)
        
    def expectation_value(self, observable: torch.Tensor) -> torch.Tensor:
        """Compute the expectation value Tr(rho * O) for the batch.
        
        Args:
            observable: Unbatched tensor (2^n, 2^n).
            
        Returns:
            Tensor of shape (B,) with real expectation values.
        """
        obs = observable.to(self.device)
        # Tr(rho * O) for each in batch
        temp = torch.matmul(self.rho, obs)
        # Trace is sum of diagonal
        b_trace = torch.diagonal(temp, dim1=-2, dim2=-1).sum(-1)
        return b_trace.real
