"""Noiseless fidelity: five pairs at both required system sizes."""
import math
import numpy as np
import pytest
import torch
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from src.kernel.fidelity import compute_kernel_matrix

@pytest.mark.parametrize('n', [4,6])
def test_t2_vs_qiskit_noiseless(n):
    rng = np.random.default_rng(41)
    fm = ZZFeatureMap(feature_dimension=n,reps=2,entanglement='linear')
    errors=[]
    for _ in range(5):
        x,y = rng.uniform(0,math.pi,(2,n))
        a=Statevector(fm.assign_parameters(x)).data
        b=Statevector(fm.assign_parameters(y)).data
        actual=compute_kernel_matrix(torch.tensor(x)[None],torch.tensor(y)[None],reps=2).item()
        errors.append(abs(actual-abs(np.vdot(a,b))**2))
    assert max(errors)<1e-8, errors
