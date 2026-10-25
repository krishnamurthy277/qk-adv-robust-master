"""Independent gate-by-gate Qiskit Aer validation; no circuit optimization."""
import math
import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from src.kernel.fidelity import compute_kernel_matrix


def explicit_circuit(x, reps, c):
    qc = QuantumCircuit(len(x))
    for _ in range(reps):
        for q in range(len(x)): qc.h(q)
        for q in range(len(x)): qc.rz(2*c*x[q], q)
        for q in range(len(x)-1):
            qc.cx(q, q+1)
            qc.rz(2*(math.pi-c*x[q])*(math.pi-c*x[q+1]), q+1)
            qc.cx(q, q+1)
    return qc


@pytest.mark.parametrize('reps,c,p', [(1,1.0,.002),(2,.5,.01),(3,1.0,.03)])
def test_t3_vs_qiskit_noisy(reps,c,p):
    rng = np.random.default_rng(42)
    n = 4
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error((p/10)*4/3,1), ['h','rz'])
    model.add_all_qubit_quantum_error(depolarizing_error(p*16/15,2), ['cx'])
    sim = AerSimulator(method='density_matrix', noise_model=model)
    errors = []
    for j in range(3):
        x,y = rng.uniform(0,math.pi,(2,n))
        if j == 2: y = x.copy()
        qc = explicit_circuit(x,reps,c).compose(explicit_circuit(y,reps,c).inverse())
        qc.save_density_matrix()
        # Deliberately no transpilation: preserve every H/RZ/CX and its noise.
        dm = np.asarray(sim.run(qc,shots=1).result().data(0)['density_matrix'])
        actual = compute_kernel_matrix(torch.tensor(x)[None],torch.tensor(y)[None],reps,c,p).item()
        errors.append(abs(actual-dm[0,0].real))
    assert max(errors) < 1e-6, errors
