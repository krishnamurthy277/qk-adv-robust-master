import numpy as np
import torch
from scripts.review_diagnostics import decision_bounds
from src.kernel.fidelity import feature_density,density_overlap


def test_density_decisions_respect_eigenvalue_bounds():
    torch.manual_seed(91)
    train=torch.rand(8,3,dtype=torch.float64)*np.pi
    test=torch.rand(20,3,dtype=torch.float64)*np.pi
    weights=torch.randn(8,dtype=torch.float64)
    rho=feature_density(train,reps=2,p=.03)
    other=feature_density(test,reps=2,p=.03)
    lower,upper=decision_bounds(rho,weights,.25)
    values=density_overlap(other,rho)@weights+.25
    assert torch.all(values>=lower-1e-12)
    assert torch.all(values<=upper+1e-12)


def test_zero_operator_constant_sign_is_not_attack_failure():
    rho=torch.eye(2,dtype=torch.complex128)[None]/2
    lower,upper=decision_bounds(rho,torch.zeros(1,dtype=torch.float64),-1.)
    assert lower==upper==-1.
