"""Checks for class-balanced attack scoring and the spectral diagnostic."""
import numpy as np
import torch
import pytest
from scripts.publication_followup import balanced
from scripts.review_diagnostics import decision_bounds

def test_majority_prediction_is_half_balanced_despite_high_accuracy():
    y=np.array([-1]*9+[1]); correct=y==-1
    assert correct.mean()==.9
    assert balanced(correct,y)==.5
    assert balanced(np.zeros(10,dtype=bool),y)==0

def test_spectral_bounds_contain_scores_of_mixed_states():
    rho=torch.tensor([[[1.,0.],[0.,0.]],[[0.,0.],[0.,1.]]],dtype=torch.complex128)
    coefficients=torch.tensor([2.,-1.],dtype=torch.float64)
    lo,hi=decision_bounds(rho,coefficients,.25)
    assert (lo,hi)==pytest.approx((-.75,2.25))
    for t in np.linspace(0,1,31):
        score=2*t-(1-t)+.25
        assert lo-1e-12<=score<=hi+1e-12
    lo,hi=decision_bounds(rho,coefficients,2.)
    assert lo>0 and hi>0
