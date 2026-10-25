import numpy as np
import pandas as pd
import pytest
from src.analysis.statistics import bootstrap_mean,benjamini_hochberg,partial_spearman
from scripts.run_classical_controls import matched_gamma
from scripts.analyze_results import validate_table
from sklearn.metrics.pairwise import rbf_kernel


def test_rank_controls_remove_shared_linear_rank_confounder():
    z=np.arange(10,dtype=float)
    assert np.isnan(partial_spearman(z,z,z))


def test_bootstrap_preserves_constant_effect_and_single_seed_uncertainty():
    assert bootstrap_mean([.1,.1,.1])==pytest.approx((.1,.1,.1))
    mean,low,high=bootstrap_mean([.2])
    assert mean==.2 and np.isnan(low) and np.isnan(high)


def test_bh_known_example_and_missing_values():
    np.testing.assert_allclose(benjamini_hochberg([.01,.04,.03,np.nan]),[.03,.04,.04,np.nan])


def test_gamma_matching_achieves_training_variance():
    X=np.random.default_rng(22).normal(size=(25,3))
    K=rbf_kernel(X,gamma=.02)
    target=K[~np.eye(len(X),dtype=bool)].var()
    gamma,variance,roots=matched_gamma(X,target)
    assert roots>=1
    assert gamma==pytest.approx(.02,rel=1e-6)
    assert variance==pytest.approx(target,rel=1e-8)


def test_analysis_rejects_incomplete_grid():
    cfg=dict(datasets=['wbcd'],seeds=[0],reps=[1],noise_p=[0.],bandwidth=[1.],eps_rel=[.1,.2])
    frame=pd.DataFrame([dict(dataset='wbcd',seed=0,reps=1,noise_p=0.,bandwidth=1.,eps_rel=.1)])
    with pytest.raises(ValueError,match='Incomplete'):
        validate_table(frame,cfg)
