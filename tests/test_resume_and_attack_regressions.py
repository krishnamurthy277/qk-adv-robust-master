import numpy as np
import pandas as pd
import torch
from src.attacks.a2_pgd import pgd_l2_single
from src.attacks.a1_transfer import a1_transfer_attack
from src.kernel.gram import get_gram_matrix
from scripts.run_experiment import valid_cell, RESULTS_VERSION


def test_transfer_uses_target_denominator_and_never_repairs_clean_errors(monkeypatch):
    def surrogate(*args, **kwargs):
        return {'X_adv': torch.tensor([[1.5],[.5]],dtype=torch.float64),
                'initially_correct': np.array([True,False])}
    monkeypatch.setattr('src.attacks.a1_transfer.a2_pgd_attack',surrogate)
    X=torch.tensor([[.5],[1.5]],dtype=torch.float64)
    coefficient=torch.ones(1,dtype=torch.float64)
    kernel=lambda x,sv:x
    result=a1_transfer_attack(X,np.ones(2),X,coefficient,0,kernel,
                              X,coefficient,-1,kernel,eps=1)
    np.testing.assert_array_equal(result['initially_correct'],[False,True])
    assert result['robust_acc']==0
    assert result['attack_success_rate']==1


def test_pgd_keeps_successful_intermediate_when_final_iterate_recovers():
    kernel=lambda x,sv:(x-1)**2-.1
    x,loss=pgd_l2_single(torch.tensor([2.],dtype=torch.float64),1,None,
                         torch.ones(1,dtype=torch.float64),0,kernel,
                         eps=2,alpha=.75,T=2,R=1)
    assert loss<0
    assert abs(x.item()-1.25)<1e-12


def test_cache_distinguishes_inputs_and_qubit_count(tmp_path):
    for n in (2,3):
        X=torch.rand(3,n,dtype=torch.float64)
        get_gram_matrix(X,reps=1,cache_dir=tmp_path,config_dict={'legacy':1})
        get_gram_matrix(X+.1,reps=1,cache_dir=tmp_path,config_dict={'legacy':1})
    assert len(list(tmp_path.glob('*.npy')))==4


def test_partial_cell_is_not_complete(tmp_path):
    path=tmp_path/'cell.csv'
    cfg={'eps_rel':[.1,.2],'_source_hash':'source'}
    row=dict(eps_rel=.1,version=RESULTS_VERSION,source_hash='source',
             clean_acc=.8,a1_robust_acc=.7,a2_robust_acc=.6,var_K=.1)
    pd.DataFrame([row]).to_csv(path,index=False)
    assert not valid_cell(path,cfg)
    pd.DataFrame([row,{**row,'eps_rel':.2}]).to_csv(path,index=False)
    assert valid_cell(path,cfg)
    cfg['_source_hash']='changed'
    assert not valid_cell(path,cfg)
