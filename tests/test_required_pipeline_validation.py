"""Close legacy validation coverage gaps without relaxing plan tolerances."""
import numpy as np
import torch
from scipy.stats import binom
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from src.data.synthetic import load_synthetic
from src.data.split import get_train_test_split
from src.data.preprocessing import QKPreprocessor
from src.models.qsvm import QSVM
from src.kernel.fidelity import fixed_reference_kernel,compute_kernel_matrix
from src.attacks.a2_pgd import a2_pgd_attack
from src.utils.seed import seed_everything


def split_blobs():
    raw,y=load_synthetic('classification',n_samples=200,n_features=4,random_state=42)
    return get_train_test_split(raw,y,100,100,42)


def test_required_blob_baseline_at_least_90_percent():
    # PI delegated this decision on 2026-09-26. Preserve the original 0.90
    # threshold; select only from planned values using training CV. The old
    # fixed c=1 result (0.65) remains recorded in LOG.md, not erased.
    tr,te,ytr,yte=split_blobs()
    prep=QKPreprocessor(4)
    X=prep.fit_transform(tr,ytr)
    best_score=-np.inf
    selected=None
    for reps in (1,2,3):
        for bandwidth in (.1,.25,.5,1.,2.):
            with torch.no_grad():
                K=compute_kernel_matrix(X,reps=reps,c=bandwidth,p=0.).numpy()
            fit=GridSearchCV(SVC(kernel='precomputed'),{'C':[.1,1,10,100]},cv=3).fit(K,ytr)
            if fit.best_score_>best_score:
                best_score=fit.best_score_
                selected=(reps,bandwidth,fit.best_estimator_)
    reps,bandwidth,model=selected
    Y=prep.transform(te)
    with torch.no_grad():
        K_test=compute_kernel_matrix(Y,X,reps=reps,c=bandwidth,p=0.).numpy()
    assert np.mean(model.predict(K_test)==yte)>=.90


def test_shuffled_labels_full_pipeline_binomial_control():
    tr,te,ytr,yte=split_blobs()
    shuffled=np.random.default_rng(20260926).permutation(ytr)
    prep=QKPreprocessor(4)
    X=prep.fit_transform(tr,shuffled);Y=prep.transform(te)
    model=QSVM(reps=2,c=1.,p=0.).fit(X,shuffled)
    correct=int(np.sum(model.predict(Y)==yte))
    lower,upper=binom.ppf([.025,.975],len(yte),.5)
    assert lower<=correct<=upper, (correct,lower,upper)


def test_noisy_support_function_matches_sklearn():
    torch.manual_seed(6)
    X=torch.rand(20,3,dtype=torch.float64)*np.pi
    y=np.array([-1,1]*10)
    model=QSVM(reps=1,c=.5,p=.03).fit(X,y)
    Y=torch.rand(4,3,dtype=torch.float64)*np.pi
    sv=X[model.svm.support_]
    kernel=fixed_reference_kernel(sv,reps=1,c=.5,p=.03)
    actual=(kernel(Y,sv)@torch.tensor(model.svm.dual_coef_[0])+model.svm.intercept_[0]).numpy()
    np.testing.assert_allclose(actual,model.decision_function(Y),rtol=0,atol=1e-10)


def test_seed_reproduces_attack_candidates():
    X=torch.tensor([[.5,.2],[1.5,1.8]],dtype=torch.float64)
    sv=torch.tensor([[0.,0.],[2.,2.]],dtype=torch.float64)
    coefficients=torch.tensor([-1.,1.],dtype=torch.float64)
    kernel=lambda x,z:torch.exp(-((x[:,None]-z[None])**2).sum(-1))
    def run():
        seed_everything(9)
        return a2_pgd_attack(X,np.array([-1,1]),sv,coefficients,0,kernel,eps=1,T=5,R=3)
    first,second=run(),run()
    assert torch.equal(first['X_adv'],second['X_adv'])
    assert first['robust_acc']==second['robust_acc']
