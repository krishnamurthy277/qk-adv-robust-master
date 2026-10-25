"""Post-review utility and independent query checks; original artifacts read-only."""
from pathlib import Path
import sys, json, itertools, hashlib, platform
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from joblib import Parallel, delayed
from scripts.run_experiment import load_dataset,fit_svm_precomputed
from scripts.review_diagnostics import decision_bounds
from src.data.split import get_train_test_split
from src.data.preprocessing import QKPreprocessor
from src.kernel.fidelity import feature_density,density_overlap
OUT=ROOT/'results/publication_followup'; OUT.mkdir(exist_ok=True)
T=ROOT/'results/tables'; RUN=ROOT/'results/runs/tier_a_main/24d365adc00fcd58/cache'
def split(d,s):
    raw,y=load_dataset(d,5)
    a,b,c,e=get_train_test_split(raw,y,60,60,int(s))
    return a,b,np.where(c<=0,-1,1),np.where(e<=0,-1,1)
def prep(d): return QKPreprocessor(5,1000 if d=='gse2034' else None)
def balanced(correct,y): return float(np.mean([np.mean(correct[y==label]) for label in [-1,1]]))
def utility():
    audit=pd.read_csv(T/'analysis/model_degeneracy_audit.csv'); rows=[]
    for (d,s),part in audit.groupby(['dataset','seed']):
        _,_,yt,y=split(d,s); majority=1 if (yt==1).sum()>(yt==-1).sum() else -1
        for m in part.itertuples():
            points=pd.read_csv(RUN/m.cell_hash/'attack_points.csv')
            for e,g in points.groupby('eps_rel'):
                g=g.sort_values('test_index'); assert np.array_equal(g.test_index,np.arange(60))
                rows.append(dict(cell_hash=m.cell_hash,dataset=d,seed=s,reps=m.reps,noise_p=m.noise_p,bandwidth=m.bandwidth,constant=m.certified_constant,eps_rel=e,clean_acc=m.clean_acc,clean_balanced_acc=m.balanced_acc,positive_fraction=m.positive_prediction_fraction,majority_acc=float(np.mean(y==majority)),a2_acc=float(g.a2_correct.mean()),a2_balanced_acc=balanced(g.a2_correct.to_numpy(),y)))
    pd.DataFrame(rows).to_csv(OUT/'utility_by_model_budget.csv',index=False)
    frame=pd.DataFrame(rows)
    frame.groupby(['dataset','constant','eps_rel']).agg(models=('cell_hash','size'),clean_balanced_mean=('clean_balanced_acc','mean'),robust_balanced_mean=('a2_balanced_acc','mean'),clean_mean=('clean_acc','mean'),robust_mean=('a2_acc','mean'),majority_mean=('majority_acc','mean')).to_csv(OUT/'utility_summary.csv')
def query(m):
    torch.set_num_threads(1); out=OUT/'query_cells'/f'{m.cell_hash}.csv'; out.parent.mkdir(exist_ok=True)
    if out.exists(): return
    a,b,yt,y=split(m.dataset,m.seed); pp=prep(m.dataset); X=pp.fit_transform(a,yt); Y=pp.transform(b)
    with torch.no_grad():
        rho=feature_density(X,m.reps,m.bandwidth,m.noise_p); svm,C=fit_svm_precomputed(density_overlap(rho,rho).numpy(),yt)
        assert C==m.C
        coef=torch.tensor(svm.dual_coef_[0]); operator=torch.einsum('i,ijk->jk',coef.to(torch.complex128),rho[svm.support_])
        def score(z): return torch.einsum('bij,ji->b',feature_density(z,m.reps,m.bandwidth,m.noise_p),operator).real+svm.intercept_[0]
        clean=score(Y).numpy(); assert abs(np.mean((clean>0)==(y>0))-m.clean_acc)<1e-10
        eps=.4*float(torch.pdist(X).median()); # Match numpy median exactly below.
        eps=.4*float(np.median(torch.pdist(X).numpy()))
        rng=np.random.default_rng(20260930+int(m.seed)); records=[]
        original=pd.read_csv(RUN/m.cell_hash/'attack_points.csv'); original=original[original.eps_rel==.4].sort_values('test_index')
        for j,x in enumerate(Y):
            v=rng.normal(size=(128,5)); v/=np.linalg.norm(v,axis=1)[:,None]
            radii=np.ones(128); radii[:64]=rng.random(64)**(1/5)
            z=(x+torch.tensor(eps*v*radii[:,None])).clamp(0,np.pi)
            assert torch.max(torch.linalg.vector_norm(z-x,dim=1))<=eps+1e-10
            scores=score(z).numpy()*y[j]; best=min(float(clean[j]*y[j]),float(scores.min()))
            pgd=bool(original.iloc[j].a2_correct); query_correct=best>0
            records.append(dict(cell_hash=m.cell_hash,dataset=m.dataset,seed=m.seed,test_index=j,label=int(y[j]),eps_rel=.4,initially_correct=bool(clean[j]*y[j]>0),pgd_correct=pgd,query_correct=query_correct,union_correct=pgd and query_correct,minimum_signed_score=best,queries=128))
    pd.DataFrame(records).to_csv(out,index=False); print('Query completed',m.cell_hash,flush=True)
def selected(d,s):
    torch.set_num_threads(1); out=OUT/'selection_cells'/f'{d}_{s}.csv'; out.parent.mkdir(exist_ok=True)
    if out.exists(): return
    a,b,yt,y=split(d,s); grid=list(itertools.product([.1,.5,1.],[0.,.03],[.1,1.,10.,100.],[False,True])); scores={g:[] for g in grid}
    for tr,va in StratifiedKFold(3,shuffle=True,random_state=s).split(a,yt):
        pp=prep(d); X=pp.fit_transform(a[tr],yt[tr]); V=pp.transform(a[va])
        with torch.no_grad():
            for c,p in itertools.product([.1,.5,1.],[0.,.03]):
                rho=feature_density(X,1,c,p); vrho=feature_density(V,1,c,p); K=density_overlap(rho,rho).numpy(); KV=density_overlap(vrho,rho).numpy()
                for C,weight in itertools.product([.1,1.,10.,100.],[False,True]):
                    model=SVC(kernel='precomputed',C=C,class_weight='balanced' if weight else None).fit(K,yt[tr]); scores[c,p,C,weight].append(balanced_accuracy_score(yt[va],model.predict(KV)))
    pp=prep(d); X=pp.fit_transform(a,yt); Y=pp.transform(b); rows=[]
    with torch.no_grad():
        for weight in [False,True]:
            choices=[g for g in grid if g[3]==weight]; best=max(choices,key=lambda g:np.mean(scores[g])); c,p,C,_=best
            rho=feature_density(X,1,c,p); test=feature_density(Y,1,c,p); model=SVC(kernel='precomputed',C=C,class_weight='balanced' if weight else None).fit(density_overlap(rho,rho).numpy(),yt)
            pred=model.predict(density_overlap(test,rho).numpy()); lo,hi=decision_bounds(rho[model.support_],torch.tensor(model.dual_coef_[0]),float(model.intercept_[0]))
            majority=1 if np.sum(yt==1)>np.sum(yt==-1) else -1
            rows.append(dict(dataset=d,seed=s,class_weight='balanced' if weight else 'none',reps=1,bandwidth=c,noise_p=p,C=C,inner_balanced_acc=float(np.mean(scores[best])),clean_acc=float(np.mean(pred==y)),balanced_acc=balanced_accuracy_score(y,pred),majority_acc=float(np.mean(y==majority)),positive_fraction=float(np.mean(pred==1)),global_lower=lo,global_upper=hi,constant=bool(lo>1e-8 or hi< -1e-8)))
    pd.DataFrame(rows).to_csv(out,index=False)
    (out.with_suffix('.json')).write_text(json.dumps({str(g):v for g,v in scores.items()},indent=2))
    print('Selection completed',d,s,flush=True)
def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['utility','query','selection','all'],default='all'); ap.add_argument('--jobs',type=int,default=4); args=ap.parse_args()
    torch.set_num_threads(1)
    if args.mode in ['utility','all']: utility()
    if args.mode in ['query','all']:
        audit=pd.read_csv(T/'analysis/model_degeneracy_audit.csv'); Parallel(n_jobs=args.jobs)(delayed(query)(m) for m in audit[~audit.certified_constant].itertuples(index=False,name='Model'))
        pd.concat([pd.read_csv(p) for p in sorted((OUT/'query_cells').glob('*.csv'))]).to_csv(OUT/'query_validation.csv',index=False)
    if args.mode in ['selection','all']:
        Parallel(n_jobs=args.jobs)(delayed(selected)(d,s) for d in ['wbcd','gse2034'] for s in range(3,13))
        pd.concat([pd.read_csv(p) for p in sorted((OUT/'selection_cells').glob('*.csv'))]).to_csv(OUT/'nested_selection.csv',index=False)
    files=[Path(__file__),ROOT/'docs/PUBLICATION_FOLLOWUP.md',T/'tier_a_results.csv',T/'analysis/model_degeneracy_audit.csv']
    files+=list((ROOT/'src').rglob('*.py'))+list((ROOT/'data/raw').glob('*'))
    manifest={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'files':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
