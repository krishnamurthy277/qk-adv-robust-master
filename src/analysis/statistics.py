"""Seed-level uncertainty and explicitly descriptive partial rank associations."""
import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon


def bootstrap_mean(values, draws=5000, seed=20260926):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if not len(values): return (np.nan,np.nan,np.nan)
    if len(values)<2: return (float(values.mean()),np.nan,np.nan)
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),(draws,len(values)))].mean(1)
    low,high=np.quantile(means,[.025,.975])
    return float(values.mean()),float(low),float(high)


def partial_spearman(x,y,controls):
    x,y=np.asarray(x,float),np.asarray(y,float)
    controls=np.asarray(controls,float)
    if controls.ndim==1: controls=controls[:,None]
    valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(controls).all(1)
    x,y,controls=x[valid],y[valid],controls[valid]
    if len(x)<5 or np.ptp(x)==0 or np.ptp(y)==0: return np.nan
    design=np.column_stack([np.ones(len(x))]+[rankdata(v) for v in controls.T])
    if len(x)<=np.linalg.matrix_rank(design)+2: return np.nan
    rx,ry=rankdata(x),rankdata(y)
    rx=rx-design@np.linalg.lstsq(design,rx,rcond=None)[0]
    ry=ry-design@np.linalg.lstsq(design,ry,rcond=None)[0]
    if np.linalg.norm(rx)<1e-10 or np.linalg.norm(ry)<1e-10:return np.nan
    return float(np.corrcoef(rx,ry)[0,1])


def safe_spearman(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    valid=np.isfinite(x)&np.isfinite(y)
    if valid.sum()<3 or np.ptp(x[valid])==0 or np.ptp(y[valid])==0:return np.nan
    return float(spearmanr(x[valid],y[valid]).statistic)


def paired_p(differences):
    diff=np.asarray(differences,float)
    diff=diff[np.isfinite(diff)]
    if len(diff)<2:return np.nan
    if np.allclose(diff,0):return 1.
    return float(wilcoxon(diff,zero_method='wilcox',alternative='two-sided',method='auto').pvalue)


def benjamini_hochberg(pvalues):
    p=np.asarray(pvalues,float)
    result=np.full_like(p,np.nan)
    valid=np.flatnonzero(np.isfinite(p))
    order=valid[np.argsort(p[valid])]
    adjusted=p[order]*len(order)/np.arange(1,len(order)+1)
    result[order]=np.minimum(1,np.minimum.accumulate(adjusted[::-1])[::-1])
    return result
