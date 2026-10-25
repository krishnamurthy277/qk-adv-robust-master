"""Publication exports from saved results; does not execute experiments."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'manuscript/.figure_dependencies'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
T=ROOT/'results/tables'
OUT=ROOT/'manuscript/submission_figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
df=pd.read_csv(T/'tier_a_results.csv'); models=df.drop_duplicates('cell_hash')
names=['wbcd','gse2034']; colors=['#0072B2','#E69F00','#009E73','#CC79A7']
def finish(fig,axes,num):
    for letter,ax in zip('abcdefghijkl',np.asarray(axes).ravel()):
        ax.text(-.15,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)
    fig.savefig(OUT/f'Figure_{num}.png',dpi=300)
    fig.savefig(OUT/f'Figure_{num}.pdf')
    plt.close(fig)
def boot(v):
    v=np.asarray(v); rng=np.random.default_rng(0)
    samples=rng.choice(v,(5000,len(v))).mean(axis=1)
    return v.mean(),*np.quantile(samples,[.025,.975])
fig,axs=plt.subplots(1,2,figsize=(7,2.9),layout='constrained')
for ax,d in zip(axs,names):
    p=models[(models.dataset==d)&(models.bandwidth==1)].pivot_table(index='reps',columns='noise_p',values='log10_var_K')
    im=ax.imshow(p,origin='lower',aspect='auto'); ax.set(xticks=range(4),xticklabels=p.columns,yticks=range(3),yticklabels=p.index,xlabel='Noise probability p',ylabel='Repetitions',title=d.upper())
    fig.colorbar(im,ax=ax,label='Mean log₁₀ variance',shrink=.85)
finish(fig,axs,1)
fig,axs=plt.subplots(3,4,figsize=(7,6.6),layout='constrained')
for j,(d,a) in enumerate([(d,a) for d in names for a in ['a1','a2']]):
    for i,r in enumerate([1,2,3]):
        ax=axs[i,j]; part=df[(df.dataset==d)&(df.bandwidth==1)&(df.reps==r)]
        for color,(p,curve) in zip(colors,part.groupby('noise_p')):
            x=[]; ys=[]
            for e,g in curve.groupby('eps_rel'): x.append(e); ys.append(boot(g[a+'_robust_acc']))
            y=np.asarray(ys); ax.plot(x,y[:,0],'.-',color=color,label=str(p)); ax.fill_between(x,y[:,1],y[:,2],color=color,alpha=.15)
        ax.set(title=f'{d.upper()} {a.upper()}, r={r}',ylim=(0,1),xticks=[.1,.2,.4])
        if i==2: ax.set_xlabel('Relative L₂ budget')
        if j==0: ax.set_ylabel('Robust accuracy')
        if i==0 and j==0: ax.legend(title='Noise p',fontsize=6,title_fontsize=7)
finish(fig,axs,2)
fig,axs=plt.subplots(1,2,figsize=(7,3.2),layout='constrained')
for ax,d in zip(axs,names):
    p=models[models.dataset==d]; im=ax.scatter(p.log10_var_K,p.a0_median_margin,c=p.noise_p,cmap='plasma',s=16,alpha=.7)
    ax.set(xlabel='log₁₀ kernel variance',ylabel='Median A0 proxy (L₂)',yscale='log',title=d.upper()); fig.colorbar(im,ax=ax,label='Noise p')
finish(fig,axs,3)
gaps=pd.read_csv(T/'analysis/h3_masking_gap.csv')
fig,axs=plt.subplots(3,4,figsize=(7,5.8),layout='constrained')
for i,r in enumerate([1,2,3]):
    for j,e in enumerate([.05,.1,.2,.4]):
        ax=axs[i,j]; p=gaps[(gaps.dataset=='wbcd')&(gaps.bandwidth==1)&(gaps.reps==r)&(gaps.eps_rel==e)].sort_values('noise_p')
        ax.plot(p.noise_p,p['mean'],'.-',color=colors[0]); ax.fill_between(p.noise_p,p.low,p.high,alpha=.2,color=colors[0]); ax.axhline(.05,color='gray',ls='--'); ax.axhline(0,color='black',lw=.5)
        ax.set(title=f'r={r}, budget={e}',xticks=[0,.01,.03])
        if i==2: ax.set_xlabel('Noise p')
        if j==0: ax.set_ylabel('A1 − A2 accuracy')
finish(fig,axs,4)
extra=pd.read_csv(T/'iso_concentration_results.csv'); extension=pd.read_csv(T/'iso_overlap_followup_results.csv')
fig,axs=plt.subplots(3,2,figsize=(7,6.8),layout='constrained')
for i,(d,extended) in enumerate([('wbcd',False),('wbcd',True),('gse2034',True)]):
    full=pd.concat([df,extra]+([extension] if extended else [])); part=full[(full.dataset==d)&(full.reps==3)&(full.eps_rel==.1)]
    curves=[]
    for seed,g in part.groupby('seed'):
        n=g[g.bandwidth==1].sort_values('log10_var_K').drop_duplicates('log10_var_K'); b=g[g.noise_p==0].sort_values('log10_var_K').drop_duplicates('log10_var_K'); curves.append((n,b))
    lo=max(max(n.log10_var_K.min(),b.log10_var_K.min()) for n,b in curves); hi=min(min(n.log10_var_K.max(),b.log10_var_K.max()) for n,b in curves)
    for j,metric in enumerate(['a2_robust_acc','a0_median_margin']):
        ax=axs[i,j]
        for n,b in curves:
            for c,color in [(n,colors[0]),(b,colors[1])]: ax.plot(c.log10_var_K,c[metric],'.-',color=color,alpha=.35,lw=.8,ms=3)
        if hi-lo>.001:
            x=np.linspace(lo,hi,22)[1:-1]
            for k,color in enumerate(colors[:2]):
                arr=np.array([np.interp(x,curves[s][k].log10_var_K,curves[s][k][metric]) for s in range(3)])
                y=np.array([boot(arr[:,t]) for t in range(20)]); ax.plot(x,y[:,0],color=color); ax.fill_between(x,y[:,1],y[:,2],color=color,alpha=.15)
        else: ax.text(.03,.06,'No common interior range',transform=ax.transAxes,fontsize=7)
        ax.set(title=f'{d.upper()} '+('extended' if extended else 'original'),xlabel='log₁₀ kernel variance',ylabel='A2 robust accuracy' if j==0 else 'Median A0 proxy')
        if i==0 and j==0:
            for color,label in zip(colors,['Noise family','Bandwidth family']): ax.plot([],[],color=color,label=label)
            ax.legend(loc='upper left')
finish(fig,axs,5)
pairs=pd.read_csv(T/'analysis/h5_pairs.csv')
fig,axs=plt.subplots(1,2,figsize=(7,3.4),layout='constrained')
for ax,d in zip(axs,names):
    p=pairs[pairs.dataset==d]; g=p[p.eligible]; ax.scatter(p.a2_robust_acc_q,p.a2_robust_acc_rbf,c='gray',alpha=.2,s=18,label='All pairs'); ax.scatter(g.a2_robust_acc_q,g.a2_robust_acc_rbf,c=colors[0],s=18,label='Eligible pairs')
    x=np.linspace(0,1,100); ax.plot(x,x,c='black',lw=.8); ax.fill_between(x,x-.03,x+.03,color=colors[2],alpha=.12)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='QSVM A2 robust accuracy',ylabel='RBF A2 robust accuracy',title=d.upper()); ax.legend(loc='upper left')
finish(fig,axs,6)
print(OUT)
