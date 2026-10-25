"""Build diagnostic-focused main figures and retain original supporting plots."""
from pathlib import Path
import sys,shutil,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'manuscript/.figure_dependencies'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'manuscript/publication_figures'; OUT.mkdir(exist_ok=True)
OLD=ROOT/'manuscript/submission_figures'; DATA=ROOT/'results/publication_followup'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,axs,n):
    for letter,ax in zip('abcd',np.asarray(axs).ravel()): ax.text(-.12,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=11)
    fig.savefig(OUT/f'Figure_{n}.png',dpi=300); fig.savefig(OUT/f'Figure_{n}.pdf'); plt.close(fig)
u=pd.read_csv(DATA/'utility_by_model_budget.csv')
fig,axs=plt.subplots(1,2,figsize=(7,3.3),layout='constrained')
for ax,d in zip(axs,['wbcd','gse2034']):
    for constant,label,color in [(True,'Constant sign','#D55E00'),(False,'Bound unresolved','#0072B2')]:
        g=u[(u.dataset==d)&(u.constant==constant)]; y=g.groupby('eps_rel').a2_balanced_acc.mean(); clean=g.drop_duplicates('cell_hash').clean_balanced_acc.mean()
        ax.plot([0]+list(y.index),[clean]+list(y.values),'o-',color=color,label=f'{label} (n={g.cell_hash.nunique()})',ms=4)
    ax.axhline(.5,color='gray',ls=':',lw=1); ax.set(title=d.upper(),xlabel='Relative L₂ budget (0 = clean)',ylabel='Balanced accuracy',ylim=(0,.75)); ax.legend(fontsize=7)
save(fig,axs,1)
s=pd.read_csv(DATA/'nested_selection.csv')
fig,axs=plt.subplots(1,2,figsize=(7,3.3),layout='constrained')
for ax,d in zip(axs,['wbcd','gse2034']):
    g=s[s.dataset==d].pivot(index='seed',columns='class_weight',values='balanced_acc')
    for _,r in g.iterrows(): ax.plot([0,1],[r['none'],r['balanced']],'-o',color='#0072B2',alpha=.45,ms=4)
    ax.plot([0,1],[g['none'].mean(),g['balanced'].mean()],'D',color='black',ms=7,label='Mean over ten splits'); ax.axhline(.5,color='gray',ls=':')
    ax.set(xticks=[0,1],xticklabels=['Unweighted','Class weighted'],ylabel='Held-out balanced accuracy',title=d.upper(),ylim=(.4,1)); ax.legend(fontsize=7)
save(fig,axs,2)
q=pd.read_csv(DATA/'query_by_model.csv')
fig,axs=plt.subplots(1,2,figsize=(7,3.3),layout='constrained')
for ax,d in zip(axs,['wbcd','gse2034']):
    g=q[q.dataset==d]
    for r in g.itertuples(): ax.plot([0,1],[r.pgd_correct,r.union_correct],'-o',color='#D55E00' if r.new_failures else '#777777',alpha=.5,ms=3)
    ax.plot([0,1],[g.pgd_correct.mean(),g.union_correct.mean()],'D',color='black',ms=7,label='Model mean')
    ax.set(xticks=[0,1],xticklabels=['PGD','PGD + random search'],ylabel='Empirical robust accuracy',title=f'{d.upper()} (n={len(g)})',ylim=(-.01,.3)); ax.legend(fontsize=7)
save(fig,axs,4)
for old,new in [(1,3),(5,5),(6,6),(2,'S1'),(3,'S2'),(4,'S3')]:
    for ext in ['png','pdf']: shutil.copy2(OLD/f'Figure_{old}.{ext}',OUT/f'Figure_{new}.{ext}')
captions={
1:'Balanced utility separates constant predictions from attack vulnerability. a, WBCD; b, GSE2034. The zero-budget point shows clean balanced accuracy; positive budgets show balanced A2 robust accuracy. Lines are descriptive means across all models in each spectral-audit group, with each model counted once per budget. Constant-sign models remain at 0.5. An unresolved bound does not establish nonconstant predictions. Configurations are not independent replicates; no confidence band is inferred from their count.',
2:'Held-out utility after nested model selection on ten additional splits. a, WBCD; b, GSE2034. Each line pairs unweighted and class-weighted SVMs on the same split (seeds 3–12); diamonds show the mean. Preprocessing is refitted within each inner fold. Bandwidth, noise and C are selected using inner balanced accuracy at one repetition. The dotted line marks 0.5. No selected model meets the sufficient constant-sign bound, but one unweighted GSE2034 model predicts one class on its test set. These are utility measurements, not adversarial robustness results.',
3:'Kernel concentration at c=1. a, WBCD; b, GSE2034. Heatmaps show mean log10 off-diagonal training-kernel variance over three original split seeds for each repetition count and noise level. This fixed five-qubit experiment does not establish asymptotic scaling.',
4:'An independent query check finds failures missed by PGD. a, WBCD; b, GSE2034. Each line pairs a spectrally unresolved primary model before and after combining saved PGD outcomes with 128 gradient-free candidates per point at relative budget 0.4. Orange lines identify models with additional failures; diamonds show descriptive model means. The 25 additional model–point failures occur in 13 models. Repeated appearances of a sample under different models are not independent failures. The union retains every PGD success and is not a robustness certificate.',
5:'Original and exploratory concentration comparisons at r=3 and relative budget 0.1. a,b, Original WBCD family; c,d, exploratory WBCD extension; e,f, exploratory GSE2034 extension. Left panels show A2 robust accuracy; right panels show median A0. Faint curves show individual seeds; opaque curves and bands summarize common support. Extensions add c=4 and c=6 using training-only selection. Bands are pointwise 95% bootstrap intervals, not simultaneous bands.',
6:'Classical comparison coverage. a, WBCD; b, GSE2034. Gray points show all matching candidates; blue points meet both clean-accuracy and variance filters. The diagonal marks equality; shading indicates ±0.03. All 168 eligible budget pairs are GSE2034 and 160 involve a constant QSVM. Repeated budgets are not independent replicates.'}
(OUT/'captions.json').write_text(json.dumps(captions,indent=2,ensure_ascii=False),encoding='utf-8')
print(OUT)
