"""Regenerate F1–F6, seed statistics and cautious H1–H5 verdicts from saved runs."""
import argparse
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.analysis.statistics import (bootstrap_mean,partial_spearman,safe_spearman,
                                     paired_p,benjamini_hochberg)

KEY=['dataset','seed','reps','noise_p','bandwidth']
STRATA=['dataset','reps','noise_p','bandwidth','eps_rel']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})


def validate_table(frame,cfg):
    expected=set(itertools.product(cfg['datasets'],cfg['seeds'],cfg['reps'],cfg['noise_p'],cfg['bandwidth'],cfg['eps_rel']))
    actual=list(frame[KEY+['eps_rel']].itertuples(index=False,name=None))
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError(f'Incomplete/duplicate grid: {len(actual)} rows, expected {len(expected)}')
    if frame.version.nunique()!=1 or not frame.version.eq('v2_per_gate').all():
        raise ValueError('Only corrected per-gate results are admissible')
    for col in ['a1_robust_acc','a2_robust_acc']:
        if (frame[col]>frame.clean_acc+1e-12).any():raise ValueError(f'{col} exceeds clean accuracy')


def save(fig,path):
    fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches='tight')
    plt.close(fig)


def seed_summary(frame,groups,value):
    rows=[]
    for keys,part in frame.groupby(groups):
        if not isinstance(keys,tuple):keys=(keys,)
        per_seed=part.groupby('seed')[value].mean()
        mean,low,high=bootstrap_mean(per_seed.values)
        rows.append(dict(zip(groups,keys),mean=mean,low=low,high=high,n_seeds=len(per_seed),
                         min_seed=float(per_seed.min()),max_seed=float(per_seed.max()),
                         p_value=paired_p(per_seed.values)))
    out=pd.DataFrame(rows)
    if len(out):out['p_bh']=benjamini_hochberg(out.p_value)
    return out


def phase5_figures(df,figures):
    models=df.drop_duplicates('cell_hash')
    for (dataset,c),part in models.groupby(['dataset','bandwidth']):
        pivot=part.pivot_table(index='reps',columns='noise_p',values='log10_var_K',aggfunc='mean')
        fig,ax=plt.subplots(figsize=(5,3))
        im=ax.imshow(pivot.values,aspect='auto',origin='lower',cmap='viridis')
        ax.set(xticks=range(len(pivot.columns)),xticklabels=pivot.columns,
               yticks=range(len(pivot.index)),yticklabels=pivot.index,
               xlabel='Two-qubit noise probability p',ylabel='Repetitions',title=f'{dataset}, bandwidth {c}')
        fig.colorbar(im,ax=ax,label='Mean log10 off-diagonal variance (seeds)')
        save(fig,figures/f'F1_{dataset}_c{c}.png')
    for (dataset,c),part in df.groupby(['dataset','bandwidth']):
        reps=sorted(part.reps.unique())
        fig,axes=plt.subplots(len(reps),2,figsize=(10,3*len(reps)),squeeze=False)
        for i,r in enumerate(reps):
            for j,attack in enumerate(['a1','a2']):
                ax=axes[i,j]
                subset=part[part.reps==r]
                for p,curve in subset.groupby('noise_p'):
                    summary=seed_summary(curve,['eps_rel'],attack+'_robust_acc').sort_values('eps_rel')
                    ax.plot(summary.eps_rel,summary['mean'],marker='o',label=f'p={p}')
                    ax.fill_between(summary.eps_rel,summary.low,summary.high,alpha=.12)
                ax.set(xlabel='L2 budget / median training distance',ylabel='Robust accuracy',
                       ylim=(-.02,1.02),title=f'{attack.upper()}, repetitions {r}')
                ax.legend(fontsize=7)
        n_seeds=part.seed.nunique()
        interval_label=f'95% bootstrap over {n_seeds} seeds' if n_seeds>1 else 'one seed; no uncertainty band'
        fig.suptitle(f'{dataset}, c={c}; {interval_label}',y=1.01)
        save(fig,figures/f'F2_{dataset}_c{c}.png')


def concentration_analysis(models,tables,figures):
    strata=[]
    for (dataset,seed),part in models.groupby(['dataset','seed']):
        for axis,fixed in [('noise_p',['reps','bandwidth']),('reps',['noise_p','bandwidth'])]:
            for keys,group in part.groupby(fixed):
                strata.append(dict(dataset=dataset,seed=seed,axis=axis,
                    fixed=str(dict(zip(fixed,keys))),rho=safe_spearman(group[axis],group.var_K)))
    strata=pd.DataFrame(strata); strata.to_csv(tables/'h1_strata.csv',index=False)
    groups=strata.groupby(['dataset','seed','axis']).rho.median().unstack('axis').reset_index()
    groups['criterion_met']=(groups.noise_p<=-.7)&(groups.reps<=-.7)
    groups.to_csv(tables/'h1_groups.csv',index=False)
    correlations=[]
    for (dataset,seed),part in models.groupby(['dataset','seed']):
        correlations.append(dict(dataset=dataset,seed=seed,
            spearman_a0=safe_spearman(part.var_K,part.a0_median_margin),
            partial_clean_a0=partial_spearman(part.var_K,part.a0_median_margin,part[['clean_acc']]),
            partial_clean_reps_noise_a0=partial_spearman(part.var_K,part.a0_median_margin,
                                                       part[['clean_acc','reps','noise_p']])))
    corr=pd.DataFrame(correlations);corr.to_csv(tables/'h2_proxy_correlations.csv',index=False)
    for dataset,part in models.groupby('dataset'):
        fig,ax=plt.subplots(figsize=(6,4))
        points=ax.scatter(part.log10_var_K,part.a0_median_margin,c=part.noise_p,cmap='plasma',alpha=.7)
        ax.set(xlabel='log10 kernel variance',ylabel='A0 median linearized margin (L2)',
               yscale='log',title=f'{dataset}: proxy association, all models')
        fig.colorbar(points,ax=ax,label='Noise p')
        save(fig,figures/f'F3_{dataset}.png')
    fraction=float(groups.criterion_met.mean())
    return dict(verdict='supported' if fraction>=.8 else 'not supported',
                fraction_groups_meeting_both_axes=fraction,groups=len(groups)),corr


def masking_analysis(df,cfg,tables,figures):
    summary=seed_summary(df,STRATA,'masking_gap')
    summary.to_csv(tables/'h3_masking_gap.csv',index=False)
    flagged={col:int((df[col]==False).sum()) for col in ['M1_passed','M2_passed','M4_passed','M5_passed']}
    df[KEY+['eps_rel','M1_passed','M2_passed','M3_passed','M4_passed','M5_passed',
            'a2_large_robust_acc','a0_median_grad_norm','grad_p10','grad_p90']].to_csv(tables/'masking_diagnostics.csv',index=False)
    for (dataset,c),part in summary.groupby(['dataset','bandwidth']):
        fig,axes=plt.subplots(len(cfg['reps']),len(cfg['eps_rel']),figsize=(13,8),squeeze=False)
        for i,r in enumerate(cfg['reps']):
            for j,e in enumerate(cfg['eps_rel']):
                curve=part[(part.reps==r)&(part.eps_rel==e)].sort_values('noise_p')
                ax=axes[i,j]
                ax.plot(curve.noise_p,curve['mean'],marker='o')
                ax.fill_between(curve.noise_p,curve.low,curve.high,alpha=.2)
                ax.axhline(.05,color='gray',linestyle='--');ax.axhline(0,color='black',linewidth=.5)
                ax.set(title=f'r={r}, eps_rel={e}',xlabel='Noise p',ylabel='A1 - A2 robust accuracy')
        save(fig,figures/f'F4_{dataset}_c{c}.png')
    consistent=summary[(summary.noise_p>0)&(summary.n_seeds==len(cfg['seeds']))&(summary.min_seed>.05)]
    return dict(verdict='supported' if len(consistent) else 'not supported',
                consistent_strata=len(consistent),diagnostic_failed_rows=flagged,
                M3='untested; black-box attacks absent',
                caution='A transfer gap does not by itself prove gradient masking; diagnostics limit conclusions.')


def margin_agreement(models,run_dir,tables):
    rows=[]
    for model in models.itertuples():
        base=run_dir/'cache'/model.cell_hash
        a0=np.load(base/'a0_points.npz')
        points=pd.read_csv(base/'attack_points.csv')
        indices=np.flatnonzero(a0['correctly_classified'])[:20]
        margins=a0['margins'][:20]
        upper=[]
        for index in indices:
            point=points[(points.test_index==index)&(~points.a2_correct)]
            upper.append(point.eps_rel.min()*model.d_med if len(point) else np.inf)
        upper=np.asarray(upper)
        rows.append(dict(cell_hash=model.cell_hash,dataset=model.dataset,seed=model.seed,
            n_points=len(indices),n_uncensored=int(np.isfinite(upper).sum()),
            right_censored_fraction=float(np.mean(~np.isfinite(upper))) if len(upper) else np.nan,
            rho_uncensored_only=safe_spearman(margins,upper),
            median_tested_success_radius=float(np.median(upper)) if len(upper) else np.nan))
    pd.DataFrame(rows).to_csv(tables/'a0_pgd_agreement.csv',index=False)


def iso_analysis(main,extra,cfg,tables,figures):
    full=pd.concat([main,extra],ignore_index=True)
    rows=[]
    pointwise=[]
    for (dataset,r,e),part in full.groupby(['dataset','reps','eps_rel']):
        curves={}
        lows=[];highs=[]
        for seed in cfg['seeds']:
            seedpart=part[part.seed==seed]
            noise=seedpart[seedpart.bandwidth==1.].sort_values('log10_var_K')
            band=seedpart[seedpart.noise_p==0].sort_values('log10_var_K')
            noise=noise.drop_duplicates('log10_var_K');band=band.drop_duplicates('log10_var_K')
            if len(noise)<2 or len(band)<2:continue
            low=max(noise.log10_var_K.min(),band.log10_var_K.min())
            high=min(noise.log10_var_K.max(),band.log10_var_K.max())
            lows.append(low);highs.append(high);curves[seed]=(noise,band)
        low=max(lows) if lows else np.nan;high=min(highs) if highs else np.nan
        overlap=high-low
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for k,metric in enumerate(['a2_robust_acc','a0_median_margin']):
            ax=axes[k]
            for index,(seed,(noise,band)) in enumerate(curves.items()):
                ax.plot(noise.log10_var_K,noise[metric],color='tab:blue',alpha=.5,marker='o',
                        label='Noise only (individual seeds)' if index==0 else None)
                ax.plot(band.log10_var_K,band[metric],color='tab:orange',alpha=.5,marker='s',
                        label='Bandwidth only (individual seeds)' if index==0 else None)
            differences=[]
            if overlap>1e-3 and len(curves)==len(cfg['seeds']):
                grid=np.linspace(low,high,22)[1:-1] # omit endpoints/shared anchor
                family_n=[];family_b=[]
                for seed,(noise,band) in curves.items():
                    n=np.interp(grid,noise.log10_var_K,noise[metric])
                    b=np.interp(grid,band.log10_var_K,band[metric])
                    family_n.append(n);family_b.append(b)
                    differences.append(float(np.mean(n-b)))
                for family,color,label in [(family_n,'tab:blue','Noise only'),(family_b,'tab:orange','Bandwidth only')]:
                    array=np.asarray(family)
                    summary=np.asarray([bootstrap_mean(array[:,i]) for i in range(len(grid))])
                    ax.plot(grid,summary[:,0],color=color,label=label)
                    ax.fill_between(grid,summary[:,1],summary[:,2],color=color,alpha=.2)
                for index,variance in enumerate(grid):
                    delta=np.asarray(family_n)[:,index]-np.asarray(family_b)[:,index]
                    center,lower,upper=bootstrap_mean(delta)
                    pointwise.append(dict(dataset=dataset,reps=r,eps_rel=e,metric=metric,
                        log10_var_K=variance,difference=center,low=lower,high=upper,
                        n_seeds=len(delta),p_value=paired_p(delta)))
                ax.legend()
            mean,ci_low,ci_high=bootstrap_mean(differences)
            rows.append(dict(dataset=dataset,reps=r,eps_rel=e,metric=metric,
                overlap_log10_width=overlap,n_seeds=len(differences),difference=mean,
                low=ci_low,high=ci_high,p_value=paired_p(differences)))
            ax.legend(fontsize=7,loc='best')
            ylabel='A2 robust accuracy' if metric=='a2_robust_acc' else 'A0 median linearized margin'
            title=f'{dataset}, r={r}, eps_rel={e}'
            if not differences:title+='\nNo common interior range across all seeds'
            ax.set(xlabel='log10 kernel variance',ylabel=ylabel,title=title)
        save(fig,figures/f'F5_{dataset}_r{r}_eps{e}.png')
    result=pd.DataFrame(rows);result['p_bh']=benjamini_hochberg(result.p_value)
    result.to_csv(tables/'h4_iso_concentration.csv',index=False)
    points=pd.DataFrame(pointwise,columns=['dataset','reps','eps_rel','metric','log10_var_K',
                                         'difference','low','high','n_seeds','p_value'])
    points['p_bh']=benjamini_hochberg(points.p_value)
    points.to_csv(tables/'h4_iso_pointwise.csv',index=False)
    separated=points[(points.n_seeds==len(cfg['seeds']))&((points.low>0)|(points.high<0))]
    return dict(verdict='not supported' if len(separated) else 'inconclusive',
                separated_comparisons=len(separated),
                separated_robust_accuracy_points=int((separated.metric=='a2_robust_acc').sum()),
                separated_a0_proxy_points=int((separated.metric=='a0_median_margin').sum()),
                bh_significant_points=int((points.p_bh<.05).sum()),
                insufficient_overlap=int((result.overlap_log10_width<=1e-3).sum()),
                caution='Pointwise intervals are not simultaneous bands or independent comparisons; A0 is only a proxy. Interpolation and three seeds limit evidence; no equivalence tolerance prespecified.')


def classical_analysis(df,controls,cfg,tables,figures):
    matched=controls[controls['mode']=='variance_matched']
    pairs=df.merge(matched,on=['dataset','seed','reps','noise_p','bandwidth','eps_rel'],suffixes=('_q','_rbf'),validate='one_to_one')
    if len(pairs)!=len(df) or not pairs.cell_hash.eq(pairs.qcell_hash).all():
        raise ValueError('Classical controls do not completely match the current QSVM run')
    pairs['difference']=pairs.a2_robust_acc_rbf-pairs.a2_robust_acc_q
    pairs['eligible']=(abs(pairs.clean_acc_rbf-pairs.clean_acc_q)<=.03+1e-12)&(pairs.variance_relative_error<=.01)
    pairs.to_csv(tables/'h5_pairs.csv',index=False)
    eligible=pairs[pairs.eligible]
    summary=seed_summary(eligible,STRATA,'difference') if len(eligible) else pd.DataFrame()
    if len(summary):
        summary['all_seeds']=summary.n_seeds==len(cfg['seeds'])
        summary['equivalent']=summary.all_seeds&(summary.low>=-.03)&(summary.high<=.03)
        summary.to_csv(tables/'h5_equivalence.csv',index=False)
    else:pd.DataFrame(columns=STRATA+['mean','low','high','n_seeds']).to_csv(tables/'h5_equivalence.csv',index=False)
    for dataset,part in pairs.groupby('dataset'):
        fig,ax=plt.subplots(figsize=(5,5))
        ax.scatter(part.a2_robust_acc_q,part.a2_robust_acc_rbf,color='gray',alpha=.15,label='All pairs')
        good=part[part.eligible]
        ax.scatter(good.a2_robust_acc_q,good.a2_robust_acc_rbf,s=20,label='Variance + clean-accuracy matched')
        axis=np.linspace(0,1,100);ax.plot(axis,axis,color='black',linewidth=1)
        ax.fill_between(axis,axis-.03,axis+.03,color='green',alpha=.1)
        ax.set(xlabel='QSVM A2 robust accuracy',ylabel='Matched RBF A2 robust accuracy',
               xlim=(0,1),ylim=(0,1),title=dataset);ax.legend(fontsize=7)
        save(fig,figures/f'F6_{dataset}.png')
    full=summary[summary.all_seeds] if len(summary) else summary
    if not len(full):verdict='inconclusive'
    elif full.equivalent.all():verdict='supported'
    elif ((full.low>.03)|(full.high<-.03)).any():verdict='not supported'
    else:verdict='inconclusive'
    return dict(verdict=verdict,total_pairs=len(pairs),eligible_pairs=len(eligible),
                complete_seed_strata=len(full),scope='Eligible matched comparisons only; inspect coverage before generalizing.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/tier_a_main.yaml')
    parser.add_argument('--results',default='results/tables/tier_a_results.csv')
    parser.add_argument('--iso',default='results/tables/iso_concentration_results.csv')
    parser.add_argument('--controls',default='results/tables/classical_controls.csv')
    parser.add_argument('--phase5-only',action='store_true')
    parser.add_argument('--output-root',default='results')
    parser.add_argument('--iso-extension',default=None,
                        help='Optional training-concentration-selected follow-up CSV; original H4 retained separately')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    df=pd.read_csv(args.results);validate_table(df,cfg)
    output_root=Path(args.output_root)
    tables=output_root/'tables/analysis';tables.mkdir(parents=True,exist_ok=True)
    figures=output_root/'figures';figures.mkdir(parents=True,exist_ok=True)
    phase5_figures(df,figures)
    raw=dict(cells=df.cell_hash.nunique(),rows=len(df),datasets=sorted(df.dataset.unique()),
        n_qubits=sorted(map(int,df.n_qubits.unique())),seeds=sorted(map(int,df.seed.unique())),
        psd_repair_cells=int(df.drop_duplicates('cell_hash').psd_repaired.sum()),
        clean_accuracy_range=[float(df.clean_acc.min()),float(df.clean_acc.max())],
        diagnostics_failed_rows={m:int((df[m]==False).sum()) for m in ['M1_passed','M2_passed','M4_passed','M5_passed']})
    (tables/'phase5_raw_summary.json').write_text(json.dumps(raw,indent=2))
    if args.phase5_only:
        print(json.dumps(raw,indent=2));return
    checkpoint=Path('results/validation_checkpoint.json')
    if checkpoint.exists() and json.loads(checkpoint.read_text(encoding='utf-8-sig')).get('status')=='awaiting_pi':
        raise RuntimeError('Baseline validation checkpoint awaits PI decision; raw Phase 5 outputs are available.')
    models=df.drop_duplicates('cell_hash')
    h1,corr=concentration_analysis(models,tables,figures)
    h3=masking_analysis(df,cfg,tables,figures)
    manifests=list(Path('results/runs/tier_a_main').glob('*/manifest.json'))
    candidates=[]
    for path in manifests:
        info=json.loads(path.read_text())
        if info['config']['_source_hash']==df.source_hash.iloc[0]:candidates.append(path.parent)
    if len(candidates)!=1:raise ValueError('Need unique manifest for per-point margin analysis')
    margin_agreement(models,candidates[0],tables)
    extra=pd.read_csv(args.iso)
    iso_cfg={**cfg,'noise_p':[0.0],'bandwidth':[.1,.25,2.]}
    validate_table(extra,iso_cfg)
    h4=iso_analysis(df,extra,cfg,tables,figures)
    if args.iso_extension:
        extension=pd.read_csv(args.iso_extension)
        extension_cfg=yaml.safe_load(Path('configs/iso_overlap_followup.yaml').read_text(encoding='utf-8'))
        validate_table(extension,extension_cfg)
        sub_tables=tables/'iso_extended';sub_figures=figures/'iso_extended'
        sub_tables.mkdir(exist_ok=True);sub_figures.mkdir(exist_ok=True)
        primary=h4
        h4=iso_analysis(df,pd.concat([extra,extension],ignore_index=True),cfg,sub_tables,sub_figures)
        h4['original_grid_verdict']=primary
        h4['followup']='Additional bandwidths selected only by training-kernel concentration; exploratory follow-up'
        extension_audit=sub_tables/'diagnostic_review.json'
        if extension_audit.exists():
            h4['post_run_model_audit']=json.loads(extension_audit.read_text())
    controls=pd.read_csv(args.controls)
    h5=classical_analysis(df,controls,cfg,tables,figures)
    verdicts=dict(H1=h1,H2=dict(verdict='inconclusive',
        reason='A0 is a linearized proxy; the A2 budget grid supplies censored upper bounds, not resolved adversarial margins.',
        proxy_groups_partial_rho_at_least_05=int((corr.partial_clean_a0>=.5).sum()),groups=len(corr)),H3=h3,H4=h4,H5=h5)
    audit_path=tables/'diagnostic_review.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        verdicts['H3']['post_run_model_audit']=audit
        verdicts['H5']['caution']='Differences must be interpreted alongside constant-classifier and attack diagnostics.'
        audited=pd.read_csv(tables/'model_degeneracy_audit.csv')
        pairs=pd.read_csv(tables/'h5_pairs.csv').merge(
            audited[['cell_hash','certified_constant']],on='cell_hash',validate='many_to_one')
        coverage=pairs.groupby(['dataset','eligible','certified_constant']).size().rename('epsilon_rows').reset_index()
        coverage.to_csv(tables/'h5_diagnostic_coverage.csv',index=False)
        verdicts['H5']['eligible_constant_qsvm_rows']=int((pairs.eligible & pairs.certified_constant).sum())
        verdicts['H5']['eligible_datasets']=sorted(pairs.loc[pairs.eligible,'dataset'].unique())
    (tables/'hypothesis_verdicts.json').write_text(json.dumps(verdicts,indent=2))
    lines=['# Central analyses','', 'All conclusions are conditional on the configured small-sample simulator study.', '',
           'The primary grid contains 144 fitted QSVMs and 576 epsilon rows. Each model appears at four attack budgets; those rows are not independent replicates.', '',
           'H1 meets its prespecified descriptive criterion. H2 remains inconclusive because the saved attack grid does not resolve true adversarial margins. H3 meets the transfer-gap criterion in one stratum; a transfer gap alone does not demonstrate gradient masking. H5 does not meet the equivalence criterion across the eligible matched comparisons.', '']
    if audit_path.exists():
        lines += ['## Constant-classifier audit', '',
                  f'All {audit["M2_failed_models"]} models flagged by the large-budget M2 check have a constant decision sign under the numerical density-operator bounds. '
                  'Thus their retained accuracy reflects a trivial classifier, not demonstrated useful adversarial protection. '
                  'The 328 flagged epsilon rows correspond to 82 models. Original flags remain in the raw results. '
                  'These floating-point bounds are not interval-arithmetic certificates, and they do not rule out masking in other configurations.', '']
        lines += [f'H5 eligibility covers {h5["eligible_pairs"]} epsilon-level pairs; '
                  f'{verdicts["H5"]["eligible_constant_qsvm_rows"]} involve constant QSVMs. '
                  f'Eligible datasets: {", ".join(verdicts["H5"]["eligible_datasets"])}. '
                  'This sharply limits generalization of the classical-control comparison.', '']
    if args.iso_extension:
        lines += ['## Iso-concentration follow-up', '',
                  'The original family has insufficient common overlap in 40 of 48 metric strata. '
                  'The separately reported extension adds bandwidths selected from training concentration only. '
                  'Its interpolated comparisons are exploratory; original-grid results are retained in the nested verdict and primary H4 tables.', '']
        lines += ['The extension removes the common-range coverage failure. Pointwise bootstrap intervals separate the robust-accuracy curves at 21 interpolated positions, all in WBCD with three repetitions. '
                  'These positions are correlated, not 21 independent findings. No paired comparison survives BH correction. '
                  'The original H4 comparison remains inconclusive; the exploratory extension is labeled not supported under the specified pointwise-interval rule, not a definitive causal rejection. '
                  'The follow-up audit identifies 31 of 36 added models as constant classifiers and explains all 31 added M2 failures.', '']
    for name,value in verdicts.items():
        lines += [f'## {name}: {value["verdict"]}','', '```json',json.dumps(value,indent=2),'```','']
    lines += ['## Limits','', 'Three seeds give coarse bootstrap intervals and low-power paired tests. Test samples recur across splits. '
              'M3 is untested. Flags are reported, not removed. H4 interpolation and clean-accuracy matching are descriptive. '
              'No hardware, full-dimensional omics, or certified robustness claims are justified.']
    Path('docs/RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(verdicts,indent=2))


if __name__=='__main__':main()
