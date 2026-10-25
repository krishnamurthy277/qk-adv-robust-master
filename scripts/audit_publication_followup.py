"""Validate complete post-review evidence before manuscript integration."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'results/publication_followup'
q=pd.read_csv(out/'query_validation.csv'); s=pd.read_csv(out/'nested_selection.csv'); u=pd.read_csv(out/'utility_by_model_budget.csv')
a=pd.read_csv(ROOT/'results/tables/analysis/model_degeneracy_audit.csv')
assert set(q.cell_hash)==set(a.loc[~a.certified_constant,'cell_hash'])
assert len(q)==3720 and not q.duplicated(['cell_hash','test_index']).any()
assert q.groupby('cell_hash').size().eq(60).all() and q.queries.eq(128).all()
assert (q.union_correct==(q.pgd_correct & q.query_correct)).all()
assert not (q.query_correct & ~q.initially_correct).any()
assert len(s)==40 and not s.duplicated(['dataset','seed','class_weight']).any()
assert set(s.seed)==set(range(3,13)) and not s.constant.any()
assert len(u)==576 and u[u.constant].a2_balanced_acc.eq(.5).all()
rows=[]
for (d,h),g in q.groupby(['dataset','cell_hash']):
    row=dict(dataset=d,cell_hash=h,new_failures=int((g.pgd_correct & ~g.query_correct).sum()))
    for metric in ['pgd_correct','query_correct','union_correct']:
        row[metric]=float(g[metric].mean()); row[metric+'_balanced']=float(g.groupby('label')[metric].mean().mean())
    rows.append(row)
qm=pd.DataFrame(rows); qm.to_csv(out/'query_by_model.csv',index=False)
report={'query_models':62,'query_model_point_pairs':3720,'candidate_evaluations':int(q.queries.sum()),'additional_failure_pairs':int((q.pgd_correct & ~q.query_correct).sum()),'models_with_additional_failures':int((qm.new_failures>0).sum()),'query_dataset_means':qm.groupby('dataset').mean(numeric_only=True).to_dict('index'),'selection_summary':[]}
for (d,w),g in s.groupby(['dataset','class_weight']):
    report['selection_summary'].append(dict(dataset=d,class_weight=w,models=len(g),balanced_mean=float(g.balanced_acc.mean()),balanced_min=float(g.balanced_acc.min()),balanced_max=float(g.balanced_acc.max()),accuracy_mean=float(g.clean_acc.mean()),constant_bound_models=int(g.constant.sum()),single_label_test_models=int(g.positive_fraction.isin([0,1]).sum())))
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
