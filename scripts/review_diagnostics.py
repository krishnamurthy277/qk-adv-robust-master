"""Post-run diagnostic audit; does not tune or overwrite production results.

For A=sum_i alpha_i*y_i*rho_i, f(x)=Tr(rho_x A)+b. Every valid
density matrix has unit trace and is PSD, so f lies between b+eigmin(A)
and b+eigmax(A). A strict common sign proves a constant decision over
all feature states (numerically evaluated certificate, tolerance 1e-8).
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import json
import argparse
import numpy as np
import pandas as pd
import torch
import yaml
from scripts.run_experiment import load_dataset,fit_svm_precomputed
from src.data.split import get_train_test_split
from src.data.preprocessing import QKPreprocessor
from src.kernel.fidelity import feature_density,density_overlap
from src.utils.seed import seed_everything


def decision_bounds(densities,coefficients,intercept):
    operator=torch.einsum('i,ijk->jk',coefficients.to(torch.complex128),densities)
    operator=(operator+operator.conj().T)/2
    values=torch.linalg.eigvalsh(operator)
    return float(values[0]+intercept),float(values[-1]+intercept)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',default='results/tables/tier_a_results.csv')
    parser.add_argument('--output',default='results/tables/analysis')
    parser.add_argument('--skip-scan',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    cfg=yaml.safe_load(Path('configs/tier_a_main.yaml').read_text())
    main=pd.read_csv(args.results).drop_duplicates('cell_hash')
    rows=[];scan=[]
    bandwidths=[.05,.075,.1,.15,.2,.25,.35,.5,.65,.75,.85,1.,1.15,1.25,1.5,1.75,2.,2.5,3.,4.,6.,8.]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for (dataset,seed),part in main.groupby(['dataset','seed']):
        seed_everything(int(seed))
        raw,y=load_dataset(dataset,cfg['n_qubits'])
        tr,te,ytr,yte=get_train_test_split(raw,y,cfg['train_size'],cfg['test_size'],int(seed))
        prep=QKPreprocessor(cfg['n_qubits'],cfg.get('omics_select_k',1000) if dataset=='gse2034' else None)
        X=prep.fit_transform(tr,ytr);Y=prep.transform(te)
        ytr=np.where(ytr<=0,-1,1);yte=np.where(yte<=0,-1,1)
        mask=~np.eye(len(X),dtype=bool)
        for model in part.itertuples():
            with torch.no_grad():
                rho=feature_density(X,model.reps,model.bandwidth,model.noise_p)
                gram=density_overlap(rho,rho).numpy()
                svm,C=fit_svm_precomputed(gram,ytr)
                coefficients=torch.tensor(svm.dual_coef_[0],dtype=torch.float64)
                lower,upper=decision_bounds(rho[svm.support_],coefficients,float(svm.intercept_[0]))
                test_rho=feature_density(Y,model.reps,model.bandwidth,model.noise_p)
                decisions=density_overlap(test_rho,rho[svm.support_])@coefficients+svm.intercept_[0]
                predicted=np.where(decisions.numpy()>0,1,-1)
                clean=float(np.mean(predicted==yte))
            if C!=model.C_chosen or abs(clean-model.clean_acc)>1e-10:
                raise ValueError('Reconstruction does not match original fitted model')
            row=dict(cell_hash=model.cell_hash,dataset=dataset,seed=seed,reps=model.reps,
                     bandwidth=model.bandwidth,noise_p=model.noise_p,C=C,
                     clean_acc=clean,majority_test_acc=float(max(np.mean(yte==1),np.mean(yte==-1))),
                     positive_prediction_fraction=float(np.mean(predicted==1)),
                     balanced_acc=float(np.mean([np.mean(predicted[yte==label]==label) for label in [-1,1]])),
                     global_lower=lower,global_upper=upper,
                     certified_constant=bool(lower>1e-8 or upper < -1e-8),
                     original_M2_passed=bool(model.M2_passed),
                     original_large_budget_robust_acc=model.a2_large_robust_acc)
            rows.append(row)
        # Concentration-only scan: training features; NO test outcomes used.
        for reps in ([] if args.skip_scan else cfg['reps']):
            for c in bandwidths:
                with torch.no_grad():
                    rho=feature_density(X,reps,c,0)
                    gram=density_overlap(rho,rho).numpy()
                variance=float(gram[mask].var())
                scan.append(dict(dataset=dataset,seed=seed,reps=reps,bandwidth=c,
                                 var_K=variance,log10_var_K=np.log10(variance)))
        pd.DataFrame(rows).to_csv(output/'model_degeneracy_audit.csv',index=False)
        if not args.skip_scan:
            pd.DataFrame(scan).to_csv(output/'bandwidth_concentration_scan.csv',index=False)
        print(f'Audited {dataset}, seed {seed}',flush=True)
    frame=pd.DataFrame(rows)
    failed=frame[~frame.original_M2_passed]
    summary=dict(models=len(frame),M2_failed_models=len(failed),
                 M2_failed_certified_constant=int(failed.certified_constant.sum()),
                 M2_failed_unresolved=int((~failed.certified_constant).sum()),
                 constant_models=int(frame.certified_constant.sum()),
                 note='Constant classifiers are uninformative robustness, not demonstrated gradient masking. Original M2 flags retained.')
    (output/'diagnostic_review.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
