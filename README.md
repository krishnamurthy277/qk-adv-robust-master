# Kernel Concentration and Adversarial Robustness in Noisy QSVMs

An isolated, simulator-only study of concentration, adversarial vulnerability,
and apparent robustness caused by weak attacks. The corrected main study uses
WBCD and GSE2034, five qubits, 60/60 train/test samples and three seeds.

## Seven project phases

1. Project setup, simulator and kernel validation.
2. Data pipeline and classification baselines.
3. Concentration metrics and compute planning.
4. Adversarial attacks and diagnostic validation.
5. Main Tier A experiments.
6. Statistical analyses, figures and reproducibility package.
7. Paper writing and final submission review.

Phases 1–6 are complete. Phase 7 has a manuscript draft; final claim,
reference, layout and venue checks remain. Tier B/C experiments are outside
the current scope. Setup belongs to Phase 1 and reproducibility to Phase 6;
there is no separate Phase 0 or Phase 8.

## Current execution

The main grid is complete: 144 model cells and 576 epsilon rows, recorded in
`results/runs/tier_a_main/24d365adc00fcd58/`. The initial Phase 6 bandwidth
supplement and classical controls are also complete. The separate 36-cell
concentration-overlap follow-up, final figure review, and clean-environment
validation are complete. Status is in `results/workflow_status.json`.

The initial implementation's simplified noise model was invalid for the plan.
Old outputs are preserved locally but
must not be mixed with `v2_per_gate` results.

## Environment and validation

Production used Windows / Python 3.10.3, an earlier environment than the
3.11+ requirement, pinned in `requirements-lock.txt`. A fresh Python 3.12.14
environment now passes all 34 tests and an independently computed two-cell
smoke run. Its scientific numeric outputs match the production smoke exactly.
The tested clean environment is pinned in `requirements-repro-lock.txt`.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-repro-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pytest tests -q
```

GSE2034's series-matrix file must be in `data/raw/` (compressed or plain text).
The loader uses the bone-relapse annotation; preprocessing, imputation and
omics selection are fitted on each training split only. WBCD is bundled with
scikit-learn. No other project's source or results are imported.

## Reproduce the corrected study

Run commands from this project directory. On Windows, the terminal must permit
Joblib's multiprocessing pipes. Each worker uses one numerical-library thread.

```powershell
.venv/Scripts/python.exe scripts/run_experiment.py --config configs/smoke_test.yaml
.venv/Scripts/python.exe scripts/run_experiment.py --config configs/tier_a_main.yaml
```

The second command safely resumes completed **cells**. An interrupted cell is
rerun from its last completed cell checkpoint; there is no within-attack-step
checkpoint. Source/config/data fingerprints prevent incompatible cache reuse.
Source snapshots and manifests accompany each run. Never run two copies of the
same experiment simultaneously; a run lock rejects duplicate writers.

## Reproduce F1 and F2, then Phase 6

These commands reject incomplete or duplicate grids.

```powershell
# F1 concentration maps; F2 robustness curves; raw Phase 5 summary
.venv/Scripts/python.exe scripts/analyze_results.py --phase5-only

# Complete Phase 6's specified noiseless bandwidth family (not Tier B)
.venv/Scripts/python.exe scripts/run_experiment.py --config configs/iso_concentration.yaml

# Variance-matched and training-CV RBF controls
.venv/Scripts/python.exe scripts/run_classical_controls.py

# F1-F6, bootstrap intervals, paired tests, and H1-H5 assessments
.venv/Scripts/python.exe scripts/analyze_results.py

# Post-review overlap extension, selected from training variance only
.venv/Scripts/python.exe scripts/review_diagnostics.py
.venv/Scripts/python.exe scripts/select_iso_bandwidths.py
.venv/Scripts/python.exe scripts/run_experiment.py --config configs/iso_overlap_followup.yaml
.venv/Scripts/python.exe scripts/analyze_results.py --iso-extension results/tables/iso_overlap_followup_results.csv
```

Figures are written to `results/figures/` as PNGs, with names prefixed F1-F6.
Statistics and raw completion metadata are in `results/tables/analysis/`.
`docs/RESULTS.md` is generated only after all required analysis inputs exist.
`docs/ANALYSIS_SPECIFICATION.md` describes operational statistical choices and
limits. `docs/METHODS.md` describes the implemented protocol.

An optional supervisor can wait for a launched grid and execute these stages:

```powershell
.venv/Scripts/python.exe scripts/complete_workflow.py --tier-a-run results/runs/tier_a_main/24d365adc00fcd58
```

Its progress is `results/workflow_status.json`. It does not certify final visual
review, sufficient iso-concentration overlap, or a fresh dependency installation.
If those checks fail, further work is required. It never launches Tier B.

## Final verification

```powershell
.venv/Scripts/python.exe -m pytest tests -q --junitxml=results/validation_final.xml
.venv/Scripts/python.exe scripts/run_experiment.py --config configs/smoke_test.yaml
```

The existing environment and a clean dependency installation are distinct checks.
Clean-install evidence is in `results/validation_clean.xml` and
`results/clean_smoke_comparison.json`; see `docs/REPRODUCIBILITY.md`.

## Interpretation limits

A0 is a local approximation, not a certified margin. M3/black-box and shot-noise
EOT checks are absent from this grid. Passing the available checks cannot rule
out gradient masking. Three seeds give coarse intervals and weak statistical
power. Simulator results do not establish hardware or clinical robustness.

## Repository contents

This repository contains source code, tests, configurations, and scientific
documentation. Install dependencies from the requirements files; environments
and installed packages are excluded. Raw data, generated results, local work
logs, and manuscript working files are also excluded. Result paths and recorded
validation evidence above refer to local runs; regenerate outputs with the
documented commands.
