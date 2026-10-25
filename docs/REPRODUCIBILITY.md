# Reproducibility record

Run commands from the repository root. Production used Python 3.10.3 on Windows; independent clean-environment verification used Python 3.12.14. Both use CPU PyTorch and float64/complex128. The optional GPU benchmark did not change production computation.

## Clean installation and checks

```powershell
python -m venv .venv-repro
.venv-repro/Scripts/python.exe -m pip install -r requirements-repro-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv-repro/Scripts/python.exe -m pip check
.venv-repro/Scripts/python.exe -m pytest tests -q --junitxml=results/validation_clean.xml
.venv-repro/Scripts/python.exe scripts/run_experiment.py --config configs/repro_smoke.yaml
.venv-repro/Scripts/python.exe scripts/compare_smoke_environments.py
```

Use Python 3.12 for the tested installation. The recorded result is 34 passed in 20.26 seconds; pip reports no broken requirements. Initial sandbox runs failed at pytest temporary-directory permissions, not scientific assertions. The successful run used normal Windows filesystem permissions. The initial error report is retained as `results/validation_clean_initial_sandbox.xml`.

The fresh smoke experiment uses a distinct configuration/run ID (`5acf75aa41539a65`) and computed both cells rather than reusing production checkpoints. Its four epsilon rows match all 37 non-timing numeric fields from the original smoke table exactly (`results/clean_smoke_comparison.json`). This does not claim bitwise replication of the entire Tier A grid on a different environment.

## Experiments and figures

The README gives commands for every input experiment and the analysis entry point. Main results are `results/tables/tier_a_results.csv`; the original noiseless supplement is `iso_concentration_results.csv`; classical controls are `classical_controls.csv`. The post-review supplement is `iso_overlap_followup_results.csv`. The latter must contain all 36 cells before generating extended analyses.

`scripts/analyze_results.py` regenerates F1-F6 and hypothesis tables. With `--iso-extension results/tables/iso_overlap_followup_results.csv`, original H4 figures/tables remain in their primary locations and follow-up comparisons are written under `figures/iso_extended/` and `tables/analysis/iso_extended/`. No original rows are discarded. `scripts/review_diagnostics.py` reproduces the constant-classifier audit and training-only bandwidth scan; `scripts/select_iso_bandwidths.py` reproduces the deterministic follow-up choice.

Before regenerating the final extended report, reproduce its additional classifier audit with:

```powershell
.venv/Scripts/python.exe scripts/review_diagnostics.py --results results/tables/iso_overlap_followup_results.csv --output results/tables/analysis/iso_extended --skip-scan
```

The GSE2034 series matrix must be available in `data/raw/`; the loader uses bone-relapse labels. Each manifest records the actual input fingerprints. WBCD is bundled with scikit-learn. No clinical effectiveness or hardware robustness is inferred.

## Resume and provenance

`results/artifact_manifest.json` records SHA-256 hashes of the final source, configs, documentation, tables, figures, validation outputs and run metadata. Verify the delivered snapshot with `.venv-repro/Scripts/python.exe scripts/artifact_manifest.py --verify`. Regenerating figures or updating logs can legitimately change hashes; build a new manifest explicitly only after reviewing those changes. Large binary caches and raw data are excluded from this small-artifact manifest; run manifests retain data fingerprints.

Run manifests and source snapshots live under `results/runs/<experiment>/<run-id>/`. Source, input and configuration fingerprints prevent reuse of incompatible cells. Completed cell CSVs are atomic checkpoints; an interrupted cell is recomputed. Keep the entire run directory with the aggregate table to preserve per-point outcomes and caches. Avoid concurrent writers for the same run.

Production was limited to ten logical CPUs with below-normal priority; worker count remains twelve and each worker uses one numerical thread. Scheduling changes affect wall time, not the scientific configuration. Tier B, black-box M3 and shot-noise/EOT experiments were not run. Reproducibility belongs to Phase 6; Phase 7 covers paper writing and final review.


## Publication follow-up completed 30 September 2026

The protocol is `docs/PUBLICATION_FOLLOWUP.md`. Run
`python scripts/publication_followup.py --mode all --jobs 1`, then
`python scripts/audit_publication_followup.py`. Jobs=1 was used for the verified
completion to avoid worker-process startup issues on this Windows session.
All original results are read-only inputs. Follow-up output is under
`results/publication_followup/`; 62 query cells and 20 selection cells are complete.
The query union found 25 additional model?point failures across 13 models.
The 40 selected models use ten additional split seeds per dataset and two
weighting modes, with fold-specific preprocessing. No selected model meets
the sufficient constant-sign bound; this does not establish robustness.

The recorded follow-up test report contains 36 passing tests with no failures
or skips. `audit.json` verifies coverage and provides numerical summaries.
The 476,160 query candidates are reproducible from the recorded generator,
seed and protocol; only pointwise minimum scores and success indicators are
stored. They are not saved as individual perturbation vectors. The query check
is bounded random search at one budget, not a full adaptive black-box benchmark.

`manuscript/publication_review_package.zip` is a local scientific review archive
with its own SHA-256 manifest. It is not a public release or a replacement for
original experimental manifests. Input GSE2034 data must be obtained separately.
