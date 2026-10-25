# Central analyses

All conclusions are conditional on the configured small-sample simulator study.

The primary grid contains 144 fitted QSVMs and 576 epsilon rows. Each model appears at four attack budgets; those rows are not independent replicates.

H1 meets its prespecified descriptive criterion. H2 remains inconclusive because the saved attack grid does not resolve true adversarial margins. H3 meets the transfer-gap criterion in one stratum; a transfer gap alone does not demonstrate gradient masking. H5 does not meet the equivalence criterion across the eligible matched comparisons.

## Constant-classifier audit

All 82 models flagged by the large-budget M2 check have a constant decision sign under the numerical density-operator bounds. Thus their retained accuracy reflects a trivial classifier, not demonstrated useful adversarial protection. The 328 flagged epsilon rows correspond to 82 models. Original flags remain in the raw results. These floating-point bounds are not interval-arithmetic certificates, and they do not rule out masking in other configurations.

H5 eligibility covers 168 epsilon-level pairs; 160 involve constant QSVMs. Eligible datasets: gse2034. This sharply limits generalization of the classical-control comparison.

## Iso-concentration follow-up

The original family has insufficient common overlap in 40 of 48 metric strata. The separately reported extension adds bandwidths selected from training concentration only. Its interpolated comparisons are exploratory; original-grid results are retained in the nested verdict and primary H4 tables.

The extension removes the common-range coverage failure. Pointwise bootstrap intervals separate the robust-accuracy curves at 21 interpolated positions, all in WBCD with three repetitions. These positions are correlated, not 21 independent findings. No paired comparison survives BH correction. The original H4 comparison remains inconclusive; the exploratory extension is labeled not supported under the specified pointwise-interval rule, not a definitive causal rejection. The follow-up audit identifies 31 of 36 added models as constant classifiers and explains all 31 added M2 failures.

## H1: supported

```json
{
  "verdict": "supported",
  "fraction_groups_meeting_both_axes": 0.8333333333333334,
  "groups": 6
}
```

## H2: inconclusive

```json
{
  "verdict": "inconclusive",
  "reason": "A0 is a linearized proxy; the A2 budget grid supplies censored upper bounds, not resolved adversarial margins.",
  "proxy_groups_partial_rho_at_least_05": 0,
  "groups": 6
}
```

## H3: supported

```json
{
  "verdict": "supported",
  "consistent_strata": 1,
  "diagnostic_failed_rows": {
    "M1_passed": 0,
    "M2_passed": 328,
    "M4_passed": 0,
    "M5_passed": 0
  },
  "M3": "untested; black-box attacks absent",
  "caution": "A transfer gap does not by itself prove gradient masking; diagnostics limit conclusions.",
  "post_run_model_audit": {
    "models": 144,
    "M2_failed_models": 82,
    "M2_failed_certified_constant": 82,
    "M2_failed_unresolved": 0,
    "constant_models": 82,
    "note": "Constant classifiers are uninformative robustness, not demonstrated gradient masking. Original M2 flags retained."
  }
}
```

## H4: not supported

```json
{
  "verdict": "not supported",
  "separated_comparisons": 281,
  "separated_robust_accuracy_points": 21,
  "separated_a0_proxy_points": 260,
  "bh_significant_points": 0,
  "insufficient_overlap": 0,
  "caution": "Pointwise intervals are not simultaneous bands or independent comparisons; A0 is only a proxy. Interpolation and three seeds limit evidence; no equivalence tolerance prespecified.",
  "original_grid_verdict": {
    "verdict": "inconclusive",
    "separated_comparisons": 0,
    "separated_robust_accuracy_points": 0,
    "separated_a0_proxy_points": 0,
    "bh_significant_points": 0,
    "insufficient_overlap": 40,
    "caution": "Pointwise intervals are not simultaneous bands or independent comparisons; A0 is only a proxy. Interpolation and three seeds limit evidence; no equivalence tolerance prespecified."
  },
  "followup": "Additional bandwidths selected only by training-kernel concentration; exploratory follow-up",
  "post_run_model_audit": {
    "models": 36,
    "M2_failed_models": 31,
    "M2_failed_certified_constant": 31,
    "M2_failed_unresolved": 0,
    "constant_models": 31,
    "note": "Constant classifiers are uninformative robustness, not demonstrated gradient masking. Original M2 flags retained."
  }
}
```

## H5: not supported

```json
{
  "verdict": "not supported",
  "total_pairs": 576,
  "eligible_pairs": 168,
  "complete_seed_strata": 12,
  "scope": "Eligible matched comparisons only; inspect coverage before generalizing.",
  "caution": "Differences must be interpreted alongside constant-classifier and attack diagnostics.",
  "eligible_constant_qsvm_rows": 160,
  "eligible_datasets": [
    "gse2034"
  ]
}
```

## Limits

Three seeds give coarse bootstrap intervals and low-power paired tests. Test samples recur across splits. M3 is untested. Flags are reported, not removed. H4 interpolation and clean-accuracy matching are descriptive. No hardware, full-dimensional omics, or certified robustness claims are justified.