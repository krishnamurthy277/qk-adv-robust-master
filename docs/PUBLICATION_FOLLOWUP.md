# Publication follow-up protocol

Recorded 30 September 2026 before follow-up outcomes. These are post-review,
exploratory extensions, not preregistered or replacements for the original grid.

1. Reconstruct original splits and use saved per-point attack outcomes to report
   clean and robust balanced accuracy, the training-majority baseline, prediction
   diversity, and constant-sign versus unresolved strata. Keep all configurations.
2. For every one of the 62 spectrally unresolved primary models, evaluate a
   gradient-free random search at relative budget 0.4 on all 60 held-out points.
   Use 128 candidates per point: half uniform in ball volume and half on its
   surface, clipped to the feature domain. Include the clean point. Report the
   union with saved PGD successes, rather than replacing PGD by a weaker attack.
   This bounded query check is not an exhaustive attack or robustness certificate.
   Fixed RNG seed 20260930 plus original split seed. Verify every candidate's
   domain and perturbation constraints. Save individual outcomes.
3. On ten additional split seeds 3 through 12 for both datasets, use 60 training
   and 60 test samples and five qubits. Compare a training-selected unweighted
   SVM against a training-selected class-balanced SVM. Restrict to one repetition,
   bandwidths 0.1, 0.5, 1; noise 0, 0.03; C 0.1, 1, 10, 100. Select by mean inner
   three-fold balanced accuracy. Refit imputation, feature selection, PCA and
   scaling within each inner fold. Lexicographic grid order breaks ties. Audit
   the resulting decision bounds and test utility. This is a utility/sensitivity
   experiment, not a new adversarial comparison or replication of H1–H5.
4. Report all new splits and outcomes, including unfavorable findings. Ten
   repeated splits are not ten independent cohorts. No significance claim is
   based on treating grid configurations or test budgets as independent samples.

Original results and manifests remain unchanged. Follow-up outputs reside under
`results/publication_followup/`, with their own source/data hashes and environment.
