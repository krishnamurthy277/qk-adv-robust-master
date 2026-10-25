# Methods draft — corrected implementation

Status: implementation-based draft. Numerical findings belong in the generated `RESULTS.md` after the complete grid and analyses. This text describes simulator experiments, not hardware robustness.

## Data and preprocessing

The corrected main configuration contains WBCD and GSE2034, three stratified split seeds (0, 1, 2), and 60 training plus 60 test samples per split. WBCD is loaded with scikit-learn. GSE2034 is loaded from the local GEO series-matrix file; the parsed target is bone relapse. The local matrix has 286 samples and 22,283 probe rows, with 217/69 binary label counts. No claim is made that probe rows are distinct genes or that this endpoint exhausts the cohort's clinical annotations.

All transformations are fitted on the outer training split. Median imputation precedes further transformations. For GSE2034 only, the 1,000 highest ANOVA F-score features are selected using training labels. Features are standardized, reduced by full-SVD PCA to five components, and min–max scaled to [0, pi]. Test transformations use those fitted parameters and are clipped to that box. Perturbations are evaluated in this reduced, scaled feature space, not as interventions in the original biomedical measurements. The test set is not used to tune hyperparameters.

Five qubits and 60/60 samples are documented compute-budget deviations from the original six-qubit, 100/100 plan. Synthetic data is retained for validation, not substituted for the omics dataset in the corrected main grid.

## Feature map and exact noise

Each repetition applies H on every qubit, then RZ(2 c x_i), followed by a linear chain of CX(i,i+1), RZ(2(pi-c x_i)(pi-c x_(i+1))) on i+1, and CX(i,i+1). RZ(theta) has diagonal entries exp(-i theta/2) and exp(i theta/2). This matches Qiskit's ZZFeatureMap phase convention up to global phases. The grid uses repetitions 1, 2, 3 and bandwidths c=0.5,1.0.

After every H and RZ, apply a one-qubit depolarizing channel with total nonidentity-Pauli probability p/10. After every CX, apply the two-qubit channel on its complete support with probability p. The grid uses p=0,0.002,0.01,0.03. For a k-qubit gate the convention is

`E(rho) = (1-p_gate)rho + p_gate/(4^k-1) sum_(P != I) P rho P`.

The equivalent replacement-channel parameter is `lambda = p_gate*4^k/(4^k-1)`. No shot sampling is used in the main experiment. State tensors and differentiable computations use complex128/float64.

The kernel is the probability of all-zero measurement after the noisy forward circuit for x followed by the noisy inverse circuit for x'. Noise follows every elementary gate in both halves; self-kernel values are not normalized to one.

An exact algebraic factorization accelerates evaluation. A depolarizing channel on a gate's complete support is self-adjoint and commutes with conjugation by that gate. The noisy inverse circuit is therefore the adjoint of the noisy forward channel. If `rho_x` denotes the noisy forward output, then `k(x,x') = Tr(rho_x rho_x')`. This reuses forward densities instead of simulating all input pairs separately. Local gates use tensor reshaping, and depolarization uses partial traces rather than full-system Kraus sums. Fixed support-vector densities are cached only within their fitted model; autograd still differentiates the complete noisy forward channel for each attacked input.

Independent validation compares this factorization against explicit Qiskit Aer density-matrix circuits with noise after every H/RZ/CX, without transpiler optimization. Noiseless checks cover five random pairs at each of n=4 and n=6 with maximum absolute error <1e-8. Noisy checks use n=4, repetitions 1–3, two bandwidths, three noise levels, and self/pair kernels with maximum absolute error <1e-6. Central finite differences validate decision gradients for both p=0 and p=0.03 at relative error <1e-5.

The restored noiseless-blob sanity check retains the original >=0.90 accuracy criterion. After a fixed reps=2,c=1 model scored 0.65, the PI delegated the protocol decision: use training-only CV over the already planned repetitions, bandwidths and C values. The selected model is evaluated once on held-out points. This post-failure protocol amendment is disclosed; the fixed-bandwidth failure is preserved and does not alter the production grid. The diagnostic selected reps=1,c=0.1,C=10 and obtained held-out accuracy 1.0. This is a sanity check on synthetic data, not a main-study accuracy claim.

## SVMs and kernel statistics

QSVMs use `SVC(kernel='precomputed')`. C is selected from {0.1,1,10,100} using inner three-fold GridSearchCV slices of the outer-training Gram matrix. Preprocessing is fitted once on the outer training split, not refitted inside each inner kernel slice; the held-out outer test set remains unused for fitting. Attacks evaluate the same dual coefficients, intercept and support-vector kernel as the deployed decision function.

Concentration statistics use the raw training Gram matrix: off-diagonal mean, population variance, coefficient of variation, log10 variance, distance of the mean from 1/2^n, largest-eigenvalue/trace ratio, entropy-based effective rank, and kernel-target alignment. The geometric margin uses `1/sqrt(alpha_y.T K_SV alpha_y)`.

The minimum eigenvalue is measured before any repair. If below -1e-8, negative eigenvalues are clipped to zero for fitting; the original minimum and repair Frobenius norm are retained. The exact density-overlap kernel is theoretically PSD, so a substantial negative eigenvalue would warrant investigation rather than routine interpretation. Concentration metrics continue to describe the raw matrix.

## Adversarial evaluation

Budgets are L2 radii equal to {0.05,0.1,0.2,0.4} times the median training-pair Euclidean distance. Every candidate is projected into its original-point L2 ball and clipped to [0,pi]^5.

A0 is the local linearized distance estimate `abs(f(x))/norm(grad f(x))` on initially correct target points. It is not a certified lower bound or an exact adversarial margin. The denominator is clamped at 1e-12 for numerical stability; vanishing-gradient diagnostics accompany the estimate.

A2 minimizes the signed target margin y f(x) with 40 normalized-gradient steps, step size eps/10, and three starts. The first start is the clean point; the remaining two use random directions and uniformly sampled radii inside the ball. The best clean, initial, intermediate and final candidate is retained. Points already misclassified are counted as nonrobust. Gradient batches contain at most eight points to bound memory use.

A1 generates candidates with the same PGD procedure on a noiseless surrogate at matching repetitions and bandwidth, then evaluates them using the noisy target's own support vectors and coefficients. Surrogate-misclassified points are left at their clean inputs by the surrogate attack. Both robust accuracy and conditional attack success are scored against the target's initial correctness, not the surrogate's. Originally incorrect target points never count as robust even if a transfer perturbation corrects them.

Robust accuracy is the fraction of the entire test set initially and finally correct. Attack success rate is the fraction of initially correct target points made incorrect. Seeds control data splits and random starts. Independent attack runs at successive epsilon values can produce nonmonotonic empirical outcomes; raw results are retained, not smoothed or selectively discarded.

M1 compares A2 with its one-step version (step eps/10, clean start), M2 uses radius five times the median training distance with the same 40 steps/three starts, M4 records gradient-norm summaries, and M5 compares against a clean-start-only 40-step run. M1/M5 are consistency checks because A2 includes the clean-start trajectory. They are not independent evidence ruling out masking. M3 is not evaluated: no black-box attack or shot-noise/EOT experiment is included. Existing diagnostic thresholds are preserved, including M2 robust accuracy <=0.1, M4 median gradient norm >1e-6 and M1/M5 tolerance 0.02.

## Central analyses and classical controls

`ANALYSIS_SPECIFICATION.md` records the operational choices before interpretation. Seed-level percentile bootstrap intervals use 5,000 draws. Paired Wilcoxon tests and Benjamini–Hochberg corrections accompany effect sizes, with explicit low-power warnings for three seeds. Grid cells and epsilon rows are not treated as independent statistical replicates. Repeated random splits can share samples.

Phase 6 adds noiseless bandwidths c=0.1,0.25,2.0 to complete the specified five-value bandwidth family. Noise-only curves hold c=1.0 fixed. Comparisons interpolate in overlapping log10-variance ranges at fixed dataset, repetition and seed, without extrapolation. Shared anchors cannot provide independent evidence of equivalence. Overlapping confidence intervals are not proof of a causal concentration mechanism.

RBF-SVMs use the same preprocessed splits, C grid and attack settings. One control selects gamma by training CV from {scale,auto,0.01,0.1,1.0}. Another matches training-kernel off-diagonal variance to each QSVM. Because RBF variance need not be monotone in gamma, matching searches log gamma from -16 to 12 and chooses the smallest root deterministically; unattainable targets retain their residual mismatch. Matching never uses test labels. Clean-accuracy matching is a descriptive reporting filter (absolute difference <=0.03), with eligible counts disclosed. Gaussian-kernel derivatives are computed analytically through differentiable tensor expressions.

## Reproducibility and limitations

Post-run review reconstructs the fitted QSVMs and forms the Hermitian operator A=sum_i(alpha_i y_i rho_i). Since f(x)=Tr(rho_x A)+b and every feature density has unit trace and is positive semidefinite, its decision lies between lambda_min(A)+b and lambda_max(A)+b. A common sign separated from zero by 1e-8 identifies a globally constant classifier within numerical precision. These are floating-point bounds, not interval-arithmetic certificates. Original M2 flags are retained; constant classifiers are distinguished from unresolved attack failure and gradient masking.

After the original iso-concentration comparison showed insufficient overlap, a training-concentration-only scan considered c in {0.05,0.075,0.1,0.15,0.2,0.25,0.35,0.5,0.65,0.75,0.85,1,1.15,1.25,1.5,1.75,2,2.5,3,4,6,8}. From candidates outside the original five values, two were selected to maximize the number of dataset/repetition groups with common cross-seed overlap wider than 0.001 in log10 variance, then total positive overlap width, with ascending bandwidth ties. This selects c=4 and c=6. No test accuracy or attack outcomes enter selection. All 36 additional cells are reported separately as an exploratory Phase 6 extension; the original comparison remains available.

Each production run stores its complete config, data/content fingerprints, computational-source fingerprint, source snapshot, platform/Python information, raw Gram caches, per-point attack outcomes and per-cell CSV checkpoint. Each worker has a separate cache namespace. Checkpoints and aggregate CSVs are replaced atomically; failed cells retain traceback records and can be retried. Old-model results are preserved and excluded from corrected analyses.

The production environment is Windows, Python 3.10.3, recorded in run manifests and `requirements-lock.txt`, an explicit deviation from the guide's Python 3.11+ requirement. A fresh Python 3.12.14 installation pinned in `requirements-repro-lock.txt` passes all 34 tests and a newly computed two-cell smoke experiment. All 37 non-timing numeric fields compared in the four smoke rows match exactly across environments. This verifies the small pipeline, not an independent full-grid replication.

The study uses small samples, only three split seeds, five PCA dimensions, idealized gate-local depolarization and empirical attacks. It cannot establish hardware-level protection, clinical utility, full-dimensional omics robustness, certified margins, or the absence of gradient masking. AI-assisted implementation and drafting should be disclosed according to the eventual venue's policy.
