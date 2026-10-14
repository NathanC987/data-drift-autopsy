# Component Rationale

Why each design choice was made. Every choice below replaces an earlier
hand-tuned constant or an invalid test.

- **One significance level, family-wise control.** Every decision is a test at
  alpha with Holm or Bonferroni correction, so false-alarm rates are known in
  advance and verified on no-drift control windows (ACS held-out test split,
  CLEAR-10 bucket 1 test split, benchmark controls).
- **Nominal features are categorical.** Census codes such as occupation have
  no order; KS and Euclidean kernels on them are meaningless. They are tested
  with chi-square homogeneity, one-hot encoded for the model and MMD, and
  scored by rarity for OOD.
- **Two-sample chi-square, not goodness of fit.** The reference is itself a
  sample; treating its proportions as exact is anti-conservative.
- **Permutation tests for embedding detectors.** MMD, PCA reconstruction and
  the Frechet distance have no convention for a threshold on embeddings; a
  permutation test gives a calibrated p-value instead of a tuned cut-off.
- **CBPE calibrated on held-out data.** Calibrating on training data inherits
  the model's optimism.
- **SHAP on the full model's probability.** Explaining hard labels gives
  near-zero, noisy attributions; explaining a classifier stripped of its
  preprocessing explains a different model. Bootstrap intervals separate real
  attribution changes from sampling noise.
- **BBSE plus label-shift consistency for prior vs covariate.** Under pure
  prior shift P(X|Y) is fixed, so reweighting the reference by the estimated
  class ratio must reproduce the production inputs; what remains is covariate
  shift (Lipton et al., 2018).
- **Error-structure test relative to the reference.** The model's error
  pattern depends on the inputs even without drift; testing the *change* in
  errors, with a per-audit confidence curve in the null, separates a
  feature-localised rule change from unstructured label noise.
- **Learned reliability fusion.** Hand-set weights have no meaning; a logistic
  model fitted on held-out labelled data turns the signals into an estimated
  error probability, with labels tied to the reference error rate and 0.5.
- **Population signals stay at window level.** The CBPE estimate, confidence
  shift and explanation shift describe a window, not a row.
- **Label-free triage by importance-weighted validation.** Grading probes with
  production labels would not be label free. Importance-weighted validation is
  unbiased under covariate shift and honestly blind to changes in P(Y|X).
- **Benchmark replication.** Five reference distributions and ten seeds per
  type and intensity, with Wilson and bootstrap intervals, replace a single
  13-window run.
