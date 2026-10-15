# Capability Status (As Implemented)

Status labels: Implemented, Partial, Not Implemented.

## Detection

| Capability | Status | Decision rule |
|---|---|---|
| KS test | Implemented | numeric features only; Bonferroni (or Holm) over features; p < alpha |
| PSI | Implemented | quantile bins with open outer edges, category cells with Cochran pooling, Jeffreys smoothing; PSI >= threshold (conventional 0.2) |
| MMD | Implemented | permutation p < alpha; median-heuristic bandwidth fixed across permutations; numeric z-scoring optional; nominal features one-hot scaled to unit distance |
| PCA reconstruction | Implemented | permutation p < alpha on the relative rise in out-of-sample reconstruction error |
| Frechet distance | Implemented | permutation p < alpha |
| Confidence-shift test + CBPE estimate | Implemented | two-sample chi-square on confidence bins; CBPE estimate when the reference is labelled; baselines average confidence, DoC and ATC (`estimation.label_free_estimates`) |
| Domain classifier (classifier two-sample test) | Implemented | gradient boosted trees on half of equal-sized reference / production samples; exact binomial test of held-out accuracy against 1/2; baseline in every study |
| BBSD | Implemented | KS on each class probability, Bonferroni over classes; baseline in every study |

## Localization and root cause

| Capability | Status |
|---|---|
| Per-feature localization (KS / chi-square homogeneity, Holm) with effect sizes and unseen-category share | Implemented |
| Slice localization (metadata column or predicted class) | Implemented |
| SHAP attribution change with Bonferroni-adjusted bootstrap intervals | Implemented |
| CLIP concept probe with per-concept intervals and permutation-tested tag validation | Implemented |
| Class activation maps for a linear head on a frozen ResNet (verified against the logit) | Implemented |

## Diagnosis

| Capability | Status |
|---|---|
| Label-free: none / covariate / prior via input, prediction and MMD tests, BBSE and weighted residual tests | Implemented |
| Audit: accuracy-drop test; concept vs label noise via error-structure likelihood-ratio tests | Implemented |
| Mixed drift (several types at once) | Partial (the verdict names the dominant label-free finding; the evidence is reported in full) |
| Baselines (input monitoring, BBSE first, error-rate monitoring), learned meta-classifier and ablations (`diagnosis_baselines.py`) | Implemented |

## Reliability

| Capability | Status |
|---|---|
| Confidence risk, OOD (Isolation Forest + standardised distance on numeric z-scores and category rarity), instability (reference-scaled noise on numeric features) | Implemented |
| Learned logistic fusion with interpretable risk labels | Implemented |
| Evaluation against labels (AUROC, error rate per label) | Implemented |
| Text, audio and raw-image reliability inputs | Not Implemented (feature vectors only) |

## Remediation

| Capability | Status |
|---|---|
| Strategies: importance-weighted retrain, retrain on recent, full retrain, feature drop, label retrain, isotonic recalibration | Implemented |
| Paired-bootstrap grading of every strategy | Implemented |
| Label-free triage (importance-weighted validation, Holm across probes) | Implemented |
| Automated retraining or rollback execution | Not Implemented |

## Evaluation

| Capability | Status |
|---|---|
| Drift-injection benchmark: design A (4 types x 3 intensities + control) and design B (5 held-out mechanisms x 3 intensities), ACS Income and ACS Employment, 5 states x 10 seeds, two model families, paired tests against baselines | Implemented |
| Real-drift studies: ACS Income and ACS Employment (CA 2014 -> 2015-2019, 2021-2023; four states), UCI gas sensor array drift, CLEAR-10 temporal; logistic regression and gradient boosting | Implemented |

## Runtime

| Capability | Status |
|---|---|
| Batch execution | Implemented |
| Streaming execution | Not Implemented |
| Enterprise controls (RBAC, audit log, SLOs) | Not Implemented |
