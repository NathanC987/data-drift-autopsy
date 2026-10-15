# Codebase Headstart

Fast orientation for a developer joining the project.

## What the project does

Label-free monitoring of a deployed classifier. Given the model, a labelled
reference set and unlabelled production windows, the pipeline:

1. estimates accuracy without labels (CBPE),
2. tests whether the inputs, the predictions and the confidence distribution moved,
3. localises the shift to features (or embedding dimensions and predicted classes),
4. explains it (SHAP attribution change for tables; CLIP concept probe and class activation maps for images),
5. names the drift type (covariate, prior, concept, label noise) by hypothesis tests, optionally with a small delayed-label audit,
6. scores the reliability of every prediction,
7. predicts, without production labels, whether a cheap retrain will help.

Every decision is a hypothesis test at one significance level (default 0.05) with family-wise error control.

## Package map (`src/drift_autopsy/`)

| Module | Role |
|---|---|
| `core/` | `Dataset` (with `categorical_features`), result dataclasses, `DriftPipeline` (detect -> localise -> RCA, slice mode) |
| `stats.py` | every test the package uses: Holm/Bonferroni and weighted Holm, exact logistic likelihood (Newton, float64, CPU or GPU), severity ladders, chi-square homogeneity with Cochran pooling, weighted KS / chi-square (Kish), two-proportion test, Wilson interval, paired bootstrap, generic permutation test |
| `estimation.py` | CBPE: histogram-binning map from confidence to accuracy |
| `detectors/` | KS (numeric only, corrected), PSI, MMD (permutation test, mixed-type encoding), PCA reconstruction and Frechet distance (permutation tests), domain classifier, CBPE (confidence-shift test plus estimate) |
| `localizers/univariate.py` | KS for numeric, chi-square homogeneity for nominal features, Holm step-down; effect sizes (KS statistic, Cramer's V) and unseen-category share |
| `rca/` | `SHAPAnalyzer` (explains the full model's probability, bootstrap intervals), `concept_probe.py` (CLIP), `cam.py` (class activation maps for a linear head on a frozen ResNet) |
| `diagnosis.py` | drift-type diagnosis: label-free stage (input, prediction and MMD tests under one weighted Holm family; BBSE; label-shift consistency) and audit stage (accuracy-drop test; error-structure likelihood-ratio tests) |
| `reliability/` | per-prediction signals (confidence risk, OOD, instability), learned logistic fusion, risk labels |
| `remediation/` | strategies with paired-bootstrap grading, density-ratio weights, label-free triage (importance-weighted validation) |
| `benchmark/` | drift injectors, per-window evaluation, multi-state multi-seed orchestration, aggregation with confidence intervals |
| `experiments/` | shared setups: `acs.py` (ACS Income model, split, feature typing) and `clear10.py` (CLEAR-10 embeddings, model, split) |
| `data/` | loaders (Folktables cache, CLEAR-10), multiclass proxy metrics, validators, generic image tabularizer |

## Reproducing the paper

```bash
pip install -e ".[all,dev]"
python examples/quickstart/folktables_demo.py            # ACS temporal + geographic study
python examples/quickstart/clear10_tabularize_demo.py    # once: ResNet-18 embeddings (GPU recommended)
python examples/quickstart/clear10_drift_study.py        # CLEAR-10 study
python examples/quickstart/clip_concept_probe_demo.py    # concept probe (needs the CLEAR-10 study output)
python examples/quickstart/remediation_demo.py           # remediation + triage (needs both studies)
python examples/quickstart/drift_injection_benchmark.py  # 650-window benchmark (parallel)
python examples/paper/build_artifacts.py                 # every paper number, table and figure
pytest                                                   # unit + integration tests
streamlit run examples/dashboard/app.py                  # dashboard
```

All randomness is seeded (default 42); remediation fits run on one BLAS thread so results are bit-for-bit reproducible.

## Reading order

1. `experiments/acs.py` and `examples/quickstart/folktables_demo.py` (one full study end to end)
2. `stats.py`, `detectors/`, `localizers/univariate.py`
3. `diagnosis.py`, `reliability/`, `remediation/triage.py`
4. `benchmark/evaluate.py`, `benchmark/window.py`, `benchmark/metrics.py`
5. `examples/paper/build_artifacts.py` (how outputs become paper content)
