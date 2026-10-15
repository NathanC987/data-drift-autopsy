# Data Drift Autopsy

Label-free monitoring for deployed classifiers: estimate accuracy without
labels, detect and localise drift, explain it, name its type, score every
prediction's reliability and predict whether a cheap retrain will help. Every
decision is a hypothesis test at one significance level with family-wise error
control; nothing is tuned on the evaluation data.

## What it does

| Stage | Method |
|---|---|
| Accuracy estimate | CBPE: histogram-binning map from confidence to accuracy, fitted on held-out labelled data; baselines average confidence, DoC, ATC |
| Detection | KS (numeric features, corrected), PSI, MMD / PCA reconstruction / Frechet distance permutation tests, confidence-shift chi-square test; baselines BBSD and a classifier two-sample test |
| Localisation | KS or chi-square homogeneity per feature (Holm), effect sizes, unseen-category share, predicted-class slices |
| Root cause | SHAP attribution change with bootstrap intervals (tables); CLIP concept probe and class activation maps (images) |
| Drift type | label-free: input, prediction and MMD tests, black-box shift estimation, label-shift consistency; audit: accuracy-drop and error-structure tests; compared with rule baselines, a learned meta-classifier and ablations |
| Reliability | confidence risk, out-of-distribution score, instability; logistic fusion learned on held-out data |
| Remediation | strategies graded by paired bootstrap; label-free triage by importance-weighted validation |

## GPU

The permutation tests (MMD, Frechet distance, PCA reconstruction) run on a
CUDA GPU through PyTorch when one is available, in float64 with the same
seeded permutations as the CPU code, so the decisions are identical
(`tests/unit/test_baselines_models_and_design_b.py::test_gpu_permutation_tests_match_cpu`).
Set `DRIFT_AUTOPSY_DEVICE=cpu` to force the CPU; `DRIFT_AUTOPSY_GPU_SLOTS`
(default 4) caps how many parallel benchmark workers use the GPU at once. The
models themselves (scikit-learn) run on the CPU.

## Install

```bash
pip install -e ".[all,dev]"     # Python >= 3.10
```

## Quick start

```python
import numpy as np
from drift_autopsy import Dataset, DriftPipeline
from drift_autopsy.detectors import KSTest

rng = np.random.default_rng(0)
reference = Dataset.from_numpy(rng.normal(size=(1000, 5)))
production = Dataset.from_numpy(rng.normal(loc=0.5, size=(1000, 5)))

result = DriftPipeline(detector=KSTest(), localizer="univariate").run(reference, production)
print(result.detection.drift_detected, result.localization.drifted_features)
```

Declare nominal features with `Dataset(..., categorical_features=[...])` so
they are tested with count-based statistics.

## Reproducing the studies and the paper

Every study takes `--model lr` (logistic regression) or `--model hgb`
(histogram gradient boosting).

```bash
python examples/quickstart/folktables_demo.py --task income --model lr       # ACS Income, CA 2014 -> 2015-2023 and four states
python examples/quickstart/folktables_demo.py --task employment --model lr   # ACS Employment, same windows
python examples/quickstart/gas_drift_study.py --model lr                     # UCI gas sensor array drift
python examples/quickstart/clear10_tabularize_demo.py                        # once: ResNet-18 embeddings for CLEAR-10
python examples/quickstart/clear10_drift_study.py --model lr                 # CLEAR-10
python examples/quickstart/clip_concept_probe_demo.py                        # concept-level root cause
python examples/quickstart/remediation_demo.py --model lr                    # remediation and triage (after the studies)
python examples/quickstart/drift_injection_benchmark.py --dataset acs_income --design A --model lr
                                                                             # repeat for acs_employment, design B, hgb
python examples/quickstart/benchmark_summary.py                              # baselines, ablations, paired tests
python examples/paper/build_artifacts.py                                     # every number, table and figure in the paper
streamlit run examples/dashboard/app.py                                      # dashboard over the outputs
pytest                                                                       # tests
```

Data: ACS Income and ACS Employment via
[Folktables](https://github.com/socialfoundations/folktables) (raw files under
`data/<year>/`, task caches under `data/folktables_us_census/tasks/`; from
2019 the relationship variable RELSHIPP is mapped back to RELP), the
[UCI gas sensor array drift dataset](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset)
(downloaded and checksummed under `data/gas_sensor_drift/`), and CLEAR-10 from
the [CLEAR benchmark](https://clear-benchmark.github.io/) under `data/clear10/`.

## Documentation

- `docs/02-current-implementation/` - what is implemented, why, data contracts, limitations
- `docs/01-end-goal/` - long-term product direction
