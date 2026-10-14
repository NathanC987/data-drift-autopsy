# Data Drift Autopsy

Label-free monitoring for deployed classifiers: estimate accuracy without
labels, detect and localise drift, explain it, name its type, score every
prediction's reliability and predict whether a cheap retrain will help. Every
decision is a hypothesis test at one significance level with family-wise error
control; nothing is tuned on the evaluation data.

## What it does

| Stage | Method |
|---|---|
| Accuracy estimate | CBPE: histogram-binning map from confidence to accuracy, fitted on held-out labelled data |
| Detection | KS (numeric features, corrected), PSI, MMD / PCA reconstruction / Frechet distance permutation tests, confidence-shift chi-square test |
| Localisation | KS or chi-square homogeneity per feature (Holm), effect sizes, unseen-category share, predicted-class slices |
| Root cause | SHAP attribution change with bootstrap intervals (tables); CLIP concept probe and class activation maps (images) |
| Drift type | label-free: input, prediction and MMD tests, black-box shift estimation, label-shift consistency; audit: accuracy-drop and error-structure tests |
| Reliability | confidence risk, out-of-distribution score, instability; logistic fusion learned on held-out data |
| Remediation | strategies graded by paired bootstrap; label-free triage by importance-weighted validation |

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

```bash
python examples/quickstart/folktables_demo.py            # ACS Income, temporal + geographic
python examples/quickstart/clear10_tabularize_demo.py    # once: ResNet-18 embeddings for CLEAR-10
python examples/quickstart/clear10_drift_study.py        # CLEAR-10
python examples/quickstart/clip_concept_probe_demo.py    # concept-level root cause
python examples/quickstart/remediation_demo.py           # remediation and triage
python examples/quickstart/drift_injection_benchmark.py  # 650-window injection benchmark
python examples/paper/build_artifacts.py                 # every number, table and figure in the paper
streamlit run examples/dashboard/app.py                  # dashboard over the outputs
pytest                                                   # tests
```

Data: ACS Income via [Folktables](https://github.com/socialfoundations/folktables)
(cached under `data/folktables_us_census/`), CLEAR-10 from the
[CLEAR benchmark](https://clear-benchmark.github.io/) under `data/clear10/`.

## Documentation

- `docs/02-current-implementation/` - what is implemented, why, data contracts, limitations
- `docs/01-end-goal/` - long-term product direction

## License

MIT
