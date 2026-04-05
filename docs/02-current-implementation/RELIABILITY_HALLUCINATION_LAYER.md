# Reliability & Hallucination Detection Layer

## Purpose

This module adds a model-agnostic reliability layer to Drift Autopsy so the system can estimate whether predictions are **likely unreliable (high risk)** without requiring ground-truth labels.

The layer works for:
- Tabular
- Image
- Text
- Audio / unstructured

The output is risk-oriented, not correctness-oriented.

## Implemented Components

Implementation lives under:
- `src/drift_autopsy/reliability/`

Files:
- `confidence.py`
- `ood.py`
- `stability.py`
- `calibration.py`
- `explanation.py`
- `risk_engine.py`
- `analyzer.py`
- `__init__.py`

Top-level exports are also added in:
- `src/drift_autopsy/__init__.py`

## Architecture

### 1) Confidence Extraction (`confidence.py`)

`ConfidenceExtractor` provides a unified confidence interface:

- Classification: uses `predict_proba` and max probability
- Regression: uses prediction uncertainty proxy:
  - external std estimator if provided
  - ensemble variance (`estimators_`) if available
  - fallback dispersion heuristic
- Transformer/LLM style: uses logits (via callback or model method) and softmax max probability

Output:
- `confidence_score` in `[0, 1]`

### 2) OOD Detection (`ood.py`)

`OODDetector` supports both tabular and embedding-based workflows.

- Tabular default: Isolation Forest
- Image/Text/Audio/Unstructured default: embedding distance from reference distribution center
- Unified API:
  - `fit(reference_data)`
  - `compute_ood_score(input)`
  - `compute_ood_score_batch(inputs)`

Output:
- `ood` score in `[0, 1]` (higher means more OOD)

### 3) Stability Check (`stability.py`)

`StabilityChecker` evaluates sensitivity by perturbing inputs and recomputing predictions:

- Tabular: Gaussian noise per feature scale
- Image: Gaussian pixel noise
- Text: token dropout perturbation
- Generic fallback for unstructured arrays

Output:
- `stability` score in `[0, 1]` (higher means less stable / riskier)

### 4) Indirect Calibration (`calibration.py`)

`CalibrationChecker` avoids label-dependent ECE and instead uses:

- Confidence
- External CBPE score
- Confidence distribution shift

It detects overconfidence patterns (high confidence + weak CBPE alignment).

Output:
- `calibration`: `good | suspicious`
- `calibration_risk` in `[0, 1]`

### 5) Explanation Consistency (`explanation.py`)

`ExplanationConsistencyChecker` reuses existing explainability stack:

- Tabular: reuses `SHAPAnalyzer` from `drift_autopsy.rca`
  - compares explanation shift magnitudes / vectors
- Image: supports Grad-CAM compatibility through adapter callback
  - compares baseline/current heatmaps using cosine distance

Output:
- `explanation` score in `[0, 1]`

### 6) Risk Engine (`risk_engine.py`)

`RiskScoringEngine` combines all reliability signals.

Implemented methods:

1. **Weighted score**

```
risk_score =
  w_conf * (1 - confidence)
+ w_ood * ood
+ w_stab * stability
+ w_cal * calibration_risk
+ w_exp * explanation
```

2. **Rule fallback / escalation**

Examples:
- `ood > 0.7 and confidence > 0.9 => HIGH`
- additional high-risk escalation rules for unstable + overconfident behavior

Output:
- `risk_score` in `[0, 1]`
- `risk_label`: `LOW | MEDIUM | HIGH`

### 7) Orchestration (`analyzer.py`)

`ReliabilityAnalyzer` is the single model-agnostic interface:

- Constructor receives model, data type, reference data, and optional adapters
- Runs all reliability modules end-to-end
- Supports:
  - single sample: `analyze(...)`
  - batch input: `analyze_batch(...)`
- Supports dataset directory bootstrapping:
  - `from_dataset_dir(model, "/data/<dataset_name>/", ...)`
  - auto-detects data format and inferred data type
- Supports output persistence:
  - `save_results(..., output_path)` to JSON

## Output Contract

Each record follows this schema:

```json
{
  "prediction_id": "...",
  "confidence": 0.93,
  "ood": 0.78,
  "stability": 0.61,
  "calibration": "suspicious",
  "calibration_risk": 0.74,
  "explanation": 0.55,
  "cbpe_score": 0.80,
  "risk_score": 0.82,
  "risk_label": "HIGH",
  "details": { ... }
}
```

## Dashboard Integration

Dashboard files updated:
- `examples/dashboard/data_loader.py`
- `examples/dashboard/visualizations.py`
- `examples/dashboard/app.py`

### Added Dashboard Section

New section:
- **Prediction Reliability (Hallucination Detection)**

Displayed items:
- Confidence
- OOD Score
- Stability Score
- Calibration Status (including suspicious count)
- Explanation Consistency
- Final Risk Label distribution
- Reliability detail table

### Data Loader Support

`DriftResultsLoader` now includes:
- `get_reliability_results(scope="all"|"folktables"|"clear10")`
- `get_reliability_summary(scope=...)`

It supports multiple result schemas:
- top-level reliability list
- per-analysis reliability blocks
- per-pipeline reliability blocks
- CLEAR-10 per-bucket reliability blocks

## Usage Example

```python
from drift_autopsy.reliability import ReliabilityAnalyzer

analyzer = ReliabilityAnalyzer(
    model=model,
    data_type="tabular",
    reference_data=reference_df,
    task_type="classification",
    cbpe_reference_score=0.81,
)

result = analyzer.analyze(input_sample=test_df.iloc[0].values)
batch_results = analyzer.analyze_batch(test_df.values)

analyzer.save_results(batch_results, "outputs/reliability_results.json")
```

## Design Notes

- Model-agnostic by relying on capability checks (`predict_proba`, `predict`, logits callbacks)
- Label-free by design for production use
- Reuses existing CBPE/SHAP/Grad-CAM ecosystem
- Modular so each reliability signal can be replaced or extended independently
