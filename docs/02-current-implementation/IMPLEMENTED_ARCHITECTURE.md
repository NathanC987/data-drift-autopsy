# Implemented Architecture (Current Code)

This file describes the architecture that exists in code now.

## 1. Main Execution Path

Primary flow:

1. Construct `DriftPipeline` with detector and optional localizer/RCA.
2. Validate reference and test dataset compatibility.
3. Optionally apply runtime threshold overrides for detector/localizer.
4. Run aggregate detector `fit_detect`.
5. If enabled, run aggregate localizer with detection signal.
6. If enabled, run aggregate RCA with optional model and localization output.
7. If slice analysis is enabled, run per-slice analysis and attach slice results in metadata.
8. Return `PipelineResult` with detection/localization/RCA and metadata.

Primary source:

- `src/drift_autopsy/core/pipeline.py`

## 2. Composition Model

The current architecture is interface-based and registry-enabled:

- Detector can be passed as instance or registry string.
- Localizer can be passed as instance or registry string.
- RCA analyzer can be passed as instance or registry string.
- Registry string construction supports parameter dictionaries (`detector_params`, `localizer_params`, `rca_params`).

This allows modular swapping with low coupling.

## 3. Error Handling Behavior

Current behavior in pipeline:

- Detection failure: raises and stops pipeline.
- Localization failure: logs error, continues with localization result as null.
- RCA failure: logs error, continues with RCA result as null.

This design keeps detection as mandatory core step and optional stages resilient.

## 4. Data Contracts

Core data structures:

- `Dataset` as standard data carrier
- `DetectionResult`
- `LocalizationResult`
- `RCAResult`
- `PipelineResult`

JSON export from demo uses `result.to_dict()` and year-level aggregation.

Additional metadata contracts in current pipeline runs include:

- Effective threshold values used at runtime
- Threshold source (`runtime_override` or component default)
- Optional `slice_analysis` block with per-slice results

## 5. Localizer Architecture

Current implemented localizer:

- Univariate statistical testing per feature
- Numeric and categorical branches
- Multiple testing correction support
- Sorted feature drift outputs by p-value
- Aggregate localization result can include per-slice localization outputs via `slice_drifts`

## 6. RCA Architecture

Current implemented RCA path:

- SHAP-based analyzer
- Uses all numeric features for model input compatibility
- Can focus reporting/recommendations on selected or drifted features
- Produces feature importance shifts and recommendations

## 7. Dashboard Data Path

Current dashboard path:

1. Load JSON results via `DriftResultsLoader`.
2. Normalize both supported JSON styles (yearly_results wrapper or direct year keys).
3. Build tabular views for detectors, feature drift, performance, RCA, and slice-level outputs.
4. Feed visualizations in Streamlit app.

## 8. Slice Analysis Architecture (Current)

Current slice path is generic and metadata-column based:

1. User passes `slice_config` to `pipeline.run(...)`.
2. Pipeline validates metadata and slice column availability.
3. Pipeline runs aggregate analysis first.
4. Pipeline runs per-slice analysis in one of two modes:
	- Same-slice mode: ref slice value compared to same test slice value.
	- Cross-slice mode: fixed reference slice value compared against each test slice.
5. Slice results are stored in pipeline metadata for dashboard and downstream use.

## 9. Current Deployment Shape

- Local script execution
- Streamlit dashboard process
- File-based outputs under `outputs/`

No service-oriented runtime orchestration layer is currently implemented.
