# Codebase Headstart (Current)

This file is for new core developers who need a complete fast orientation of the current project.

## 1. What This Project Currently Is

Data Drift Autopsy is currently a Python-first modular framework for:

1. Drift detection
2. Drift localization
3. Root cause analysis (RCA)
4. Result visualization in Streamlit dashboard
5. Generic slice-level drift analysis (including geographic cross-slice comparisons)

Current operational style is batch/script driven, with JSON output and dashboard inspection.

## 2. Top-Level Project Layout

- `src/drift_autopsy/`: core package
- `examples/quickstart/`: runnable demos
- `examples/dashboard/`: streamlit app and visualization logic
- `configs/examples/`: yaml config examples
- `tests/`: tests
- `outputs/`: generated run artifacts (for example demo JSON output)
- `docs/`: developer blueprint and implementation documentation

## 3. Core Package Layout (`src/drift_autopsy/`)

- `core/`: primary interfaces, pipeline orchestration, result models
- `detectors/`: detector implementations (statistical/distribution/model/proxy)
- `localizers/`: localization methods (currently univariate)
- `rca/`: root cause analyzers (currently SHAP analyzer)
- `registry/`: registry/factory classes for detectors/localizers/rca
- `data/`: dataset loaders/validators
- `config/`: config schema and loading
- `utils/`: logging and helper utilities

## 4. Start Reading Order (Recommended)

1. `src/drift_autopsy/core/pipeline.py`
   - Understand end-to-end execution flow first.

2. `src/drift_autopsy/core/result.py`
   - Understand output contract and serialized fields.

3. `src/drift_autopsy/registry/`
   - Understand how components are discovered and instantiated.

4. `src/drift_autopsy/detectors/`
   - Understand algorithm behavior and score semantics.

5. `src/drift_autopsy/localizers/univariate.py`
   - Understand feature-level localization path.

6. `src/drift_autopsy/rca/shap_analyzer.py`
   - Understand RCA data path and recommendation generation.

7. `examples/quickstart/folktables_demo.py`
   - See practical end-to-end usage and output generation.

8. `examples/dashboard/`
   - Understand result ingestion and UI presentation.

## 5. End-to-End Runtime Path (Current)

1. Demo script builds datasets and model.
2. Drift pipelines are configured with detector/localizer/optional RCA.
3. Pipeline `run(...)` can receive runtime threshold overrides and optional slice config.
4. Pipeline runs aggregate detection -> localization -> RCA.
5. If slice config is enabled, pipeline computes per-slice outputs in the same run.
6. Results are serialized to JSON.
7. Dashboard loader normalizes JSON and builds analysis tables.
8. Visualizations render trend, feature, RCA, and slice-level insights.

Current demo coverage includes both:

- Temporal drift analysis (across years)
- Geographic drift analysis (cross-state, via generic slice engine)

## 6. Available Detector Set (Current)

Registry currently exposes at least:

- `ks_test`
- `psi`
- `mmd`
- `domain_classifier`
- `cbpe`

## 7. Thresholds In Current Code

Current API pattern supports user-provided thresholds through:

1. Algorithm constructors
2. Pipeline run-time overrides

Examples:

- `KSTest(threshold=...)`
- `PSI(threshold=...)`
- `MMD(threshold=...)`
- `CBPE(threshold=...)`

- `pipeline.run(..., detection_threshold=..., localization_threshold=...)`

This is already supported in current implementation and used by demos.

## 8. Current JSON Contract Highlights

Per year, demo output currently stores:

- performance metrics (`actual_accuracy`, `accuracy_drop`)
- `pipelines` map
- each pipeline has `detection`, `localization`, optional `rca`, and execution metadata

Current execution metadata can include:

- effective thresholds used for detector/localizer
- threshold source (runtime override vs default)
- optional `slice_analysis` block with per-slice outputs

Dashboard loader supports two top-level formats:

- wrapped (`yearly_results`)
- direct year keys

## 9. Current Known Boundaries

- Streaming runtime is not implemented.
- Full remediation engine is not implemented.
- Drift scope is strongest in covariate/proxy paths currently.
- Enterprise access/compliance controls are not implemented yet.

## 10. Where To Update Docs After Code Changes

Follow:

- `UPDATE_PROTOCOL.md`

This keeps documentation updates small, predictable, and non-messy.
