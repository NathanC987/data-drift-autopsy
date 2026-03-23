# Capability Status (As Implemented)

This file tracks current implementation status by subsystem.

Status labels:

- Implemented
- Partial
- Not Implemented

## 1. Pipeline Orchestration

- Drift pipeline composition: Implemented
- Conditional localization step: Implemented
- Conditional RCA step: Implemented
- Continue-on-localization-failure behavior: Implemented
- Continue-on-RCA-failure behavior: Implemented
- Runtime threshold overrides at run boundary (detector/localizer): Implemented
- Generic slice analysis at run boundary (aggregate + per-slice): Implemented

Source areas:

- `src/drift_autopsy/core/pipeline.py`

## 2. Drift Detection Methods

Implemented detectors currently visible in project behavior:

- KS Test: Implemented
- PSI: Implemented
- MMD: Implemented
- Domain classifier: Implemented (registered)
- CBPE: Implemented

Registry listing confirms these detector names are available.

## 3. Localization

- Univariate localizer: Implemented
- Numeric feature tests (KS): Implemented
- Categorical feature tests (chi-square): Implemented
- Multiple testing correction (bonferroni/holm): Implemented
- Top-k truncation: Implemented
- Slice-localization rollup storage (`slice_drifts`): Implemented

## 4. Root Cause Analysis

- SHAP analyzer: Implemented
- NumPy 2.x compatibility shims for SHAP dependency behavior: Implemented
- Focused feature reporting with full-feature SHAP input path: Implemented
- RCA currently enabled in demo pipeline for KS Test path: Implemented

## 5. Demo Coverage

- Temporal drift demo (CA 2014 baseline -> 2015-2018 tests): Implemented
- Geographic drift demo (cross-state via generic slice engine): Implemented
- Multi-detector run in demo: Implemented
- Results export to JSON: Implemented
- Runtime threshold passing at call time (standard values): Implemented

## 6. Dashboard

- Summary metrics: Implemented
- Detector timelines and comparisons: Implemented
- Performance accuracy and delta chart: Implemented
- Feature-level drift visuals: Implemented
- RCA visuals and tables: Implemented
- Separation of CBPE view from drift detectors: Implemented
- Slice-level drift section (heatmap + details table): Implemented

## 7. Remediation

- Automated remediation actions: Not Implemented
- Human-in-the-loop remediation workflow: Not Implemented
- Retraining orchestration triggers: Not Implemented

## 8. Drift Scope Coverage

Current code path emphasis:

- Covariate/data drift: Implemented baseline
- Proxy performance estimation: Implemented
- Prior drift: Not Implemented as dedicated module
- Concept drift: Not Implemented as dedicated module
- Label drift: Not Implemented as dedicated module
- Data quality/schema drift: Partial (validation exists, dedicated drift module not complete)
- Generic sub-slice drift analysis (metadata-column based): Implemented baseline
- Feature-group drift analysis: Partial (config placeholders/interfaces only)

## 9. Runtime Modes

- Batch workflow: Implemented
- Streaming near-real-time workflow: Not Implemented

## 10. Enterprise Controls

- RBAC/SSO: Not Implemented
- Audit log framework: Partial (logging exists, audit-grade event model not complete)
- Compliance control mapping: Not Implemented
- SLO instrumentation framework: Partial
