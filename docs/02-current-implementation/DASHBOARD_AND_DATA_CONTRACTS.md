# Dashboard And Data Contracts (Current)

This file documents dashboard behavior and expected result structure.

## 1. Input Data Source

Primary source:

- `outputs/folktables_drift_results.json`

Supported structures by loader:

1. Wrapped format
- Top-level `yearly_results` object containing year keys.

2. Direct format
- Top-level year keys directly (`2015`, `2016`, ...).

Loader handles both.

## 2. Summary Metrics Semantics

Current summary metrics include:

- Total years analyzed
- Drift events detected
- Average accuracy
- Unique drifted features

Important behavior:

- Year count is computed from digit year keys.
- This was corrected to avoid returning zero when `yearly_results` wrapper is absent.

## 3. Performance Chart Semantics

Performance table fields:

- `accuracy`
- `accuracy_delta` (stored as accuracy drop/delta from baseline in current demo outputs)

Interpretation rule:

- Accuracy shows absolute performance by year.
- Accuracy delta shows movement from training/reference context.

## 4. Slice Analysis Contract (New)

Slice analysis is currently read from pipeline metadata:

- `pipelines.<name>.metadata.slice_analysis.enabled`
- `pipelines.<name>.metadata.slice_analysis.column`
- `pipelines.<name>.metadata.slice_analysis.slice_count`
- `pipelines.<name>.metadata.slice_analysis.slices`

Optional analysis-level label map (used when available):

- `slice_value_labels` (dict mapping raw slice values to human-readable labels, e.g. `"1": "CA"`)

Each slice payload includes:

- `reference_slice_value`
- `test_slice_value`
- `reference_samples`
- `test_samples`
- `result` (nested pipeline result dict for that slice)

Dashboard loader emits both raw and labeled fields:

- `slice_key` (raw, e.g. `1->2`)
- `slice_key_label` (labeled, e.g. `CA->TX`, falls back to raw values)
- `reference_slice` / `test_slice` (raw)
- `reference_slice_label` / `test_slice_label` (labeled fallback fields)

## 5. Detector Visual Semantics

Current dashboard separates:

- Performance estimator section (CBPE)
- Drift detector section (KS Test, PSI, MMD)

Reason:

- Different metric scales and interpretation domains.

## 6. Localization Data Contract

Expected localization structure per pipeline result:

- `localization.feature_drifts[]`
  - `feature_name`
  - `score`
  - `drift_detected`
  - `severity`

## 7. RCA Data Contract

Expected RCA structure per pipeline result:

- `rca.analyzer_name`
- `rca.explanations`
- `rca.recommendations[]`
- `rca.feature_importances` (flat change map)
- `rca.distribution_changes` (nested ref/test/change map)

Dashboard feature-importance view currently relies on `distribution_changes` for richer fields.

## 8. Known Chart Behavior Notes

- Drift markers only appear where `drift_detected = true`.
- If a detector reports all false values, no drift marker appears for that detector line.
- This is expected behavior, not a plotting bug.

Slice chart notes:

- Slice heatmap uses per-slice detector scores from metadata and prefers labeled slice keys when available.
- Slice table prefers labeled slice keys with fallback to raw keys.
- If no pipeline output contains `slice_analysis.enabled=true`, slice section displays informational empty state.

## 9. Update Rules

When dashboard-related code changes:

- Update only impacted section(s) in this file.
- If JSON contract changes, update contracts first, then chart behavior notes.
