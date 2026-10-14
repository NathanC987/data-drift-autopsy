# Dashboard and Data Contracts

## Output files

| File | Writer | Top-level keys |
|---|---|---|
| `outputs/folktables_drift_results.json` | `examples/quickstart/folktables_demo.py` | `config`, `reference`, `reliability_model`, `windows` |
| `outputs/clear10_drift_results.json` | `examples/quickstart/clear10_drift_study.py` | `config`, `class_names`, `reference`, `reliability_model`, `windows` |
| `outputs/clip_concepts/concept_rca_results.json` | `examples/quickstart/clip_concept_probe_demo.py` | `config`, `buckets` |
| `outputs/remediation_results.json` | `examples/quickstart/remediation_demo.py` | `config`, `settings`, `summary` |
| `outputs/drift_injection_benchmark.json` | `examples/quickstart/drift_injection_benchmark.py` | `config`, `runs`, `summary`, `wall_clock_seconds` |
| `outputs/drift_injection_benchmark_runs.json` | same, written before aggregation | list of window records |

### Window record (ACS; CLEAR-10 is analogous)

`analysis_type` (control / temporal / geographic), `n_rows`, `positive_rate`,
`predicted_positive_rate`, `actual_accuracy`, `accuracy_change` (window minus
reference), `accuracy_drop_test` (one-sided two-proportion test),
`cbpe_estimated_accuracy`, `cbpe_estimate_error` (estimate minus truth),
`pipelines` (per detector: `detection`, and for KS `localization` with
per-feature `distribution_shift` and `rca` with per-feature `change`,
`change_ci_low`, `change_ci_high`, `change_significant`), `explanation_shift`,
`diagnosis` (`final` verdict, label-free evidence, BBSE, audit summary),
`reliability` (`summary` over the full window, `records` = 1,000 sampled
rows with signals, risk and correctness), `timings_seconds`.

CLEAR-10 windows hold `detection` directly (no pipelines), `proxy`
(estimated, actual and error for accuracy and macro precision / recall / F1),
`localization` (count and top dimensions), `predicted_class_slices` and
`n_images` instead of `n_rows`.

### Benchmark window record

`state`, `seed`, `drift_type`, `intensity_label`, `intensity_order`, `params`,
`n_production_rows`, `ground_truth` (the injection spec), `true_accuracy`,
`detection`, `label_free`, `audit`, `diagnosis`, `diagnosis_correct`,
`diagnosis_by_audit_size`, `localization`, `estimator`, `reliability`,
`triage`, `triage_truth`, `timings_seconds`.

## Dashboard (`examples/dashboard/`)

- `loaders.py`: pure functions from the JSON files to tables (unit tested in
  `tests/integration/test_dashboard.py`).
- `charts.py`: Plotly figures (fixed categorical palette, hover tooltips).
- `app.py`: three tabs (ACS Income, CLEAR-10, Injection benchmark). The two real
  studies follow the pipeline order: setup, accuracy vs. CBPE estimate,
  detection, localization, root cause, diagnosis, reliability, remediation.

The dashboard only reads outputs; it never runs or overwrites an experiment.
