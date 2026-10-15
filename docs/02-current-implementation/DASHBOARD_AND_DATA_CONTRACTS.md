# Dashboard and Data Contracts

## Output files

| File | Writer | Top-level keys |
|---|---|---|
Two model families are run for every study: `lr` (logistic regression, the
primary model) and `hgb` (histogram gradient boosting). The model is recorded
in `config.model_kind`.

| File | Writer | Top-level keys |
|---|---|---|
| `outputs/folktables_drift_results.json` (Income, lr), `outputs/folktables_{task}_drift_results_{model}.json` | `examples/quickstart/folktables_demo.py --task {income,employment} --model {lr,hgb}` | `config`, `reference`, `reliability_model`, `windows` |
| `outputs/clear10_drift_results.json` (lr), `outputs/clear10_drift_results_hgb.json` | `examples/quickstart/clear10_drift_study.py --model` | `config`, `class_names`, `reference`, `reliability_model`, `windows`, `embedding_projection` |
| `outputs/gas_drift_results_{model}.json` | `examples/quickstart/gas_drift_study.py --model` | `config`, `reference`, `reliability_model`, `windows` |
| `outputs/clip_concepts/concept_rca_results.json` | `examples/quickstart/clip_concept_probe_demo.py` | `config`, `buckets` |
| `outputs/remediation_results.json` (lr), `outputs/remediation_results_hgb.json` | `examples/quickstart/remediation_demo.py --model` | `config`, `settings`, `summary` |
| `outputs/benchmark/{dataset}_{design}_{model}.json` | `examples/quickstart/drift_injection_benchmark.py --dataset --design --model` | `config`, `runs`, `summary`, `wall_clock_seconds` |
| `outputs/benchmark/{dataset}_{design}_{model}_runs.json` | same, written before aggregation | list of window records |
| `outputs/benchmark/summary.json` | `examples/quickstart/benchmark_summary.py` | `alpha`, `methods` (scores and paired tests per configuration), `mechanisms` (design B) |
| `outputs/figures/cam/cam_manifest.json` (+ `cam_*_image.png`, `cam_*_overlay.png`) | `examples/paper/build_artifacts.py` | `bucket`, `k`, `colormap`, `alpha`, `selection_rule`, `scaling`, `images` |

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
rows with signals, risk and correctness), `comparison` (see below),
`timings_seconds`. Windows run from the CA 2014 control through CA 2015 to
2019 and 2021 to 2023 (no standard 2020 release) and TX, NY, FL, WA 2014;
each has a `year` and `state`. For the boosted model `pipelines["KS Test"].rca`
is absent (SHAP runs for the logistic model only).

`comparison` (every study): `tests` (the compact test record of
`drift_autopsy.diagnosis_baselines.window_tests`), `verdicts` (ours, the rule
baselines and the ablations), `estimates` and `estimate_errors` (CBPE, average
confidence, DoC, ATC).

Gas sensor windows hold `detection` directly, `batch`, `months`, `gas_counts`,
`localization` (`n_drifted_features`, `drifted_features`, `by_sensor`) and a
`diagnosis` with `n_input_shift_features` and `n_residual_shift_features`.

CLEAR-10 windows hold `detection` directly (no pipelines), `proxy`
(estimated, actual and error for accuracy and macro precision / recall / F1),
`localization` (count and top dimensions), `predicted_class_slices` and
`n_images` instead of `n_rows`.

`embedding_projection` (CLEAR-10, display only): PCA(3) fitted on the
standardised bucket 1 training split (`fitted_on`, `explained_variance_ratio`,
`n_per_window`) and, per window (`"1"` = the bucket 1 calibration split,
`"2"` ... `"10"`), `n_images`, `centroid` (mean of all images) and a seeded
sample of `points` with their `class_name`. No test uses it.

### Benchmark window record

`dataset`, `model`, `design` (A or B), `mechanism`, `state`, `seed`,
`drift_type`, `intensity_label`, `intensity_order`, `params`,
`n_production_rows`, `ground_truth` (the injection spec), `true_accuracy`,
`detection` (KS, PSI, MMD, CBPE, BBSD, domain classifier), `label_free`,
`audit`, `diagnosis`, `diagnosis_correct`, `diagnosis_by_audit_size`,
`localization` (design A), `estimator` (CBPE plus `estimates` / `errors` of
every estimator), `tests`, `verdicts`, `reliability`, `triage`,
`triage_truth`, `timings_seconds`.

## Dashboard (`examples/dashboard/`)

- `loaders.py`: pure functions from the JSON files to one tidy frame per chart.
- `charts.py`: Plotly figures. Reference palette in fixed order, one
  sequential blue ramp for magnitudes, a blue / red diverging scale with a gray
  midpoint for gains and losses, filled vs hollow markers for yes / no, hover
  tooltips everywhere, Streamlit theme (light and dark).
- `narrative.py`: the "What the data shows" sentences, computed from the
  frames (no fixed conclusions).
- `app.py`: four tabs. Each real study tab has a model selector (and the ACS
  tab a task selector: Income or Employment); selectors never hide windows.
  - ACS, CLEAR-10 and the gas sensor array follow the architecture: 1 setup and results
    (split, window sizes, measured accuracy with Wilson intervals);
    2 proxy estimate (CBPE vs measured, estimate error, confidence histogram),
    detection (one step-line panel per detector in its own statistic, alarm
    matrix) and localization (ACS: KS D and Cramér's V heatmaps, unseen
    categories; CLEAR-10: drifted dimensions, per-class heatmap, 3D PCA view;
    gas: shifted features per sensor, gas mix per window);
    3 root cause (ACS: SHAP change heatmap and dumbbell; CLEAR-10: class
    activation maps, concept probe and its tag validation, exemplar check) and
    drift type (decision path per window; ACS adds the positive rates);
    4 reliability (AUROC per signal, risk labels, risk histogram, fusion
    weights); 5 remediation triage (strategy heatmap, triage scorecard,
    predicted vs actual probe gain).
  - The benchmark (dataset, design and model selectors) follows the
    procedure: design, ground truth, detection, proxy estimate (with the
    estimator baselines), localization (design A), drift type (design B adds
    the per-mechanism view), baselines and ablations (accuracy with intervals
    and the paired tests), reliability, triage.
  - Temporal lines are not joined across the missing ACS year 2020.
  - A focus window (default: the largest significant accuracy drop) drives the
    detail charts; overview charts always show every window.
- Tests: `tests/integration/test_dashboard.py` (loaders, charts, sentences and
  an AppTest render of all four tabs).

The dashboard only reads outputs; it never runs or overwrites an experiment.
