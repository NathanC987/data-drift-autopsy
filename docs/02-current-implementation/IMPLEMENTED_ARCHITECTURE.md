# Implemented Architecture

## Data and model contract

- `Dataset` wraps a feature frame plus optional target, predictions, class
  probabilities, metadata and the list of nominal features
  (`categorical_features`). Detectors and the localizer branch on that list.
- A study fixes three disjoint reference parts (see `experiments/acs.py`,
  `experiments/clear10.py`): `train` fits the model; `calibration` is the
  reference window for every detector and fits the CBPE map and the
  reliability fusion; `test` gives the reference accuracy and a no-drift
  control window.

## Per-window flow (as run by the study scripts)

1. `DriftPipeline` runs each detector against the reference window; the KS
   pipeline also runs the univariate localizer and the SHAP analyzer.
2. The CBPE detector reports the confidence-shift test and the accuracy estimate.
3. `diagnosis.label_free_diagnosis` runs the input tests, the prediction test
   and the MMD test as one weighted Holm family (three equal blocks), estimates the prior shift (BBSE) and
   re-tests the inputs against the BBSE-reweighted reference.
4. `diagnosis.audit_diagnosis` (optional) uses a small labelled sample: an
   accuracy-drop test and per-feature likelihood-ratio tests of whether the
   change in errors depends on the feature beyond confidence.
5. `diagnosis.combine` names the drift type and its stage.
6. `ReliabilityAnalyzer.analyze` scores every prediction; `summarize` reports
   means, label shares and, when labels exist for evaluation, AUROCs.
7. `remediation.remediation_triage` trains the cheap probes and predicts their
   gain by importance-weighted validation; `run_remediation_suite` grades all
   strategies on a labelled holdout.

## Outputs

Each study writes one JSON file in `outputs/` (schemas in
`DASHBOARD_AND_DATA_CONTRACTS.md`). `examples/paper/build_artifacts.py` turns
them into `paper/generated/results.tex` (every number cited by the paper),
`paper/generated/tab_*.tex` and `paper/figures/gen_*.pdf`; the dashboard reads
the same files.

## Failure behaviour

Localisation and RCA failures inside `DriftPipeline` are logged and the
detection result is kept. Remediation strategies that need data the context
lacks raise `NotApplicable` and are listed as skipped, never silently replaced
by the deployed model.
