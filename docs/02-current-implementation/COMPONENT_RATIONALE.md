# Component Rationale (Current Choices)

This file explains why current implementation choices were made.

The goal is practical understanding for next changes, not long decision logs.

## 1. Registry-Based Component Creation

Current choice:

- Detectors/localizers/RCA can be instantiated from registry names.

Why this was chosen:

- Keeps pipeline assembly flexible.
- Supports extension without changing core pipeline constructor behavior.
- Makes config-driven construction feasible.

## 2. Detection As Mandatory Stage

Current choice:

- Pipeline stops on detection failure.
- Pipeline continues on localization/RCA failure.

Why this was chosen:

- Detection is the minimal required output.
- Localization and RCA are optional enrichments.
- Preserves useful signal when optional stages fail.

## 3. Univariate Localizer As Baseline

Current choice:

- Feature-by-feature statistical localization.

Why this was chosen:

- Interpretable and simple to validate.
- Fast enough for baseline workflows.
- Works well as first localization layer before advanced methods.

## 4. SHAP Analyzer Design

Current choice:

- SHAP runs on all numeric model inputs for shape compatibility.
- Reporting may focus on selected/drifted features.

Why this was chosen:

- Prevents model input dimension mismatch.
- Preserves meaningful feature-focus reporting for operator clarity.

## 5. NumPy Compatibility Shims In RCA Path

Current choice:

- Compatibility shims for NumPy 2.x deprecations used by SHAP internals.

Why this was chosen:

- Keeps current SHAP analysis path operational in modern environments.
- Avoids blocking RCA due to dependency API changes.

## 6. Dashboard Separation: CBPE vs Drift Detectors

Current choice:

- CBPE visualized separately from KS/PSI/MMD detector views.

Why this was chosen:

- CBPE score scale and semantics differ from drift detector scores.
- Mixed plotting caused readability and interpretation issues.

## 7. Standard Threshold Defaults

Current choice:

- Keep method-standard thresholds instead of lowering values for visualization convenience.

Why this was chosen:

- Project is general-purpose, not tuned for one demo dataset.
- Preserves method semantics and avoids misleading behavior.
- Future direction is calibration framework, not ad-hoc threshold tweaking.

## 8. RCA In Demo Enabled For KS Test Path

Current choice:

- Demo primarily enables RCA in KS Test pipeline path.

Why this was chosen:

- Balances runtime and interpretability for demonstration.
- Reduces overhead while validating RCA plumbing and dashboard integration.

## How To Update This File

- Update only the component subsection touched by code changes.
- Add concise reason for the choice.
- Avoid large chronological logs here.
- If rationale becomes obsolete, replace it in place.
