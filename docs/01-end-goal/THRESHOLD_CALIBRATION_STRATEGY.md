# Threshold Calibration Strategy (End Goal)

## Why This Is Required

Static thresholds are simple but not robust across domains, data volumes, and model types.

End-goal system must support calibrated, context-aware thresholds so detection is both sensitive and operationally useful.

At the same time, users must be able to explicitly set thresholds themselves at call time.

## Calibration Objectives

1. Keep false positives under control.
2. Preserve high recall for meaningful drift events.
3. Adapt by detector, model, feature group, and segment.
4. Keep calibration decisions auditable and reproducible.

## Scope

Calibration should cover at least:

- KS-based statistical detectors
- PSI-based detectors
- MMD-based detectors
- Proxy performance estimators
- Future concept/prior/label drift methods

It also applies to RCA algorithms where threshold-like parameters exist (for example significance cutoffs, recommendation confidence cutoffs, or alert gating criteria).

## User-Controlled Thresholds (Required)

The product must support direct user threshold parameters when calling detector/RCA functions.

Rules:

1. User-provided threshold is always accepted when valid.
2. Calibration can recommend values, but must not remove user override capability.
3. Policy defaults are fallback values, not hard lock-ins, unless explicitly configured by governance policy.
4. Effective threshold used for a run must be recorded in result metadata/audit logs.

Precedence model (end-goal):

1. Explicit runtime parameter (highest precedence)
2. Model/segment policy threshold
3. Global default threshold
4. Algorithm internal default (last fallback)

## Planned Calibration Modes

1. Manual fixed thresholds
- Default bootstrap mode
- Explicit policy-defined values

2. Historical percentile calibration
- Build score distribution from known stable windows
- Set thresholds by percentile targets

3. Labeled optimization calibration
- Use known drift/no-drift windows
- Optimize threshold by objective (F1, precision-recall, cost-weighted utility)

4. Adaptive online calibration
- Update threshold recommendations with rolling feedback
- Guardrails to prevent unstable oscillation

## Policy Model

Threshold policy should support:

- Global default policy
- Per-detector policy
- Per-model policy
- Per-segment policy
- Escalation rules by severity and confidence

Policy model must include a configuration switch for allowing or restricting runtime overrides in governed environments.

## Safety Guardrails

- Minimum/maximum threshold bounds
- Drift volume anomaly guardrails
- Human approval mode for major policy changes
- Rollback to previous threshold policy version

## Evaluation Framework

For each calibration method, evaluate:

- Precision, recall, FPR, FNR
- Alert volume stability over time
- Downstream remediation success impact
- Sensitivity by segment

## Operational Requirements

- Versioned threshold policies
- Full audit trail for threshold changes
- Replay mode to compare old vs new thresholds on historical windows
- Explainable rationale attached to each change recommendation
- Trace of effective threshold source (runtime override vs policy vs default)

## Roadmap Placement

- v0.x: design contracts + offline experiments
- v1.x: policy engine + historical/labeled calibration
- v2.0: adaptive online calibration with guardrails and orchestration
