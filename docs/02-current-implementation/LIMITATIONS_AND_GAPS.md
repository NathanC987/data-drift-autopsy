# Limitations And Gaps (Current)

This file lists current known limitations and missing capabilities.

## 1. Drift Coverage Gaps

- No dedicated concept drift module yet.
- No dedicated prior drift module yet.
- No dedicated label drift module yet.
- Data quality/schema drift is not yet a full first-class module.

## 2. Remediation Gaps

- No automated retraining trigger pipeline.
- No remediation policy execution engine.
- No champion-challenger switch orchestration.
- No rollback execution workflow.

## 3. Thresholding Gaps

- No adaptive threshold calibration engine.
- Thresholds are static and method-configured.
- No per-model/per-segment dynamic policy layer.

## 4. Runtime Gaps

- No near-real-time streaming runtime path.
- Primarily batch/script-driven execution.
- No production scheduler/orchestrator integration layer yet.

## 5. Enterprise Control Gaps

- No RBAC/SSO implementation.
- No audit-grade decision/event model.
- No compliance control mapping artifacts.
- No explicit SLA/SLO enforcement layer.

## 6. Operational Gaps

- No mature alert routing abstraction.
- No standardized runbook integration.
- No built-in replay/backfill tooling package.

## 7. Validation/Test Gaps

- Limited explicit coverage documentation by subsystem.
- No formal benchmark suite for detector tradeoff quality yet.
- No calibration quality benchmark for thresholds yet.

## 8. Documentation Gap Policy

When a gap is closed in code, update this file immediately by:

1. Marking the gap resolved or partial.
2. Adding brief note on what changed.
3. Linking updated section in relevant current-implementation file.
