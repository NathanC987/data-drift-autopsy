# Product Vision

## Product Name

Data Drift Autopsy

## Product Mission

Build a diagnostics engine for ML systems that can detect, explain, and remediate model risk from all relevant data and model failure modes.

## Final Product Shape

The final product should support:

1. Core Python SDK
   - Embeddable in training, batch, and online workflows.

2. Self-hosted enterprise deployment
   - Docker and Kubernetes deployment patterns.
   - Suitable for regulated and private environments.

3. Optional SaaS control plane
   - Central dashboard, orchestration, and policy control.
   - Works as optional layer above self-hosted runtime.

## Primary Users

- ML Engineers
- Data Scientists
- MLOps/Platform Engineers
- Risk/Model Governance Teams
- Business Analysts

## Capability Goals

The end-goal product must cover these failure classes:

- Covariate/Data Drift
- Prior Drift
- Concept Drift
- Label Drift
- Data Quality/Schema Drift
- Data Leakage
- Misinformation Signals
- Hallucination Signals

Note:

- The exact implementation and integration approach for leakage/misinformation/hallucination is intentionally undecided at this stage.
- They are committed end-goal capabilities and must be supported by detection and remediation workflows.

The system should not only detect events. It should support end-to-end diagnostics and remediation decisions.

## Modality Strategy

The product is intended to evolve as a multimodal diagnostics system.

Planned modality progression:

1. Tabular baseline (implemented)
2. Image classification via embedding-first pipeline (approved next)
3. Additional modalities over time (for example text, time-series) through the same modular contracts

For image data, the approved direction is to transform images into embedding-plus-prediction tabular contracts, then reuse the same pipeline stages for proxy estimation, drift detection, localization, RCA, and remediation reporting.

## Core Product Outcomes

1. Fast and trustworthy detection
   - Detect drift/failure modes in batch and streaming contexts.

2. Actionable diagnosis
   - Localize features/slices affected.
   - Explain impact and likely root causes.

3. Remediation readiness
   - Recommend and support remediation policies.
   - Integrate automated and human-in-the-loop paths.
   - Support remediation paths for leakage/misinformation/hallucination cases once integration design is finalized.

4. Enterprise operation
   - Operate with auditability, access controls, compliance support, and explicit reliability targets.

## Product Boundaries

Included:

- Monitoring and diagnostics engine
- Explanation and remediation policy support
- Integration interfaces for model serving and orchestration stacks

Not intended to replace:

- Full model training platforms
- Full data catalog/governance products
- Full feature store platforms

## Design Principles

- Modular first: detectors/localizers/RCA/remediation should be composable.
- Interface stable: public APIs should be consistent as internal implementations evolve.
- Explainability first: outputs should be interpretable by engineers and governance users.
- Deploy-anywhere: same logic should run locally, self-hosted, or under control plane orchestration.
- Policy-driven operations: thresholds, alerts, and remediation behavior should be configurable and auditable.
