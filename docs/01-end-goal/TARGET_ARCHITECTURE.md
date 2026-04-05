# Target Architecture

## Architecture Vision

Core SDK (Python)
-> Self-hosted deployment (Docker/Kubernetes)
-> Optional SaaS control plane (dashboard/orchestration/policy management)

Near-term approved architecture progression:

1. Tabular baseline system (completed)
2. Image classification support through embedding-first transformation (approved next)

## Logical Components

1. Ingestion and profiling layer
- Batch and stream adapters
- Feature, embedding, and target profiling
- Data quality and schema checks

1a. Representation extraction layer (image pathway)
- Image encoder/extractor interfaces (registry-driven)
- Default ResNet extraction path
- Extensible hooks for future encoders (CLIP, SimCLR, DINOv2, MAE, ViT, ConvNeXt, EfficientNet)
- Embedding cache and extraction metadata for reproducibility

2. Detection engine
- Multi-method detector runtime
- Detector scheduling and execution planner
- Detector scoring normalization and confidence modeling
- Extensible detectors for leakage, misinformation, and hallucination signals
- Embedding-space detector support (for example MMD, KS/PSI on dimensions, PCA/FID-style distances)

3. Localization engine
- Feature-level and slice-level localization
- Interaction-aware localization for multivariate patterns
- Metadata/class/slice localization for image-derived embedding records

4. Root cause analysis engine
- SHAP and model-aware explainers
- Drift-cause correlation analysis
- Impact decomposition and trace output
- Embedding-first RCA path for image workflows (dimension shift attribution + output correlation)

5. Remediation engine
- Policy-triggered retraining workflows
- Data reweighting/domain adaptation actions
- Champion-challenger switching and rollback controls
- Human-in-the-loop approvals where required
- Remediation tracks for leakage/misinformation/hallucination classes (integration design pending)

6. Policy and threshold engine
- Static and adaptive threshold support
- Per-model and per-segment policies
- Escalation policy definitions

7. Results and evidence store
- Structured event logs
- Versioned diagnostics results
- Audit-ready history and lineage

8. User interfaces
- Developer APIs and SDK calls
- Operational dashboards
- Alert and notification connectors

## Runtime Modes

1. Batch mode
- Hourly/daily scheduled analysis
- Historical comparisons and trend analysis

2. Streaming mode
- Near-real-time event analysis
- Target detection lag under 1 to 5 minutes

## Deployment Shapes

1. Local development
- Single-process execution
- Lightweight storage

2. Self-hosted enterprise
- Containerized services
- Kubernetes deployment with horizontal scaling

3. Optional control plane
- Central multi-workspace governance and orchestration
- Tenant isolation and policy distribution

## Data Flow (High Level)

1. Reference baseline and incoming data are ingested.
2. For image pathways, raw samples are transformed into embedding-plus-prediction records.
3. Detection engine evaluates relevant drift/failure modes.
4. Localization and RCA are triggered according to policy.
5. Remediation engine receives event severity and context.
6. Actions are recommended or executed based on policy.
7. Artifacts and decisions are persisted for audit and replay.

## Undecided Integration Areas (Explicit)

The following end-goal capabilities are confirmed, but integration design is not yet fixed:

- Data leakage detection and remediation
- Misinformation detection and remediation
- Hallucination detection and remediation

When design is finalized, this file must be updated with exact component boundaries and data flow contracts for these three areas.

## Scalability Targets

- 100 to 1000 monitored models
- 1M to 10M predictions per day
- Feature-level statistics in milliseconds
- RCA in seconds to minutes

## Evolution Path

- Early stage: single-node runtime
- Growth stage: distributed execution (for example Spark or Dask-backed workers)
