# Roadmap

This roadmap captures approved implementation phases from the current system baseline to the next major capability set.

## Phase 1 (Completed): Tabular Baseline System

Status: Completed

### Goal

Deliver an end-to-end tabular drift autopsy baseline with proxy performance estimation, drift detection, localization, RCA, and dashboarding.

### Delivered Scope

1. Core pipeline and contracts
- Modular `DriftPipeline` orchestration with optional localization and RCA stages.
- Runtime threshold overrides at run boundary for detector/localizer.
- Metadata-rich result contracts and JSON output.

2. Tabular detectors and proxy estimation
- KS Test, PSI, MMD, CBPE implemented and runnable.
- Registry-driven detector construction.

3. Localization and RCA
- Univariate localization with feature-level drift reporting.
- SHAP-based RCA for tabular models.

4. Temporal and geographic analyses
- Temporal drift workflow across ACS years.
- Geographic cross-slice analysis via generic metadata slicing engine.
- Human-readable geographic slice labels in output and dashboard views.

5. Dashboard and data contracts
- Streamlit dashboard for timelines, feature drift, RCA, and slice-level analysis.
- Loader/visualization contracts aligned with produced JSON structure.

### Boundaries Remaining After Phase 1

- No native image pipeline yet.
- No full remediation automation yet (reporting/recommendation only).
- No streaming runtime path yet.

## Phase 2 (Approved Next): Image Classification via Embedding-First Pipeline

Status: Planned

### Goal

Add image-data support in the cleanest modular way by converting image samples into a standardized embedding-plus-prediction tabular contract, then reusing the same core system stages:

1. Proxy performance estimation
2. Drift detection
3. Localization
4. RCA
5. Remediation reporting

Operational input model:
1. User provides:
- The image-classification model to monitor.
- The reference image dataset used to train/calibrate the model.
- The production image data stream.
2. System computes embeddings with an internal feature extractor (default: ResNet), independent from the user's classifier.
3. System builds reference and analysis tabularized datasets as the primary runtime data contract for downstream monitoring stages.

### Design Decisions (Approved)

1. Strategy
- Embedding-first MVP before full visual-RCA automation.

2. Default extractor
- ResNet as first production extractor.

3. Model role separation
- User model is the task classifier and is the source of `y_pred` and `pred_proba_*` columns.
- System ResNet is the embedding extractor and is the source of `feature_1 ... feature_n` columns.

4. Mandatory initial methods
- Multiclass proxy performance estimation (M-CBPE style).
- MMD on embeddings.
- KS/PSI on embedding dimensions.
- PCA reconstruction error.
- FID-style distance metric.

5. RCA depth in Phase 2
- Embedding-first RCA only (shift attribution and output correlation).
- Full visual RCA (for example Grad-CAM/saliency) deferred to later phase.

### Target Flow (Phase 2)

1. Receive image inputs and run two parallel inference paths:
- User classifier path for `y_pred` and `pred_proba_*`.
- System extractor path for embeddings (`feature_1 ... feature_n`).
2. Emit tabularized records for reference and analysis datasets using one standardized schema.
3. Persist tabularized datasets in efficient analytic formats (`.csv`, `.parquet`, or equivalent fast read/write/query storage).
4. Treat analysis data as append-only/continuous, with new records continuously added.
5. Build analysis chunks/batches for per-chunk monitoring with configurable chunking strategies:
- Sliding-window strategies.
- Quantity-based chunking (chunk size = N records; `sample_id` is primary for deterministic grouping/order).
- Temporal chunking (chunk size = duration; `timestamp` is primary for period assignment).
6. Run proxy estimation, drift detection, localization, RCA per chunk using previous chunk as reference by default.
7. For demos that define a fixed reference period (CLEAR-10), use the approved fixed-reference override.
8. Generate remediation recommendations and operational report artifacts.

### Phase 2 Demo Track: CLEAR-10 (Approved)

Purpose:
- Demonstrate end-to-end image pipeline behavior using a real temporal benchmark, analogous to Folktables temporal demo for tabular data.

Dataset assumptions:
- CLEAR-10 labeled images and metadata are already available locally.
- 11 classes including BACKGROUND.
- 10 temporal buckets (1..10), each with class-structured image folders.

Demo protocol:
1. Use bucket 1 as reference period.
2. Split bucket-1 labeled data into train/test (70/30).
3. Demo simplification: use the same ResNet family for both roles:
	- classifier path (as the monitored model)
	- feature extraction path (embedding generator)
	This same-model setup is demo-only and does not change production architecture.
4. Train/evaluate the demo classifier on bucket-1 split and record baseline metrics first.
5. Persist model artifact and extraction settings for reproducibility.
6. Build reference tabularized dataset from bucket 1:
	- embeddings (`feature_1 ... feature_n`) from system extractor
	- `y_pred` and `pred_proba_*` from classifier path
	- `y_true`, `timestamp`, `sample_id`, optional metadata
7. For each subsequent bucket (2..10), build analysis tabularized dataset with same schema.
8. Run pipeline per bucket against fixed bucket-1 reference:
	- proxy performance estimation
	- drift detection
	- localization
	- embedding RCA
9. CLEAR-10 chunking rule for demo: each bucket is one fixed analysis chunk (bucket size = 3300 records: 11 classes x 300 images).
10. Because `y_true` is available in CLEAR-10, report both:
	- proxy-estimated performance
	- true observed performance
	and compute proxy quality gap metrics.

Dashboard flow contract for CLEAR-10 page:
1. Baseline model performance (bucket-1 test split) appears first.
2. Proxy performance estimation section appears second.
3. Drift detection section appears third.
4. Localization section appears fourth.
5. RCA section appears fifth.

Dashboard switch contract:
1. Existing dashboard must provide a mode switch between:
	- Folktables demo view
	- CLEAR-10 demo view
2. Switching mode must not break existing Folktables behavior.
3. Mode switch must be implemented as top tabs with live switching behavior.

Proxy-performance visualization contract (CLEAR-10):
1. Show one stepped-line chart per metric:
	- Accuracy
	- Precision
	- Recall
	- F1
2. X-axis: bucket index (2..10).
3. Y-axis: metric value.
4. Plot two stepped lines:
	- Estimated metric
	- Realized/actual metric
5. Provide UI controls for per-metric lower and upper thresholds.
6. Draw threshold lines as red dotted horizontal lines.
7. Mark estimated points beyond thresholds as red markers (estimated alerts).
8. Mark realized points beyond thresholds as red markers (actual alerts).
9. Use standard default threshold values on initial page load.
10. Any threshold edit must update live:
	- threshold lines on charts
	- estimated alert markers
	- realized/actual alert markers

Drift-detection visualization contract (CLEAR-10):
1. Drift-method trends must also use stepped-line charts over buckets (2..10).
2. Include editable threshold controls where detector semantics support threshold bounds.
3. Threshold edits must update live threshold overlays and alert marker calculations.
4. Mark threshold violations as red alert markers.
4. Keep detector-specific y-axis semantics explicit (for example MMD score scale vs PSI score scale).

Expected outputs:
- Temporal image drift report across buckets 2..10.
- Per-class proxy and actual performance curves.
- Per-metric stepped proxy-vs-actual charts with threshold and alert overlays.
- Stepped drift-detector trend charts with threshold and alert overlays.
- Drift/localization/RCA summaries per bucket.
- Dashboard-ready JSON contract aligned with existing result ingestion patterns.

### Standardized Data Contract for Image-Derived Records

Required columns:
- `feature_1 ... feature_n` (embedding dimensions)
- `y_pred`
- `pred_proba_0 ... pred_proba_(k-1)`
- `y_true` (required in reference dataset; optional/delayed in analysis dataset)
- `timestamp`
- `sample_id`
- Metadata slices such as `region`, `device`, `source`, `image_type`, `class_group`

Storage/runtime notes:
- Reference and analysis tabularized datasets are the primary datasets for downstream monitoring.
- Analysis dataset is append-oriented and supports continuous ingestion.
- Missing/delayed `y_true` in analysis dataset is first-class and expected.

### Workstream Plan

#### Workstream A: Data and Contract Foundation (blocks B/C/D/E/F/G)
- Add image/embedding dataset adapters and schema validators.
- Validate embedding dimensional consistency, probability-vector integrity, and delayed-label handling.
- Validate strict source-of-truth mapping:
	- `feature_*` from system extractor
	- `y_pred`/`pred_proba_*` from user model
- Add config schema fields for extractor choice, class count, metadata columns, and artifact paths.
- Add config fields for chunking strategy and chunk parameters (sliding, quantity-based, temporal).
- Add CLEAR-10 ingestion adapter for `data/clear10/` folder structure:
	- `labeled_images/<bucket>/<class>/*.jpg`
	- `labeled_metadata/<bucket>/...`
	- `class_names.txt`

#### Workstream B: Embedding Extraction Module (depends on A)
- Implement extractor interface + registry.
- Implement ResNet baseline extractor.
- Add monitored-model adapter interface for user-supplied classifiers that provide `y_pred` and `pred_proba_*`.
- Persist extraction metadata for reproducibility.
- Add extension hooks for future extractors (CLIP, SimCLR, DINOv2, MAE, ViT, ConvNeXt, EfficientNet).
- For CLEAR-10 demo only, allow same-model configuration where ResNet serves both classifier and extractor roles.

#### Workstream C: Multiclass Proxy Performance Estimation (depends on A/B)
- Add M-CBPE-style multiclass proxy estimator.
- Provide aggregate and class-wise performance proxy outputs.
- Support delayed/missing labels in analysis stream.
- Add proxy-quality evaluation when `y_true` is present (for CLEAR-10 demo validation).

#### Workstream D: Embedding Drift Detection (depends on A/B)
- Implement mandatory methods:
	- MMD on embedding vectors
	- KS/PSI on embedding dimensions
	- PCA reconstruction error
	- FID-style distance
- Integrate into detector registry and threshold contracts.

#### Workstream E: Localization and Slice Analysis (depends on A/B/D)
- Reuse generic slice engine on metadata columns and class-based slices.
- Provide per-slice and per-class drift localization summaries.
- Preserve low-sample safeguards.

#### Workstream F: Embedding RCA (depends on C/D/E)
- Attribute drift to embedding dimensions/components.
- Summarize representation shift and correlate with prediction/output changes.
- Include optional projection artifacts (UMAP/t-SNE) as report attachments.

#### Workstream G: Dashboard Extensions (depends on C/D/E/F)
- Add image/embedding analysis mode to dashboard loader and UI.
- Add class-wise proxy and embedding drift visualizations.
- Add proxy vs actual comparison view for labeled demo datasets (CLEAR-10).
- Add explicit dashboard mode switch: Folktables <-> CLEAR-10 via top tabs.
- Enforce section ordering contract: baseline -> proxy -> drift -> localization -> RCA.
- Implement stepped-line chart templates with editable threshold controls and live alert recomputation.
- Seed threshold controls with standard default values on first load.
- Preserve backward compatibility for existing tabular dashboards.

#### Workstream H: Remediation Reporting Baseline (depends on C/D/E/F)
- Add remediation recommendations and report templates for image pipelines.
- Define trigger criteria and manual action playbooks.
- Defer fully automated retraining/champion-challenger/rollback orchestration to later phase.

### Acceptance Criteria for Phase 2

1. End-to-end image classification run produces standardized tabularized embedding records.
2. Reference vs analysis execution supports missing/delayed `y_true` in analysis data.
3. All mandatory Phase 2 methods execute and serialize consistently.
4. Slice-level localization works for metadata and class-based slices.
5. Embedding RCA outputs actionable shift attribution and output correlation.
6. Dashboard renders image-mode outputs without regressing tabular views.
7. CLEAR-10 demo runs end-to-end using bucket-1 reference and buckets 2..10 analysis.
8. CLEAR-10 outputs include proxy-vs-actual performance comparison metrics per bucket/class.
9. CLEAR-10 dashboard includes mode switch and section ordering exactly as defined.
10. Proxy metric charts (accuracy, precision, recall, F1) are stepped lines with threshold lines and red alert markers for estimated and actual threshold violations.
11. Drift detector charts are stepped lines with threshold overlays and alert markers where applicable.
12. Dashboard mode switch is implemented with top tabs and switches views live.
13. Editing thresholds in UI updates chart threshold lines and alert markers live for proxy and drift sections.

### Deferred (Post-Phase 2)

- Full visual RCA (Grad-CAM/saliency).
- Autoencoder-based embedding reconstruction detector (optional later expansion).
- Automated remediation orchestration.
- Streaming/real-time image monitoring runtime.
