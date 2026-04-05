"""CLEAR-10 report assembly helpers for dashboard contracts."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Optional
import json

import pandas as pd
import numpy as np

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.data.proxy_metrics import MulticlassProxyEstimator


def _json_safe_float(value: float) -> float | None:
    """Return finite float values, otherwise None for strict JSON compatibility."""
    result = float(value)
    return result if math.isfinite(result) else None


def _resolve_reference_key(
    bucket_key: str,
    fallback_reference_key: str,
    reference_by_bucket: Dict[str, str] | None,
) -> str:
    if reference_by_bucket is None:
        return fallback_reference_key
    return str(reference_by_bucket.get(bucket_key, fallback_reference_key))


def _severity_rank(severity: str) -> int:
    order = {
        "none": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }
    return order.get(str(severity).lower(), 0)


def _sanitize_file_stem(value: str) -> str:
    allowed = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        else:
            allowed.append("_")
    stem = "".join(allowed).strip("_")
    return stem or "sample"


def _denormalize_tensor_image(image_tensor):
    """Convert normalized CHW tensor back to uint8 RGB for overlay rendering."""
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)

    image = image_tensor.detach().cpu().clone()
    image = image * std + mean
    image = image.clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    return (image * 255.0).astype(np.uint8)


def _jet_colormap(cam: np.ndarray) -> np.ndarray:
    """Map heat values in [0,1] to a jet-like RGB spectrum (blue->cyan->yellow->red)."""
    x = np.clip(cam.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _overlay_spectrum_heatmap(rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend CAM heat into original image using a full color spectrum overlay."""
    base = rgb_image.astype(np.float32) / 255.0
    cam_rgb = _jet_colormap(heatmap)
    blended = base * (1.0 - alpha) + cam_rgb * alpha
    return np.clip(blended * 255.0, 0.0, 255.0).astype(np.uint8)


def _maybe_load_monitored_model(artifacts_dir: Optional[str | Path]):
    """Load persisted monitored model for visual RCA; fail-soft by returning reason strings."""
    if artifacts_dir is None:
        return None, "artifacts_dir not provided"

    model_path = Path(artifacts_dir) / "baseline_model.pkl"
    if not model_path.exists():
        return None, f"monitored model artifact missing at {model_path}"

    try:
        from drift_autopsy.data.image_baseline import ResNetMonitoredClassifier

        model = ResNetMonitoredClassifier.load(str(model_path))
        if model.model is None:
            return None, "monitored model is not loaded"
        return model, None
    except Exception as exc:  # pragma: no cover - environment-dependent
        return None, f"failed to load monitored model: {exc}"


def _rank_high_drift_samples(
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    drift_scores: Dict[str, Any],
    max_images: int,
) -> pd.DataFrame:
    ranked = frame.copy()

    feature_names = [
        name
        for name, _ in sorted(
            drift_scores.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
        if str(name) in frame.columns and str(name) in reference_frame.columns
    ][:20]

    if feature_names:
        ref_center = reference_frame[feature_names].mean(axis=0)
        abs_delta = (ranked[feature_names] - ref_center).abs()
        weights = np.array([abs(float(drift_scores.get(name, 1.0))) for name in feature_names], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        ranked["_visual_drift_score"] = abs_delta.to_numpy(dtype=float) @ weights
    else:
        proba_cols = [c for c in ranked.columns if c.startswith("pred_proba_")]
        if proba_cols:
            ranked["_visual_drift_score"] = 1.0 - ranked[proba_cols].max(axis=1)
        else:
            ranked["_visual_drift_score"] = np.arange(len(ranked), 0, -1, dtype=float)

    if "sample_id" in ranked.columns:
        ranked["_sort_sample_id"] = ranked["sample_id"].astype(str)
    else:
        ranked["_sort_sample_id"] = ranked.index.astype(str)

    ranked = ranked.sort_values(["_visual_drift_score", "_sort_sample_id"], ascending=[False, True])
    return ranked.head(max_images).reset_index(drop=True)


def _generate_visual_rca_samples(
    frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    drift_scores: Dict[str, Any],
    monitored_model,
    output_dir: Path,
    bucket_key: str,
    max_images: int,
) -> Dict[str, Any]:
    """Generate Grad-CAM overlays for top high-drift images with GPU acceleration when available."""
    if monitored_model is None or monitored_model.model is None:
        return {
            "enabled": False,
            "status": "skipped",
            "reason": "monitored model is not loaded",
            "samples": [],
        }

    if "image_path" not in frame.columns:
        return {
            "enabled": False,
            "status": "skipped",
            "reason": "image_path column missing",
            "samples": [],
        }

    try:
        import torch
        import torch.nn.functional as F
        from PIL import Image
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "enabled": False,
            "status": "skipped",
            "reason": f"torch/PIL unavailable: {exc}",
            "samples": [],
        }

    model = monitored_model.model
    model.eval()

    if not hasattr(model, "layer4"):
        return {
            "enabled": False,
            "status": "skipped",
            "reason": "unsupported monitored model architecture for Grad-CAM",
            "samples": [],
        }

    ranked = _rank_high_drift_samples(
        frame=frame,
        reference_frame=reference_frame,
        drift_scores=drift_scores,
        max_images=max_images,
    )
    if ranked.empty:
        return {
            "enabled": False,
            "status": "skipped",
            "reason": "no candidate images available",
            "samples": [],
        }

    bucket_dir = output_dir / f"bucket_{bucket_key}"
    bucket_dir.mkdir(parents=True, exist_ok=True)

    activations = {}
    target_layer = model.layer4[-1]

    def _forward_hook(_, __, output):
        activations["value"] = output

    hook_fwd = target_layer.register_forward_hook(_forward_hook)

    samples = []
    failure_reasons: Counter[str] = Counter()

    def _record_failure(reason: str) -> None:
        failure_reasons[str(reason)] += 1

    try:
        for idx, row in ranked.iterrows():
            image_path = Path(str(row.get("image_path", "")))
            if not image_path.exists():
                _record_failure("image path does not exist")
                continue

            try:
                input_tensor = monitored_model._load_image_tensor(str(image_path)).unsqueeze(0).to(monitored_model.device)
                input_tensor.requires_grad_(True)
                logits = model(input_tensor)

                acts = activations.get("value")
                if acts is None:
                    _record_failure("target layer activation missing")
                    continue

                if "y_pred" in row and pd.notna(row["y_pred"]):
                    class_idx = int(row["y_pred"])
                else:
                    class_idx = int(torch.argmax(logits, dim=1).item())
                class_idx = max(0, min(class_idx, logits.shape[1] - 1))

                score = logits[:, class_idx].sum()
                grads = torch.autograd.grad(score, acts, retain_graph=False, allow_unused=True)[0]
                if grads is None:
                    _record_failure("target layer gradients missing")
                    continue

                pooled_grads = torch.mean(grads, dim=(2, 3), keepdim=True)
                cam = torch.relu(torch.sum(pooled_grads * acts, dim=1, keepdim=True))
                cam = F.interpolate(
                    cam,
                    size=(input_tensor.shape[2], input_tensor.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
                cam = cam[0, 0].detach().cpu().numpy()
                cam_min = float(cam.min())
                cam_max = float(cam.max())
                if cam_max > cam_min:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                else:
                    cam = np.zeros_like(cam)

                rgb = _denormalize_tensor_image(input_tensor[0])
                overlay = _overlay_spectrum_heatmap(rgb, cam)

                sample_id = _sanitize_file_stem(str(row.get("sample_id", f"row_{idx}")))
                input_path = bucket_dir / f"{sample_id}_input.png"
                overlay_path = bucket_dir / f"{sample_id}_gradcam.png"
                Image.fromarray(rgb).save(input_path)
                Image.fromarray(overlay).save(overlay_path)

                samples.append(
                    {
                        "rank": int(idx + 1),
                        "sample_id": str(row.get("sample_id", sample_id)),
                        "class_name": str(row.get("class_name", "")),
                        "input_image_path": str(input_path.resolve()),
                        "image_path": str(image_path),
                        "gradcam_path": str(overlay_path.resolve()),
                        "drift_score": _json_safe_float(float(row.get("_visual_drift_score", 0.0))),
                    }
                )
            except Exception as exc:
                _record_failure(f"{type(exc).__name__}: {exc}")
                continue
    finally:
        hook_fwd.remove()

    status = "generated" if samples else "skipped"
    if samples:
        reason = None
    else:
        reason = "no Grad-CAM artifacts generated"
        if failure_reasons:
            top_reason, top_count = failure_reasons.most_common(1)[0]
            reason = f"{reason} (top failure: {top_reason}, count={top_count})"
    return {
        "enabled": bool(samples),
        "status": status,
        "reason": reason,
        "samples": samples,
        "max_images": int(max_images),
        "preprocessing": {
            "policy": "model_transform_resize_center_crop",
            "image_size": int(getattr(monitored_model, "image_size", 224)),
            "device": str(getattr(monitored_model, "device", "cpu")),
        },
    }


def build_clear10_proxy_report(
    bucket_frames: Dict[str, pd.DataFrame],
    baseline_metrics: Dict[str, float] | None,
    reference_bucket: int,
    reference_by_bucket: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Build dashboard-ready proxy-vs-actual report for CLEAR-10 buckets."""
    ref_key = str(reference_bucket)
    if ref_key not in bucket_frames:
        raise ValueError(f"Reference bucket {reference_bucket} not found in bucket_frames")

    estimator = MulticlassProxyEstimator(n_bins=10)

    proxy_metrics = []
    proxy_metrics_classwise = []
    proxy_quality_gaps = []
    bucket_results: Dict[str, Any] = {}

    class_id_to_name: Dict[int, str] = {}
    reference_frame = bucket_frames[ref_key]
    if "class_name" in reference_frame.columns and "y_true" in reference_frame.columns:
        class_map = (
            reference_frame[["y_true", "class_name"]]
            .dropna()
            .drop_duplicates(subset=["y_true"])
            .sort_values("y_true")
        )
        for _, row in class_map.iterrows():
            class_id_to_name[int(row["y_true"])] = str(row["class_name"])

    for bucket_key, frame in sorted(bucket_frames.items(), key=lambda item: int(item[0])):
        bucket_num = int(bucket_key)
        if bucket_num == reference_bucket:
            continue

        runtime_ref_key = _resolve_reference_key(bucket_key, ref_key, reference_by_bucket)
        if runtime_ref_key not in bucket_frames:
            raise ValueError(f"Reference bucket {runtime_ref_key} not found for analysis bucket {bucket_key}")

        estimator.fit(bucket_frames[runtime_ref_key])

        metrics = estimator.estimate(frame)
        for metric_name in ("accuracy", "precision", "recall", "f1"):
            proxy_metrics.append(
                {
                    "bucket": bucket_num,
                    "metric": metric_name,
                    "estimated": float(metrics.estimated[metric_name]),
                    "actual": _json_safe_float(metrics.actual[metric_name]) if metrics.actual[metric_name] is not None else None,
                }
            )

            if metric_name in metrics.proxy_quality_gap:
                proxy_quality_gaps.append(
                    {
                        "bucket": bucket_num,
                        "metric": metric_name,
                        "gap": float(metrics.proxy_quality_gap[metric_name]),
                    }
                )

        for class_key, estimated_payload in metrics.class_wise_estimated.items():
            class_id = int(class_key.split("_")[-1])
            actual_payload = metrics.class_wise_actual.get(class_key, {})
            gap_payload = metrics.class_wise_proxy_quality_gap.get(class_key, {})

            for metric_name in ("precision", "recall", "f1"):
                actual_value = actual_payload.get(metric_name)
                proxy_metrics_classwise.append(
                    {
                        "bucket": bucket_num,
                        "class_id": class_id,
                        "class_name": class_id_to_name.get(class_id, f"class_{class_id}"),
                        "metric": metric_name,
                        "estimated": float(estimated_payload[metric_name]),
                        "actual": _json_safe_float(actual_value) if actual_value is not None else None,
                        "gap": _json_safe_float(gap_payload.get(metric_name)) if metric_name in gap_payload else None,
                    }
                )

        bucket_results[bucket_key] = {
            "proxy_performance": {
                "estimated": metrics.estimated,
                "actual": metrics.actual,
                "class_wise_estimated": metrics.class_wise_estimated,
                "class_wise_actual": metrics.class_wise_actual,
                "proxy_quality_gap": metrics.proxy_quality_gap,
                "class_wise_proxy_quality_gap": metrics.class_wise_proxy_quality_gap,
            }
        }

    return {
        "baseline_performance": baseline_metrics or {},
        "proxy_metrics": proxy_metrics,
        "proxy_metrics_classwise": proxy_metrics_classwise,
        "proxy_quality_gaps": proxy_quality_gaps,
        "bucket_results": bucket_results,
        "metadata": {
            "analysis_type": "clear10_proxy_report",
            "reference_bucket": reference_bucket,
        },
    }


def _frame_to_dataset(frame: pd.DataFrame) -> Dataset:
    feature_cols = sorted([c for c in frame.columns if c.startswith("feature_")], key=lambda x: int(x.split("_")[-1]))
    proba_cols = sorted([c for c in frame.columns if c.startswith("pred_proba_")], key=lambda x: int(x.split("_")[-1]))

    return Dataset(
        data=frame[feature_cols].reset_index(drop=True),
        feature_names=feature_cols,
        target=frame["y_true"].reset_index(drop=True),
        target_name="y_true",
        predictions=frame["y_pred"].to_numpy(dtype=int),
        prediction_probabilities=frame[proba_cols].to_numpy(dtype=float),
        metadata=frame[[c for c in ["class_name", "device", "source"] if c in frame.columns]].reset_index(drop=True)
        if any(c in frame.columns for c in ["class_name", "device", "source"])
        else None,
    )


def _build_slice_summary_rows(slice_payloads: Dict[str, Any], column_name: str) -> list[Dict[str, Any]]:
    """Convert pipeline slice payloads into compact, dashboard-friendly summary rows."""
    rows: list[Dict[str, Any]] = []

    for _, payload in slice_payloads.items():
        result_payload = payload.get("result", {})
        detection = result_payload.get("detection", {})
        localization = result_payload.get("localization") or {}
        drifted_features = localization.get("drifted_features") or []

        rows.append(
            {
                "column": column_name,
                "reference_slice": str(payload.get("reference_slice_value")),
                "test_slice": str(payload.get("test_slice_value")),
                "drift_detected": bool(detection.get("drift_detected", False)),
                "severity": detection.get("severity", "none"),
                "score": _json_safe_float(detection.get("score", 0.0)),
                "reference_samples": int(payload.get("reference_samples", 0)),
                "test_samples": int(payload.get("test_samples", 0)),
                "n_drifted_features": len(drifted_features),
                "top_features": list(drifted_features[:5]),
            }
        )

    return rows


def _run_slice_localization(
    reference_ds: Dataset,
    test_ds: Dataset,
    detector_threshold: float,
    localizer_threshold: float,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Run slice localization for class and metadata columns with built-in low-sample safeguards."""
    from drift_autopsy.core.pipeline import DriftPipeline
    from drift_autopsy.detectors import KSTest
    from drift_autopsy.localizers import UnivariateLocalizer

    if reference_ds.metadata is None or test_ds.metadata is None:
        return [], []

    candidate_columns = [
        col for col in ["class_name", "source", "device"] if col in reference_ds.metadata.columns and col in test_ds.metadata.columns
    ]
    if not candidate_columns:
        return [], []

    class_rows: list[Dict[str, Any]] = []
    metadata_rows: list[Dict[str, Any]] = []

    for column_name in candidate_columns:
        slice_pipeline = DriftPipeline(
            detector=KSTest(threshold=detector_threshold),
            localizer=UnivariateLocalizer(threshold=localizer_threshold),
            enable_localization=True,
            enable_rca=False,
            validate_data=False,
        )
        slice_result = slice_pipeline.run(
            reference_ds,
            test_ds,
            slice_config={
                "enabled": True,
                "column": column_name,
                "min_samples_per_slice": 30,
            },
        )
        slice_payloads = (
            slice_result.metadata.get("slice_analysis", {}).get("slices", {})
            if slice_result.metadata
            else {}
        )
        slice_rows = _build_slice_summary_rows(slice_payloads, column_name=column_name)

        if column_name == "class_name":
            class_rows.extend(slice_rows)
        else:
            metadata_rows.extend(slice_rows)

    return class_rows, metadata_rows


def build_clear10_full_report(
    bucket_frames: Dict[str, pd.DataFrame],
    baseline_metrics: Dict[str, float] | None,
    reference_bucket: int,
    reference_by_bucket: Dict[str, str] | None = None,
    artifacts_dir: Optional[str | Path] = None,
    visual_rca_max_images: int = 5,
    visual_rca_min_severity: str = "high",
) -> Dict[str, Any]:
    """Build full CLEAR-10 dashboard-ready report with proxy, drift, localization, and RCA."""
    # Import analysis components lazily to avoid package initialization cycles.
    from drift_autopsy.core.pipeline import DriftPipeline
    from drift_autopsy.detectors import KSTest, PSI, MMD, PCAReconstructionError, FIDDistance
    from drift_autopsy.localizers import UnivariateLocalizer

    report = build_clear10_proxy_report(
        bucket_frames=bucket_frames,
        baseline_metrics=baseline_metrics,
        reference_bucket=reference_bucket,
        reference_by_bucket=reference_by_bucket,
    )

    ref_key = str(reference_bucket)
    if ref_key not in bucket_frames:
        raise ValueError(f"Reference bucket {reference_bucket} not found in bucket_frames")

    detector_specs = [
        ("Ks Test", KSTest(threshold=0.05)),
        ("Psi", PSI(threshold=0.025, n_bins=10)),
        ("Mmd", MMD(threshold=0.02, kernel="rbf", n_permutations=20, max_samples=3000)),
        ("Pca Reconstruction", PCAReconstructionError(threshold=0.28, explained_variance_ratio=0.95)),
        ("Fid Distance", FIDDistance(threshold=14.0, covariance_eps=1e-6)),
    ]

    drift_results = []
    bucket_results = report.get("bucket_results", {})
    monitored_model, monitored_model_error = _maybe_load_monitored_model(artifacts_dir)
    visual_output_root = None
    if artifacts_dir is not None:
        visual_output_root = Path(artifacts_dir).parent / "clear10_visual_rca"
        visual_output_root.mkdir(parents=True, exist_ok=True)

    for bucket_key, frame in sorted(bucket_frames.items(), key=lambda item: int(item[0])):
        bucket_num = int(bucket_key)
        if bucket_num == reference_bucket:
            continue

        runtime_ref_key = _resolve_reference_key(bucket_key, ref_key, reference_by_bucket)
        if runtime_ref_key not in bucket_frames:
            raise ValueError(f"Reference bucket {runtime_ref_key} not found for analysis bucket {bucket_key}")

        reference_ds = _frame_to_dataset(bucket_frames[runtime_ref_key])

        test_ds = _frame_to_dataset(frame)

        per_detector = {}
        localization_payload = None
        class_slice_summary: list[Dict[str, Any]] = []
        metadata_slice_summary: list[Dict[str, Any]] = []

        for detector_display_name, detector_instance in detector_specs:
            pipeline = DriftPipeline(
                detector=detector_instance,
                localizer=UnivariateLocalizer(threshold=0.05),
                enable_localization=True,
                enable_rca=False,
                validate_data=False,
            )
            result = pipeline.run(reference_ds, test_ds)

            detection = result.detection
            drift_results.append(
                {
                    "bucket": bucket_num,
                    "detector": detector_display_name,
                    "score": _json_safe_float(detection.score),
                    "threshold": _json_safe_float(detection.threshold),
                    "drift_detected": bool(detection.drift_detected),
                    "severity": detection.severity.value,
                }
            )

            per_detector[detector_display_name.lower().replace(" ", "_")] = {
                "score": _json_safe_float(detection.score),
                "threshold": _json_safe_float(detection.threshold),
                "drift_detected": bool(detection.drift_detected),
                "severity": detection.severity.value,
            }

            if detector_display_name == "Ks Test" and result.localization is not None:
                localization_payload = {
                    "drifted_features": list(result.localization.drifted_features),
                    "drift_scores": dict(result.localization.drift_scores),
                }
                class_slice_summary, metadata_slice_summary = _run_slice_localization(
                    reference_ds=reference_ds,
                    test_ds=test_ds,
                    detector_threshold=0.05,
                    localizer_threshold=0.05,
                )

        if bucket_key not in bucket_results:
            bucket_results[bucket_key] = {}

        bucket_results[bucket_key]["detectors"] = per_detector

        if localization_payload is None:
            localization_payload = {"drifted_features": [], "drift_scores": {}}

        localization_payload["class_slice_summary"] = class_slice_summary
        localization_payload["metadata_slice_summary"] = metadata_slice_summary
        localization_payload["slice_columns_evaluated"] = sorted(
            list({row["column"] for row in class_slice_summary + metadata_slice_summary})
        )

        bucket_results[bucket_key]["localization"] = localization_payload

        top_changes = sorted(
            localization_payload.get("drift_scores", {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_change_names = [name for name, _ in top_changes[:5]]

        embedding_shift_attribution = [
            {"feature": name, "score": _json_safe_float(score)}
            for name, score in top_changes[:10]
        ]

        proxy_payload = bucket_results[bucket_key].get("proxy_performance", {})
        proxy_quality_gap = proxy_payload.get("proxy_quality_gap", {}) or {}
        class_wise_gap = proxy_payload.get("class_wise_proxy_quality_gap", {}) or {}

        largest_gap_metric = None
        largest_gap_value = None
        if proxy_quality_gap:
            largest_gap_metric, largest_gap_value = max(
                proxy_quality_gap.items(),
                key=lambda item: abs(float(item[1])),
            )

        class_gap_summary = []
        for class_key, gap_metrics in class_wise_gap.items():
            if not gap_metrics:
                continue
            mean_abs_gap = float(sum(abs(float(v)) for v in gap_metrics.values()) / len(gap_metrics))
            class_gap_summary.append(
                {
                    "class_key": str(class_key),
                    "mean_abs_gap": _json_safe_float(mean_abs_gap),
                    "gap_metrics": {
                        metric: _json_safe_float(float(value))
                        for metric, value in gap_metrics.items()
                    },
                }
            )
        class_gap_summary = sorted(
            class_gap_summary,
            key=lambda row: abs(float(row["mean_abs_gap"] or 0.0)),
            reverse=True,
        )

        drifted_class_slices = [
            row
            for row in localization_payload.get("class_slice_summary", [])
            if row.get("drift_detected")
        ]
        drifted_metadata_slices = [
            row
            for row in localization_payload.get("metadata_slice_summary", [])
            if row.get("drift_detected")
        ]

        recommendations = []
        if top_change_names:
            recommendations.append(
                f"Audit embedding dimensions with strongest shift: {', '.join(top_change_names[:3])}."
            )
        if largest_gap_metric is not None and largest_gap_value is not None:
            recommendations.append(
                f"Prioritize '{largest_gap_metric}' calibration drift (proxy gap={float(largest_gap_value):.4f}) for this bucket."
            )
        if class_gap_summary:
            recommendations.append(
                f"Investigate class-level degradation around '{class_gap_summary[0]['class_key']}' first."
            )
        if drifted_class_slices:
            first_slice = drifted_class_slices[0]
            recommendations.append(
                "Inspect drifted class slice "
                f"{first_slice.get('reference_slice')}->{first_slice.get('test_slice')} "
                "for targeted data refresh."
            )
        if drifted_metadata_slices:
            first_meta = drifted_metadata_slices[0]
            recommendations.append(
                "Inspect metadata slice "
                f"[{first_meta.get('column')}] {first_meta.get('reference_slice')}->{first_meta.get('test_slice')} "
                "for ingestion/collection shifts."
            )

        visual_rca = {
            "enabled": False,
            "status": "skipped",
            "reason": "bucket severity below threshold",
            "samples": [],
            "max_images": int(visual_rca_max_images),
        }

        bucket_severity = "none"
        for payload in per_detector.values():
            severity_name = str(payload.get("severity", "none"))
            if _severity_rank(severity_name) > _severity_rank(bucket_severity):
                bucket_severity = severity_name

        if _severity_rank(bucket_severity) >= _severity_rank(visual_rca_min_severity):
            if monitored_model_error:
                visual_rca = {
                    "enabled": False,
                    "status": "skipped",
                    "reason": monitored_model_error,
                    "samples": [],
                    "max_images": int(visual_rca_max_images),
                }
            elif monitored_model is None or visual_output_root is None:
                visual_rca = {
                    "enabled": False,
                    "status": "skipped",
                    "reason": "visual RCA prerequisites unavailable",
                    "samples": [],
                    "max_images": int(visual_rca_max_images),
                }
            else:
                visual_rca = _generate_visual_rca_samples(
                    frame=frame,
                    reference_frame=bucket_frames[runtime_ref_key],
                    drift_scores=localization_payload.get("drift_scores", {}),
                    monitored_model=monitored_model,
                    output_dir=visual_output_root,
                    bucket_key=bucket_key,
                    max_images=visual_rca_max_images,
                )
                visual_rca["bucket_severity"] = bucket_severity
        else:
            visual_rca["bucket_severity"] = bucket_severity

        bucket_results[bucket_key]["rca"] = {
            "top_changes": top_change_names,
            "embedding_shift_attribution": embedding_shift_attribution,
            "output_correlation": {
                "proxy_quality_gap": {
                    metric: _json_safe_float(float(value))
                    for metric, value in proxy_quality_gap.items()
                },
                "largest_gap_metric": largest_gap_metric,
                "largest_gap_value": _json_safe_float(float(largest_gap_value))
                if largest_gap_value is not None
                else None,
            },
            "class_gap_summary": class_gap_summary[:5],
            "slice_correlation": {
                "drifted_class_slices": drifted_class_slices[:10],
                "drifted_metadata_slices": drifted_metadata_slices[:10],
            },
            "recommendations": recommendations,
            "visual_rca": visual_rca,
        }

    report["drift_results"] = drift_results
    report["bucket_results"] = bucket_results
    report.setdefault("metadata", {})["analysis_type"] = "clear10_full_report"
    return report


def save_clear10_proxy_report(report: Dict[str, Any], output_path: str | Path) -> Path:
    """Persist CLEAR-10 proxy report to a JSON file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    return output
