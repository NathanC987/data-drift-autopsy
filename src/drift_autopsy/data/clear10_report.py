"""CLEAR-10 report assembly helpers for dashboard contracts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Any
import json

import pandas as pd

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
