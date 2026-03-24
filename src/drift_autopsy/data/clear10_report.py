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


def build_clear10_proxy_report(
    bucket_frames: Dict[str, pd.DataFrame],
    baseline_metrics: Dict[str, float] | None,
    reference_bucket: int,
) -> Dict[str, Any]:
    """Build dashboard-ready proxy-vs-actual report for CLEAR-10 buckets."""
    ref_key = str(reference_bucket)
    if ref_key not in bucket_frames:
        raise ValueError(f"Reference bucket {reference_bucket} not found in bucket_frames")

    estimator = MulticlassProxyEstimator(n_bins=10)
    estimator.fit(bucket_frames[ref_key])

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


def build_clear10_full_report(
    bucket_frames: Dict[str, pd.DataFrame],
    baseline_metrics: Dict[str, float] | None,
    reference_bucket: int,
) -> Dict[str, Any]:
    """Build full CLEAR-10 dashboard-ready report with proxy, drift, localization, and RCA."""
    # Import analysis components lazily to avoid package initialization cycles.
    from drift_autopsy.core.pipeline import DriftPipeline
    from drift_autopsy.detectors import KSTest, PSI, MMD, CBPE, PCAReconstructionError, FIDDistance
    from drift_autopsy.localizers import UnivariateLocalizer

    report = build_clear10_proxy_report(
        bucket_frames=bucket_frames,
        baseline_metrics=baseline_metrics,
        reference_bucket=reference_bucket,
    )

    ref_key = str(reference_bucket)
    if ref_key not in bucket_frames:
        raise ValueError(f"Reference bucket {reference_bucket} not found in bucket_frames")

    reference_ds = _frame_to_dataset(bucket_frames[ref_key])

    detector_specs = [
        ("Ks Test", KSTest(threshold=0.05)),
        ("Psi", PSI(threshold=0.2, n_bins=10)),
        ("Mmd", MMD(threshold=0.1, kernel="rbf", n_permutations=20, max_samples=3000)),
        ("Pca Reconstruction", PCAReconstructionError(threshold=0.15, explained_variance_ratio=0.95)),
        ("Fid Distance", FIDDistance(threshold=50.0, covariance_eps=1e-6)),
        ("Cbpe", CBPE(threshold=0.05, n_bins=10)),
    ]

    drift_results = []
    bucket_results = report.get("bucket_results", {})

    for bucket_key, frame in sorted(bucket_frames.items(), key=lambda item: int(item[0])):
        bucket_num = int(bucket_key)
        if bucket_num == reference_bucket:
            continue

        test_ds = _frame_to_dataset(frame)

        per_detector = {}
        localization_payload = None

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

        if bucket_key not in bucket_results:
            bucket_results[bucket_key] = {}

        bucket_results[bucket_key]["detectors"] = per_detector

        if localization_payload is None:
            localization_payload = {"drifted_features": [], "drift_scores": {}}

        bucket_results[bucket_key]["localization"] = localization_payload

        top_changes = sorted(
            localization_payload.get("drift_scores", {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_change_names = [name for name, _ in top_changes[:5]]

        bucket_results[bucket_key]["rca"] = {
            "top_changes": top_change_names,
            "recommendations": [
                f"Inspect feature shift for '{name}' and evaluate retraining trigger."
                for name in top_change_names[:3]
            ],
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
