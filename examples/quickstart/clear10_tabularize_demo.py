"""CLEAR-10 tabularization demo using config-driven image extraction."""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from drift_autopsy.config import PipelineConfig
from drift_autopsy.data.image_baseline import EmbeddingBaselineClassifier
from drift_autopsy.data import ImageTabularizationRunner
from drift_autopsy.data.clear10_report import build_clear10_full_report, save_clear10_proxy_report
from drift_autopsy.reliability import ReliabilityAnalyzer
from drift_autopsy.utils import setup_logging


def build_demo_config() -> PipelineConfig:
    return PipelineConfig(
        name="clear10_tabularization_demo",
        detector={"type": "ks_test"},
        image_data={
            "dataset": "clear10",
            "root_path": "data/clear10",
            "reference_bucket": 1,
            "analysis_buckets": [2, 3, 4, 5, 6, 7, 8, 9, 10],
            "include_background": True,
            "extractor_name": "resnet",
            "extractor_params": {
                "model_name": "resnet18",
                "weights": "IMAGENET1K_V1",
                "batch_size": 16,
            },
            "artifacts_dir": "outputs/clear10_tabularized_demo",
            "chunking_strategy": "fixed_bucket",
            "reference_mode": "previous_chunk",
            "allow_missing_analysis_y_true": True,
            "bootstrap_predictions_from_y_true": False,
            "monitored_model_name": "resnet_classifier",
            "monitored_model_params": {
                "model_name": "resnet18",
                "weights": "IMAGENET1K_V1",
                "batch_size": 16,
                "epochs": 1,
                "learning_rate": 1e-3,
                "freeze_backbone": True,
            },
            "baseline_model_name": "logistic_regression",
            "baseline_model_params": {
                "max_iter": 1000,
                "random_state": 42,
            },
            "baseline_train_fraction": 0.7,
            "baseline_random_state": 42,
        },
    )


def _feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith("feature_")]


def _estimate_cbpe_proxy(df: pd.DataFrame) -> float | None:
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        return None

    eval_df = df[["y_true", "y_pred"]].dropna()
    if eval_df.empty:
        return None

    y_true = eval_df["y_true"].to_numpy(dtype=int)
    y_pred = eval_df["y_pred"].to_numpy(dtype=int)
    return float(np.mean(y_true == y_pred))


def build_clear10_reliability_records(
    bucket_frames: Dict[str, pd.DataFrame],
    reference_bucket: int,
    reference_by_bucket: Dict[str, str],
    max_samples_per_bucket: int = 80,
) -> Dict[str, List[dict]]:
    """Build reliability records for CLEAR-10 buckets using an embedding surrogate model."""
    reference_key = str(reference_bucket)
    if reference_key not in bucket_frames:
        return {}

    reference_df = bucket_frames[reference_key]
    feature_cols = _feature_columns(reference_df)
    if not feature_cols or "y_true" not in reference_df.columns:
        return {}

    surrogate = EmbeddingBaselineClassifier(
        model_name="logistic_regression",
        model_params={"max_iter": 1000, "random_state": 42},
    )
    surrogate.fit(reference_df)

    reliability_by_bucket: Dict[str, List[dict]] = {}

    for bucket_key, frame in sorted(bucket_frames.items(), key=lambda item: int(item[0])):
        if bucket_key == reference_key or frame.empty:
            continue

        runtime_ref_key = str(reference_by_bucket.get(bucket_key, reference_key))
        runtime_ref = bucket_frames.get(runtime_ref_key, reference_df)

        ref_features = runtime_ref[feature_cols].copy()
        sample_size = min(max_samples_per_bucket, len(frame))
        sample_df = frame.sample(n=sample_size, random_state=42).reset_index(drop=True)
        sample_features = sample_df[feature_cols].copy()

        cbpe_proxy = _estimate_cbpe_proxy(frame)

        analyzer = ReliabilityAnalyzer(
            model=surrogate,
            data_type="tabular",
            reference_data=ref_features,
            task_type="classification",
            cbpe_reference_score=cbpe_proxy,
        )

        prediction_ids = [f"clear10_{bucket_key}_{idx}" for idx in range(len(sample_features))]
        records = analyzer.analyze_batch(
            input_batch=sample_features,
            cbpe_score=cbpe_proxy,
            prediction_ids=prediction_ids,
            shared_explanation_for_batch=True,
        )

        for record in records:
            record["analysis_key"] = str(bucket_key)
            record["detector"] = "embedding_surrogate"

        reliability_by_bucket[bucket_key] = records

    return reliability_by_bucket


def main() -> None:
    setup_logging(level="INFO")

    config = build_demo_config()
    runner = ImageTabularizationRunner.from_pipeline_config(config)

    print("Building tabularized buckets...")
    bucket_frames = runner.build_reference_and_analysis()

    for bucket, frame in bucket_frames.items():
        print(f"  bucket={bucket}: rows={len(frame)} cols={len(frame.columns)}")

    if runner.baseline_metrics:
        print("\nBaseline bucket-1 metrics:")
        for metric_name, metric_value in runner.baseline_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

    out_dir = runner.persist_artifacts(bucket_frames)
    print(f"\nSaved artifacts to: {Path(out_dir).resolve()}")

    report = build_clear10_full_report(
        bucket_frames=bucket_frames,
        baseline_metrics=runner.baseline_metrics,
        reference_bucket=config.image_data.reference_bucket,
        reference_by_bucket=runner.analysis_reference_map,
        artifacts_dir=out_dir,
    )

    reliability_by_bucket = build_clear10_reliability_records(
        bucket_frames=bucket_frames,
        reference_bucket=config.image_data.reference_bucket,
        reference_by_bucket=runner.analysis_reference_map,
    )
    for bucket_key, records in reliability_by_bucket.items():
        report.setdefault("bucket_results", {}).setdefault(bucket_key, {})["reliability"] = records

    print(f"Added reliability outputs for {len(reliability_by_bucket)} CLEAR-10 buckets")

    report_path = save_clear10_proxy_report(report, "outputs/clear10_drift_results.json")
    print(f"Saved CLEAR-10 dashboard report to: {report_path.resolve()}")
    print("Done")


if __name__ == "__main__":
    main()
