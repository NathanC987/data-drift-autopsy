"""CLEAR-10 tabularization demo using config-driven image extraction."""

from pathlib import Path

from drift_autopsy.config import PipelineConfig
from drift_autopsy.data import ImageTabularizationRunner
from drift_autopsy.data.clear10_report import build_clear10_full_report, save_clear10_proxy_report
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
    report_path = save_clear10_proxy_report(report, "outputs/clear10_drift_results.json")
    print(f"Saved CLEAR-10 dashboard report to: {report_path.resolve()}")
    print("Done")


if __name__ == "__main__":
    main()
