"""Config-driven image embedding tabularization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

import numpy as np
import pandas as pd

from drift_autopsy.config.schema import ImageDataConfig, PipelineConfig
from drift_autopsy.data.image_baseline import EmbeddingBaselineClassifier, create_monitored_model
from drift_autopsy.data.loaders import Clear10Loader
from drift_autopsy.data.validators import DataValidator
from drift_autopsy.registry import ExtractorRegistry

logger = logging.getLogger(__name__)


class ImageTabularizationRunner:
    """Build standardized tabular embedding datasets from image buckets."""

    def __init__(self, image_config: ImageDataConfig):
        self.image_config = image_config
        self.root_path = Path(image_config.root_path)

        # Ensure built-in extractors are imported so registration side effects run.
        try:
            import drift_autopsy.extractors  # noqa: F401
        except Exception:  # pragma: no cover - optional backend import path
            logger.debug("Could not import drift_autopsy.extractors at init time", exc_info=True)

        self.extractor = ExtractorRegistry.create(
            image_config.extractor_name,
            **image_config.extractor_params,
        )

        self.class_names = Clear10Loader.load_class_names(self.root_path)
        self.class_count = len(self.class_names)
        self.baseline_classifier: Optional[EmbeddingBaselineClassifier] = None
        self.baseline_metrics: Optional[Dict[str, float]] = None

    @classmethod
    def from_pipeline_config(cls, config: PipelineConfig) -> "ImageTabularizationRunner":
        """Construct runner from a full PipelineConfig."""
        if config.image_data is None:
            raise ValueError("PipelineConfig.image_data is required for image tabularization")
        return cls(config.image_data)

    def _bootstrap_predictions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Create temporary y_pred and pred_proba_* from y_true labels."""
        if "y_true" not in frame.columns:
            raise ValueError("Cannot bootstrap predictions without y_true")

        y_true = frame["y_true"].to_numpy(dtype=int)
        frame["y_pred"] = y_true

        proba = np.zeros((len(frame), self.class_count), dtype=float)
        valid = (y_true >= 0) & (y_true < self.class_count)
        proba[np.arange(len(frame))[valid], y_true[valid]] = 1.0

        for class_idx in range(self.class_count):
            frame[f"pred_proba_{class_idx}"] = proba[:, class_idx]

        return frame

    def build_bucket_dataframe(self, bucket: int) -> pd.DataFrame:
        """Build one bucket's standardized tabular embedding dataframe."""
        base_df = Clear10Loader.build_bucket_dataframe(
            root_path=self.root_path,
            bucket=bucket,
            include_background=self.image_config.include_background,
            max_samples_per_class=self.image_config.max_samples_per_class,
        )

        image_paths = base_df["image_path"].astype(str).tolist()
        embeddings = self.extractor.extract(image_paths)

        if embeddings.shape[0] != len(base_df):
            raise ValueError(
                f"Extractor row mismatch for bucket={bucket}: "
                f"embeddings={embeddings.shape[0]} rows={len(base_df)}"
            )

        embedding_cols = [f"feature_{idx}" for idx in range(embeddings.shape[1])]
        embedding_df = pd.DataFrame(embeddings, columns=embedding_cols)
        output_df = pd.concat([embedding_df, base_df.reset_index(drop=True)], axis=1)
        return output_df

    def _validate_bucket_contract(self, frame: pd.DataFrame, bucket: int) -> None:
        is_reference_bucket = bucket == self.image_config.reference_bucket
        allow_missing_y_true = (
            self.image_config.allow_missing_analysis_y_true and not is_reference_bucket
        )
        DataValidator.validate_embedding_contract(
            frame,
            name=f"clear10_bucket_{bucket}",
            expected_embedding_dim=self.image_config.expected_embedding_dim,
            expected_class_count=(
                self.image_config.expected_class_count
                if self.image_config.expected_class_count is not None
                else self.class_count
            ),
            allow_missing_y_true=allow_missing_y_true,
        )

    def _fit_baseline_classifier(self, reference_df: pd.DataFrame) -> None:
        monitored_model_name = (
            self.image_config.monitored_model_name
            if self.image_config.monitored_model_name
            else self.image_config.baseline_model_name
        )
        monitored_model_params = (
            self.image_config.monitored_model_params
            if self.image_config.monitored_model_params
            else self.image_config.baseline_model_params
        )

        classifier = create_monitored_model(
            model_name=monitored_model_name,
            model_params=monitored_model_params,
        )
        self.baseline_metrics = classifier.fit_with_split(
            reference_df=reference_df,
            train_fraction=self.image_config.baseline_train_fraction,
            random_state=self.image_config.baseline_random_state,
        )
        self.baseline_classifier = classifier

    def _attach_predictions(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.image_config.bootstrap_predictions_from_y_true:
            return self._bootstrap_predictions(frame)

        if self.baseline_classifier is None:
            raise RuntimeError(
                "Baseline classifier is not fitted. "
                "Call build_reference_and_analysis() to train on reference bucket first."
            )

        return self.baseline_classifier.attach_predictions(frame, class_count=self.class_count)

    def build_reference_and_analysis(self) -> Dict[str, pd.DataFrame]:
        """Build standardized dataframes for reference and all analysis buckets."""
        buckets = [self.image_config.reference_bucket] + list(self.image_config.analysis_buckets)
        raw_results: Dict[str, pd.DataFrame] = {}
        results: Dict[str, pd.DataFrame] = {}

        for bucket in buckets:
            raw_results[str(bucket)] = self.build_bucket_dataframe(bucket)

        reference_key = str(self.image_config.reference_bucket)
        if not self.image_config.bootstrap_predictions_from_y_true:
            self._fit_baseline_classifier(raw_results[reference_key])

        for bucket_key, frame in raw_results.items():
            with_predictions = self._attach_predictions(frame)
            self._validate_bucket_contract(with_predictions, bucket=int(bucket_key))
            results[bucket_key] = with_predictions

        return results

    def persist_artifacts(
        self,
        bucket_frames: Dict[str, pd.DataFrame],
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Persist bucket tabular data and extractor metadata to disk."""
        target_dir = output_dir or Path(
            self.image_config.artifacts_dir
            or (self.root_path / "artifacts" / "image_tabularized")
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        for bucket, frame in bucket_frames.items():
            frame.to_parquet(target_dir / f"bucket_{bucket}.parquet", index=False)

        metadata = {
            "dataset": self.image_config.dataset,
            "root_path": str(self.root_path),
            "reference_bucket": self.image_config.reference_bucket,
            "analysis_buckets": list(self.image_config.analysis_buckets),
            "chunking_strategy": self.image_config.chunking_strategy,
            "chunk_size_records": self.image_config.chunk_size_records,
            "chunk_duration": self.image_config.chunk_duration,
            "reference_mode": self.image_config.reference_mode,
            "class_names": self.class_names,
            "extractor": self.extractor.get_extraction_metadata(),
            "allow_missing_analysis_y_true": self.image_config.allow_missing_analysis_y_true,
            "bootstrap_predictions_from_y_true": self.image_config.bootstrap_predictions_from_y_true,
            "baseline_model_name": self.image_config.baseline_model_name,
            "baseline_model_params": self.image_config.baseline_model_params,
            "monitored_model_name": (
                self.image_config.monitored_model_name
                if self.image_config.monitored_model_name
                else self.image_config.baseline_model_name
            ),
            "monitored_model_params": (
                self.image_config.monitored_model_params
                if self.image_config.monitored_model_params
                else self.image_config.baseline_model_params
            ),
            "baseline_metrics": self.baseline_metrics,
        }

        with open(target_dir / "tabularization_metadata.json", "w") as handle:
            json.dump(metadata, handle, indent=2)

        if self.baseline_classifier is not None:
            self.baseline_classifier.save(str(target_dir / "baseline_model.pkl"))

        logger.info("Saved image tabularization artifacts to %s", target_dir)
        return target_dir


def build_image_tabularized_buckets(config: PipelineConfig) -> Dict[str, pd.DataFrame]:
    """Convenience helper for config-driven CLEAR-10 tabularization."""
    runner = ImageTabularizationRunner.from_pipeline_config(config)
    return runner.build_reference_and_analysis()
