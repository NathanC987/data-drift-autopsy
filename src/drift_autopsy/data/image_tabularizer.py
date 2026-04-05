"""Config-driven image embedding tabularization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
import re

import numpy as np
import pandas as pd

from drift_autopsy.config.schema import ImageDataConfig, PipelineConfig
from drift_autopsy.data.image_baseline import MonitoredModelAdapter, create_monitored_model
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
        self.baseline_classifier: Optional[MonitoredModelAdapter] = None
        self.baseline_metrics: Optional[Dict[str, float]] = None
        self.analysis_reference_map: Dict[str, str] = {}

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

    @staticmethod
    def _sorted_numeric_keys(frames: Dict[str, pd.DataFrame]) -> List[str]:
        return sorted(frames.keys(), key=lambda key: int(key))

    @staticmethod
    def _sort_for_quantity_or_sliding(df: pd.DataFrame) -> pd.DataFrame:
        sort_cols = []
        if "sample_id" in df.columns:
            sort_cols.append("sample_id")
        if "timestamp" in df.columns:
            sort_cols.append("timestamp")

        if not sort_cols:
            raise ValueError("Chunking requires at least sample_id or timestamp for deterministic ordering")

        return df.sort_values(sort_cols).reset_index(drop=True)

    @staticmethod
    def _chunk_by_quantity(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
        if chunk_size <= 0:
            raise ValueError("chunk_size_records must be > 0")

        chunks = []
        for start in range(0, len(df), chunk_size):
            chunks.append(df.iloc[start : start + chunk_size].copy())
        return chunks

    @staticmethod
    def _chunk_by_sliding_window(df: pd.DataFrame, window_size: int) -> List[pd.DataFrame]:
        if window_size <= 0:
            raise ValueError("chunk_size_records must be > 0")
        if len(df) <= window_size:
            return [df.copy()]

        stride = max(1, window_size // 2)
        chunks = []
        last_start = len(df) - window_size

        start = 0
        while start <= last_start:
            chunks.append(df.iloc[start : start + window_size].copy())
            start += stride

        if (len(df) - window_size) % stride != 0:
            chunks.append(df.iloc[last_start : last_start + window_size].copy())

        return chunks

    @staticmethod
    def _normalize_temporal_freq(chunk_duration: str) -> str:
        match = re.fullmatch(r"\s*(\d+)\s*([a-zA-Z]+)\s*", chunk_duration)
        if not match:
            raise ValueError(
                "chunk_duration must be a compact duration like '7D', '1M', or '12H'"
            )

        value = int(match.group(1))
        unit = match.group(2).upper()
        aliases = {
            "D": "D",
            "DAY": "D",
            "DAYS": "D",
            "W": "W",
            "WEEK": "W",
            "WEEKS": "W",
            "M": "M",
            "MON": "M",
            "MONTH": "M",
            "MONTHS": "M",
            "H": "H",
            "HR": "H",
            "HOUR": "H",
            "HOURS": "H",
            "MIN": "T",
            "MINS": "T",
            "MINUTE": "T",
            "MINUTES": "T",
            "T": "T",
            "S": "S",
            "SEC": "S",
            "SECOND": "S",
            "SECONDS": "S",
        }
        if unit not in aliases:
            raise ValueError(f"Unsupported chunk_duration unit '{unit}'")

        return f"{value}{aliases[unit]}"

    def _chunk_by_temporal(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        if self.image_config.chunk_duration is None:
            raise ValueError("chunk_duration is required for temporal chunking")
        if "timestamp" not in df.columns:
            raise ValueError("Temporal chunking requires a timestamp column")

        freq = self._normalize_temporal_freq(self.image_config.chunk_duration)

        out = df.copy()
        out["_timestamp_dt"] = pd.to_datetime(out["timestamp"], errors="coerce")
        if out["_timestamp_dt"].isnull().any():
            raise ValueError("Temporal chunking requires parseable timestamp values")

        out = out.sort_values(["_timestamp_dt", "sample_id"]).reset_index(drop=True)

        try:
            out["_temporal_chunk"] = out["_timestamp_dt"].dt.to_period(freq).astype(str)
            grouped = [group.drop(columns=["_timestamp_dt", "_temporal_chunk"]) for _, group in out.groupby("_temporal_chunk")]
            return grouped
        except Exception:
            delta = pd.to_timedelta(freq)
            anchor = out["_timestamp_dt"].min()
            out["_temporal_chunk"] = ((out["_timestamp_dt"] - anchor) // delta).astype(int)
            grouped = [group.drop(columns=["_timestamp_dt", "_temporal_chunk"]) for _, group in out.groupby("_temporal_chunk")]
            return grouped

    def _build_chunked_analysis_frames(self, analysis_frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if not analysis_frames:
            return {}

        strategy = self.image_config.chunking_strategy
        ordered_keys = self._sorted_numeric_keys(analysis_frames)

        if strategy == "fixed_bucket":
            return {key: analysis_frames[key].copy() for key in ordered_keys}

        combined = []
        for key in ordered_keys:
            frame = analysis_frames[key].copy()
            frame["analysis_source_bucket"] = int(key)
            combined.append(frame)
        combined_df = pd.concat(combined, ignore_index=True)

        if strategy == "quantity":
            if self.image_config.chunk_size_records is None:
                raise ValueError("chunk_size_records is required for quantity chunking")
            ordered_df = self._sort_for_quantity_or_sliding(combined_df)
            chunks = self._chunk_by_quantity(ordered_df, self.image_config.chunk_size_records)
        elif strategy == "sliding_window":
            if self.image_config.chunk_size_records is None:
                raise ValueError("chunk_size_records is required for sliding_window chunking")
            ordered_df = self._sort_for_quantity_or_sliding(combined_df)
            chunks = self._chunk_by_sliding_window(ordered_df, self.image_config.chunk_size_records)
        elif strategy == "temporal":
            chunks = self._chunk_by_temporal(combined_df)
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        start_key = min(int(key) for key in ordered_keys)
        result: Dict[str, pd.DataFrame] = {}
        for idx, chunk in enumerate(chunks):
            chunk_key = str(start_key + idx)
            chunk_out = chunk.copy().reset_index(drop=True)
            chunk_out["analysis_chunk_id"] = idx
            result[chunk_key] = chunk_out

        return result

    def _build_reference_map(self, reference_key: str, analysis_keys: List[str]) -> Dict[str, str]:
        if not analysis_keys:
            return {}

        ordered = sorted(analysis_keys, key=lambda key: int(key))
        if self.image_config.reference_mode == "fixed_reference":
            return {key: reference_key for key in ordered}

        ref_map: Dict[str, str] = {}
        previous_key = reference_key
        for key in ordered:
            ref_map[key] = previous_key
            previous_key = key
        return ref_map

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
        predicted_results: Dict[str, pd.DataFrame] = {}

        for bucket in buckets:
            raw_results[str(bucket)] = self.build_bucket_dataframe(bucket)

        reference_key = str(self.image_config.reference_bucket)
        if not self.image_config.bootstrap_predictions_from_y_true:
            self._fit_baseline_classifier(raw_results[reference_key])

        for bucket_key, frame in raw_results.items():
            with_predictions = self._attach_predictions(frame)
            self._validate_bucket_contract(with_predictions, bucket=int(bucket_key))
            predicted_results[bucket_key] = with_predictions

        reference_key = str(self.image_config.reference_bucket)
        analysis_raw = {
            key: frame
            for key, frame in predicted_results.items()
            if key != reference_key
        }

        chunked_analysis = self._build_chunked_analysis_frames(analysis_raw)

        results: Dict[str, pd.DataFrame] = {reference_key: predicted_results[reference_key]}
        for key in sorted(chunked_analysis.keys(), key=lambda value: int(value)):
            frame = chunked_analysis[key]
            self._validate_bucket_contract(frame, bucket=int(key))
            results[key] = frame

        self.analysis_reference_map = self._build_reference_map(
            reference_key=reference_key,
            analysis_keys=[key for key in results.keys() if key != reference_key],
        )

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

        reference_key = str(self.image_config.reference_bucket)
        reference_dataset_path = target_dir / "reference_dataset.parquet"
        if reference_key in bucket_frames:
            bucket_frames[reference_key].to_parquet(reference_dataset_path, index=False)

        analysis_keys = sorted(
            [key for key in bucket_frames.keys() if key != reference_key],
            key=lambda key: int(key),
        )

        analysis_rows_added = 0
        analysis_dataset_path = target_dir / "analysis_dataset.parquet"
        if analysis_keys:
            appended_rows = []
            for key in analysis_keys:
                chunk_df = bucket_frames[key].copy()
                chunk_df["analysis_chunk_key"] = key
                appended_rows.append(chunk_df)

            analysis_append_df = pd.concat(appended_rows, ignore_index=True)
            analysis_rows_added = len(analysis_append_df)

            if analysis_dataset_path.exists():
                existing = pd.read_parquet(analysis_dataset_path)
                analysis_append_df = pd.concat([existing, analysis_append_df], ignore_index=True)

            analysis_append_df.to_parquet(analysis_dataset_path, index=False)

        metadata = {
            "dataset": self.image_config.dataset,
            "root_path": str(self.root_path),
            "reference_bucket": self.image_config.reference_bucket,
            "analysis_buckets": list(self.image_config.analysis_buckets),
            "analysis_chunk_keys": analysis_keys,
            "analysis_reference_map": self.analysis_reference_map,
            "chunking_strategy": self.image_config.chunking_strategy,
            "chunk_size_records": self.image_config.chunk_size_records,
            "chunk_duration": self.image_config.chunk_duration,
            "reference_mode": self.image_config.reference_mode,
            "reference_dataset_path": str(reference_dataset_path) if reference_key in bucket_frames else None,
            "analysis_dataset_path": str(analysis_dataset_path) if analysis_keys else None,
            "analysis_rows_added": analysis_rows_added,
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
