"""Unit tests for config-driven image tabularization runner."""

from __future__ import annotations

from pathlib import Path
import uuid
import json

import numpy as np
import pandas as pd

from drift_autopsy.config.schema import ImageDataConfig
from drift_autopsy.data.image_tabularizer import ImageTabularizationRunner
from drift_autopsy.registry import ExtractorRegistry


class _DummyExtractor:
    def __init__(self, embedding_dim: int = 4):
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def extract(self, image_paths):
        rows = len(image_paths)
        # Deterministic non-random embeddings for stable tests.
        values = np.arange(rows * self._embedding_dim, dtype=np.float32)
        return values.reshape(rows, self._embedding_dim)

    def get_extraction_metadata(self):
        return {"extractor": "dummy", "embedding_dim": self._embedding_dim}


class _ClassAwareDummyExtractor:
    def __init__(self, embedding_dim: int = 2):
        self._embedding_dim = embedding_dim

    def extract(self, image_paths):
        values = []
        for path in image_paths:
            if "CLASS_A" in path:
                values.append([0.1, 0.2])
            else:
                values.append([2.0, 2.1])
        return np.array(values, dtype=np.float32)

    def get_extraction_metadata(self):
        return {"extractor": "class_aware_dummy", "embedding_dim": self._embedding_dim}


def _create_minimal_clear10_tree(root: Path):
    (root / "labeled_images" / "1" / "BACKGROUND").mkdir(parents=True, exist_ok=True)
    (root / "labeled_images" / "2" / "BACKGROUND").mkdir(parents=True, exist_ok=True)
    (root / "labeled_metadata" / "1").mkdir(parents=True, exist_ok=True)
    (root / "labeled_metadata" / "2").mkdir(parents=True, exist_ok=True)

    # Empty file is sufficient because dummy extractor does not decode image bytes.
    (root / "labeled_images" / "1" / "BACKGROUND" / "sample_a.jpg").touch()
    (root / "labeled_images" / "2" / "BACKGROUND" / "sample_b.jpg").touch()

    (root / "class_names.txt").write_text("BACKGROUND\n")

    index = {
        "1": {"BACKGROUND": "labeled_metadata/1/BACKGROUND.json"},
        "2": {"BACKGROUND": "labeled_metadata/2/BACKGROUND.json"},
    }
    (root / "labeled_metadata.json").write_text(json.dumps(index))

    metadata_one = {
        "sample_a": {
            "DATE_TAKEN": "2020-01-01 00:00:00.0",
            "DEVICE": "cam-a",
            "USER_TAGS": "tag-a",
            "LON": "1.0",
            "LAT": "2.0",
        }
    }
    metadata_two = {
        "sample_b": {
            "DATE_TAKEN": "2020-02-01 00:00:00.0",
            "DEVICE": "cam-b",
            "USER_TAGS": "tag-b",
            "LON": "3.0",
            "LAT": "4.0",
        }
    }
    (root / "labeled_metadata" / "1" / "BACKGROUND.json").write_text(json.dumps(metadata_one))
    (root / "labeled_metadata" / "2" / "BACKGROUND.json").write_text(json.dumps(metadata_two))


def _create_multirow_analysis_clear10_tree(root: Path):
    (root / "labeled_images" / "1" / "BACKGROUND").mkdir(parents=True, exist_ok=True)
    (root / "labeled_images" / "2" / "BACKGROUND").mkdir(parents=True, exist_ok=True)
    (root / "labeled_metadata" / "1").mkdir(parents=True, exist_ok=True)
    (root / "labeled_metadata" / "2").mkdir(parents=True, exist_ok=True)

    (root / "class_names.txt").write_text("BACKGROUND\n")

    index = {
        "1": {"BACKGROUND": "labeled_metadata/1/BACKGROUND.json"},
        "2": {"BACKGROUND": "labeled_metadata/2/BACKGROUND.json"},
    }
    (root / "labeled_metadata.json").write_text(json.dumps(index))

    ref_record = {
        "ref_0": {
            "DATE_TAKEN": "2020-01-01 00:00:00.0",
            "DEVICE": "cam-ref",
            "USER_TAGS": "tag-ref",
            "LON": "1.0",
            "LAT": "2.0",
        }
    }
    analysis_records = {}
    for idx in range(5):
        sample_id = f"analysis_{idx}"
        analysis_records[sample_id] = {
            "DATE_TAKEN": f"2020-02-01 00:00:0{idx}.0",
            "DEVICE": "cam-analysis",
            "USER_TAGS": "tag-analysis",
            "LON": "3.0",
            "LAT": "4.0",
        }
        (root / "labeled_images" / "2" / "BACKGROUND" / f"{sample_id}.jpg").touch()

    (root / "labeled_images" / "1" / "BACKGROUND" / "ref_0.jpg").touch()
    (root / "labeled_metadata" / "1" / "BACKGROUND.json").write_text(json.dumps(ref_record))
    (root / "labeled_metadata" / "2" / "BACKGROUND.json").write_text(json.dumps(analysis_records))


def _create_two_class_clear10_tree(root: Path):
    for bucket in ("1", "2"):
        for class_name in ("CLASS_A", "CLASS_B"):
            (root / "labeled_images" / bucket / class_name).mkdir(parents=True, exist_ok=True)
        (root / "labeled_metadata" / bucket).mkdir(parents=True, exist_ok=True)

    (root / "class_names.txt").write_text("CLASS_A\nCLASS_B\n")

    index = {
        "1": {
            "CLASS_A": "labeled_metadata/1/CLASS_A.json",
            "CLASS_B": "labeled_metadata/1/CLASS_B.json",
        },
        "2": {
            "CLASS_A": "labeled_metadata/2/CLASS_A.json",
            "CLASS_B": "labeled_metadata/2/CLASS_B.json",
        },
    }
    (root / "labeled_metadata.json").write_text(json.dumps(index))

    def make_entries(prefix: str, n: int):
        payload = {}
        for idx in range(n):
            sample_id = f"{prefix}_{idx}"
            payload[sample_id] = {
                "DATE_TAKEN": "2020-01-01 00:00:00.0",
                "DEVICE": "cam",
                "USER_TAGS": "tag",
                "LON": "1.0",
                "LAT": "2.0",
            }
        return payload

    for sample_id in make_entries("a_ref", 4).keys():
        (root / "labeled_images" / "1" / "CLASS_A" / f"{sample_id}.jpg").touch()
    for sample_id in make_entries("b_ref", 4).keys():
        (root / "labeled_images" / "1" / "CLASS_B" / f"{sample_id}.jpg").touch()
    for sample_id in make_entries("a_test", 2).keys():
        (root / "labeled_images" / "2" / "CLASS_A" / f"{sample_id}.jpg").touch()
    for sample_id in make_entries("b_test", 2).keys():
        (root / "labeled_images" / "2" / "CLASS_B" / f"{sample_id}.jpg").touch()

    (root / "labeled_metadata" / "1" / "CLASS_A.json").write_text(json.dumps(make_entries("a_ref", 4)))
    (root / "labeled_metadata" / "1" / "CLASS_B.json").write_text(json.dumps(make_entries("b_ref", 4)))
    (root / "labeled_metadata" / "2" / "CLASS_A.json").write_text(json.dumps(make_entries("a_test", 2)))
    (root / "labeled_metadata" / "2" / "CLASS_B.json").write_text(json.dumps(make_entries("b_test", 2)))


def test_image_tabularization_runner_builds_reference_and_analysis(tmp_path):
    _create_minimal_clear10_tree(tmp_path)

    extractor_name = f"dummy_extractor_{uuid.uuid4().hex}"
    ExtractorRegistry.register(extractor_name)(_DummyExtractor)

    image_cfg = ImageDataConfig(
        root_path=str(tmp_path),
        extractor_name=extractor_name,
        extractor_params={"embedding_dim": 4},
        reference_bucket=1,
        analysis_buckets=[2],
        include_background=True,
        expected_embedding_dim=4,
        bootstrap_predictions_from_y_true=True,
    )

    runner = ImageTabularizationRunner(image_cfg)
    results = runner.build_reference_and_analysis()

    assert set(results.keys()) == {"1", "2"}

    ref = results["1"]
    assert "feature_0" in ref.columns
    assert "feature_3" in ref.columns
    assert "y_pred" in ref.columns
    assert "pred_proba_0" in ref.columns
    assert ref["pred_proba_0"].iloc[0] == 1.0


def test_image_tabularization_runner_persists_artifacts(tmp_path):
    _create_minimal_clear10_tree(tmp_path)

    extractor_name = f"dummy_extractor_{uuid.uuid4().hex}"
    ExtractorRegistry.register(extractor_name)(_DummyExtractor)

    image_cfg = ImageDataConfig(
        root_path=str(tmp_path),
        extractor_name=extractor_name,
        extractor_params={"embedding_dim": 2},
        reference_bucket=1,
        analysis_buckets=[2],
        expected_embedding_dim=2,
    )

    runner = ImageTabularizationRunner(image_cfg)
    frames = runner.build_reference_and_analysis()
    out_dir = runner.persist_artifacts(frames, output_dir=tmp_path / "artifacts_test")

    assert (out_dir / "bucket_1.parquet").exists()
    assert (out_dir / "bucket_2.parquet").exists()
    assert (out_dir / "reference_dataset.parquet").exists()
    assert (out_dir / "analysis_dataset.parquet").exists()
    assert (out_dir / "tabularization_metadata.json").exists()


def test_image_tabularization_runner_trains_baseline_when_bootstrap_disabled(tmp_path):
    _create_two_class_clear10_tree(tmp_path)

    extractor_name = f"class_aware_extractor_{uuid.uuid4().hex}"
    ExtractorRegistry.register(extractor_name)(_ClassAwareDummyExtractor)

    image_cfg = ImageDataConfig(
        root_path=str(tmp_path),
        extractor_name=extractor_name,
        extractor_params={"embedding_dim": 2},
        reference_bucket=1,
        analysis_buckets=[2],
        bootstrap_predictions_from_y_true=False,
        baseline_model_name="logistic_regression",
        baseline_train_fraction=0.7,
        baseline_random_state=42,
        expected_embedding_dim=2,
        expected_class_count=2,
    )

    runner = ImageTabularizationRunner(image_cfg)
    frames = runner.build_reference_and_analysis()

    assert runner.baseline_classifier is not None
    assert runner.baseline_metrics is not None
    assert "accuracy" in runner.baseline_metrics

    test_df = frames["2"]
    assert "y_pred" in test_df.columns
    assert "pred_proba_0" in test_df.columns
    assert "pred_proba_1" in test_df.columns

    out_dir = runner.persist_artifacts(frames, output_dir=tmp_path / "artifacts_with_model")
    assert (out_dir / "baseline_model.pkl").exists()


def test_quantity_chunking_builds_multiple_analysis_chunks_and_previous_reference_map(tmp_path):
    _create_multirow_analysis_clear10_tree(tmp_path)

    extractor_name = f"dummy_extractor_{uuid.uuid4().hex}"
    ExtractorRegistry.register(extractor_name)(_DummyExtractor)

    image_cfg = ImageDataConfig(
        root_path=str(tmp_path),
        extractor_name=extractor_name,
        extractor_params={"embedding_dim": 3},
        reference_bucket=1,
        analysis_buckets=[2],
        bootstrap_predictions_from_y_true=True,
        chunking_strategy="quantity",
        chunk_size_records=2,
        reference_mode="previous_chunk",
        expected_embedding_dim=3,
    )

    runner = ImageTabularizationRunner(image_cfg)
    frames = runner.build_reference_and_analysis()

    assert set(frames.keys()) == {"1", "2", "3", "4"}
    assert runner.analysis_reference_map == {
        "2": "1",
        "3": "2",
        "4": "3",
    }
    assert len(frames["2"]) == 2
    assert len(frames["3"]) == 2
    assert len(frames["4"]) == 1

    out_dir = runner.persist_artifacts(frames, output_dir=tmp_path / "quantity_artifacts")
    assert (out_dir / "analysis_dataset.parquet").exists()

    analysis_df = pd.read_parquet(out_dir / "analysis_dataset.parquet")
    assert "analysis_chunk_key" in analysis_df.columns
    assert len(analysis_df) == 5
