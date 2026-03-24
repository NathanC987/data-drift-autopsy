"""Unit tests for Phase 2 data foundation (CLEAR-10 loader + embedding contract validation)."""

from pathlib import Path

import pandas as pd
import pytest

from drift_autopsy.data import Clear10Loader, DataValidator


@pytest.fixture
def clear10_root() -> Path:
    """Return local CLEAR-10 dataset root for tests."""
    root = Path("data/clear10")
    if not root.exists():
        pytest.skip("CLEAR-10 dataset not available in workspace")
    return root


def test_clear10_list_buckets_and_class_names(clear10_root: Path) -> None:
    """Ensure CLEAR-10 adapter discovers buckets and class names."""
    buckets = Clear10Loader.list_buckets(clear10_root)
    class_names = Clear10Loader.load_class_names(clear10_root)

    assert buckets[0] == 1
    assert buckets[-1] == 10
    assert len(buckets) == 10
    assert len(class_names) >= 10
    assert "BACKGROUND" in class_names


def test_clear10_build_bucket_dataframe(clear10_root: Path) -> None:
    """Ensure bucket dataframe has required normalized columns."""
    df = Clear10Loader.build_bucket_dataframe(
        clear10_root,
        bucket=1,
        max_samples_per_class=3,
    )

    required_cols = {
        "sample_id",
        "bucket",
        "class_name",
        "y_true",
        "image_path",
        "timestamp",
        "source",
    }
    assert required_cols.issubset(df.columns)
    assert (df["bucket"] == 1).all()
    assert (df["source"] == "clear10").all()


def test_validate_embedding_contract_valid() -> None:
    """Validate a minimal valid embedding contract dataframe."""
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "timestamp": ["bucket-02", "bucket-02"],
            "feature_0": [0.1, 0.2],
            "feature_1": [0.3, 0.4],
            "y_pred": [0, 1],
            "pred_proba_0": [0.8, 0.2],
            "pred_proba_1": [0.2, 0.8],
        }
    )

    DataValidator.validate_embedding_contract(
        df,
        expected_embedding_dim=2,
        expected_class_count=2,
    )


def test_validate_embedding_contract_invalid_probability_sum() -> None:
    """Reject rows whose probability vectors do not sum to 1."""
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "timestamp": ["bucket-02", "bucket-02"],
            "feature_0": [0.1, 0.2],
            "feature_1": [0.3, 0.4],
            "y_pred": [0, 1],
            "pred_proba_0": [0.8, 0.7],
            "pred_proba_1": [0.3, 0.4],
        }
    )

    with pytest.raises(ValueError, match="sum to 1"):
        DataValidator.validate_embedding_contract(df)


def test_validate_embedding_contract_allows_missing_y_true() -> None:
    """Allow analysis datasets with delayed labels when configured."""
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "timestamp": ["bucket-02", "bucket-02"],
            "feature_0": [0.1, 0.2],
            "feature_1": [0.3, 0.4],
            "y_pred": [0, 1],
            "pred_proba_0": [0.8, 0.2],
            "pred_proba_1": [0.2, 0.8],
        }
    )

    DataValidator.validate_embedding_contract(
        df,
        allow_missing_y_true=True,
        expected_embedding_dim=2,
        expected_class_count=2,
    )
