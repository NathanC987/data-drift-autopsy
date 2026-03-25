"""Unit tests for embedding baseline classifier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift_autopsy.data.image_baseline import EmbeddingBaselineClassifier, create_monitored_model


def _make_frame(n_rows: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    labels = np.array([0] * (n_rows // 2) + [1] * (n_rows - n_rows // 2))

    # Make classes separable for deterministic behavior.
    feature_0 = labels.astype(float) * 2.0 + rng.normal(0.0, 0.1, size=n_rows)
    feature_1 = labels.astype(float) * 1.5 + rng.normal(0.0, 0.1, size=n_rows)

    return pd.DataFrame(
        {
            "feature_0": feature_0,
            "feature_1": feature_1,
            "y_true": labels,
        }
    )


def test_baseline_classifier_fit_with_split_and_predict():
    frame = _make_frame()
    clf = EmbeddingBaselineClassifier(model_name="logistic_regression")

    metrics = clf.fit_with_split(frame, train_fraction=0.7, random_state=42)
    assert metrics["accuracy"] >= 0.5
    assert metrics["f1_macro"] >= 0.5

    predicted = clf.attach_predictions(frame, class_count=2)
    assert "y_pred" in predicted.columns
    assert "pred_proba_0" in predicted.columns
    assert "pred_proba_1" in predicted.columns

    row_sums = predicted[["pred_proba_0", "pred_proba_1"]].to_numpy().sum(axis=1)
    assert np.allclose(row_sums, 1.0)


def test_baseline_classifier_save_and_load(tmp_path):
    frame = _make_frame()
    clf = EmbeddingBaselineClassifier(model_name="logistic_regression")
    clf.fit_with_split(frame, train_fraction=0.7, random_state=42)

    model_path = tmp_path / "baseline.pkl"
    clf.save(str(model_path))
    assert model_path.exists()

    loaded = EmbeddingBaselineClassifier.load(str(model_path))
    out = loaded.attach_predictions(frame, class_count=2)
    assert "y_pred" in out.columns


def test_create_monitored_model_logistic_regression():
    model = create_monitored_model("logistic_regression", {"max_iter": 100})
    assert isinstance(model, EmbeddingBaselineClassifier)


def test_create_monitored_model_invalid_name_raises():
    with pytest.raises(ValueError, match="Unsupported monitored model"):
        create_monitored_model("unknown_model", {})


def test_create_monitored_model_resnet_classifier_if_available():
    try:
        model = create_monitored_model(
            "resnet_classifier",
            {
                "model_name": "resnet18",
                "weights": None,
                "epochs": 1,
            },
        )
    except ImportError:
        pytest.skip("torch/torchvision/Pillow not available")

    assert getattr(model, "model_name", None) == "resnet18"