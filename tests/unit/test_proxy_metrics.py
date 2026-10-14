"""Unit tests for multiclass proxy estimator and CLEAR-10 report helpers."""

from __future__ import annotations

import json

import pandas as pd

from drift_autopsy.data.proxy_metrics import MulticlassProxyEstimator


def _bucket_frame(y_true, y_pred, proba):
    rows = []
    for idx, (yt, yp, prob) in enumerate(zip(y_true, y_pred, proba)):
        class_name = f"class_{int(yt)}"
        rows.append(
            {
                "feature_0": float(idx),
                "feature_1": float(idx + 0.1),
                "sample_id": str(idx),
                "timestamp": f"2020-01-01 00:00:{idx:02d}",
                "y_true": int(yt),
                "y_pred": int(yp),
                "class_name": class_name,
                "source": "camera_a" if idx % 2 == 0 else "camera_b",
                "device": "edge_1" if idx % 3 == 0 else "edge_2",
                "pred_proba_0": float(prob[0]),
                "pred_proba_1": float(prob[1]),
            }
        )
    return pd.DataFrame(rows)


def test_multiclass_proxy_estimator_estimate_values():
    reference = _bucket_frame(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 1],
        proba=[[0.95, 0.05], [0.90, 0.10], [0.10, 0.90], [0.08, 0.92]],
    )
    bucket = _bucket_frame(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        proba=[[0.80, 0.20], [0.25, 0.75], [0.60, 0.40], [0.70, 0.30]],
    )

    estimator = MulticlassProxyEstimator(n_bins=5)
    estimator.fit(reference)
    result = estimator.estimate(bucket)

    for metric_name in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= result.estimated[metric_name] <= 1.0
        assert result.actual[metric_name] is not None
        assert 0.0 <= float(result.actual[metric_name]) <= 1.0


def test_multiclass_proxy_estimator_supports_missing_y_true():
    reference = _bucket_frame(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 1],
        proba=[[0.95, 0.05], [0.90, 0.10], [0.10, 0.90], [0.08, 0.92]],
    )
    bucket = _bucket_frame(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        proba=[[0.80, 0.20], [0.25, 0.75], [0.60, 0.40], [0.70, 0.30]],
    )
    bucket["y_true"] = pd.Series([None, None, None, None], dtype="float")

    estimator = MulticlassProxyEstimator(n_bins=5)
    estimator.fit(reference)
    result = estimator.estimate(bucket)

    assert result.actual["accuracy"] is None
    assert result.proxy_quality_gap == {}
    assert "class_0" in result.class_wise_estimated
    assert result.class_wise_actual == {}
