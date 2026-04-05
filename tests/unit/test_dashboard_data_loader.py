"""Unit tests for dashboard data loader normalization behavior."""

from __future__ import annotations

import json

from examples.dashboard.data_loader import DriftResultsLoader


def test_clear10_drift_timeline_handles_null_scores(tmp_path):
    payload = {
        "drift_results": [
            {
                "bucket": 2,
                "detector": "Cbpe",
                "score": None,
                "threshold": 0.05,
            }
        ]
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loader = DriftResultsLoader(str(path))
    drift_df = loader.get_clear10_drift_timeline()

    assert len(drift_df) == 1
    assert float(drift_df.iloc[0]["score"]) == 0.0


def test_clear10_proxy_metrics_handles_null_values(tmp_path):
    payload = {
        "proxy_metrics": [
            {
                "bucket": 2,
                "metric": "accuracy",
                "estimated": None,
                "actual": 0.9,
            }
        ]
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loader = DriftResultsLoader(str(path))
    proxy_df = loader.get_clear10_proxy_metrics()

    assert len(proxy_df) == 1
    assert float(proxy_df.iloc[0]["estimated"]) == 0.0
    assert float(proxy_df.iloc[0]["actual"]) == 0.9


def test_clear10_classwise_proxy_metrics_loads_rows(tmp_path):
    payload = {
        "proxy_metrics_classwise": [
            {
                "bucket": 2,
                "class_id": 1,
                "class_name": "car",
                "metric": "precision",
                "estimated": 0.81,
                "actual": None,
                "gap": None,
            }
        ]
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loader = DriftResultsLoader(str(path))
    classwise_df = loader.get_clear10_proxy_metrics_classwise()

    assert len(classwise_df) == 1
    assert int(classwise_df.iloc[0]["class_id"]) == 1
    assert classwise_df.iloc[0]["class_name"] == "car"
    assert float(classwise_df.iloc[0]["actual"]) == 0.0
