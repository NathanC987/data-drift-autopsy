"""Unit tests for dashboard data loader normalization behavior."""

from __future__ import annotations

import json

import pandas as pd

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


def test_reliability_loader_preserves_unavailable_signals_as_nan(tmp_path):
    payload = {
        "2015": {
            "analysis_type": "temporal",
            "reliability": [
                {
                    "prediction_id": "p1",
                    "confidence": 0.8,
                    "ood": 0.2,
                    "stability": 0.1,
                    "calibration": "good",
                    "calibration_risk": None,
                    "explanation": None,
                    "cbpe_score": None,
                    "risk_score": 0.4,
                    "risk_label": "MEDIUM",
                    "details": {
                        "quality": {
                            "degraded": True,
                            "high_quality": False,
                            "missing_required_signals": ["explanation"],
                        }
                    },
                }
            ],
        }
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loader = DriftResultsLoader(str(path))
    rel_df = loader.get_reliability_results(scope="folktables")

    assert len(rel_df) == 1
    assert pd.isna(rel_df.iloc[0]["explanation"])
    assert "degraded" not in rel_df.columns
    assert "high_quality" not in rel_df.columns
    assert "missing_required_signals" not in rel_df.columns
