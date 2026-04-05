"""Integration tests for CLEAR-10 dashboard loader contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.dashboard.data_loader import DriftResultsLoader


@pytest.mark.integration
def test_clear10_dashboard_loader_contract_from_generated_report():
    report_path = Path("outputs/clear10_drift_results.json")
    if not report_path.exists():
        pytest.skip("Generated CLEAR-10 report not found at outputs/clear10_drift_results.json")

    loader = DriftResultsLoader(str(report_path))

    baseline = loader.get_clear10_baseline_performance()
    proxy_df = loader.get_clear10_proxy_metrics()
    drift_df = loader.get_clear10_drift_timeline()
    localization_df = loader.get_clear10_localization_summary()
    rca_df = loader.get_clear10_rca_summary()
    reliability_df = loader.get_reliability_results(scope="clear10")

    assert isinstance(baseline, dict)
    assert baseline
    assert "accuracy" in baseline

    assert not proxy_df.empty
    assert set(["bucket", "metric", "estimated", "actual"]).issubset(proxy_df.columns)

    assert not drift_df.empty
    assert set(["bucket", "detector", "score", "threshold"]).issubset(drift_df.columns)
    assert drift_df["score"].notna().all()

    assert not localization_df.empty
    assert set(["bucket", "top_features", "n_drifted_features"]).issubset(localization_df.columns)

    assert not rca_df.empty
    assert set(["bucket", "top_changes", "n_recommendations"]).issubset(rca_df.columns)

    assert set(
        [
            "source",
            "analysis_key",
            "prediction_id",
            "confidence",
            "ood",
            "stability",
            "risk_score",
            "risk_label",
        ]
    ).issubset(reliability_df.columns)
