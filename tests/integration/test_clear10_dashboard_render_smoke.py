"""Smoke tests for CLEAR-10 dashboard rendering data path."""

from __future__ import annotations

from pathlib import Path

import pytest
import plotly.graph_objects as go

from examples.dashboard.data_loader import DriftResultsLoader
from examples.dashboard import visualizations as viz


@pytest.mark.integration
def test_clear10_render_path_builds_figures_from_generated_report():
    report_path = Path("outputs/clear10_drift_results.json")
    if not report_path.exists():
        pytest.skip("Generated CLEAR-10 report not found at outputs/clear10_drift_results.json")

    loader = DriftResultsLoader(str(report_path))

    proxy_df = loader.get_clear10_proxy_metrics()
    drift_df = loader.get_clear10_drift_timeline()
    localization_df = loader.get_clear10_localization_summary()
    rca_df = loader.get_clear10_rca_summary()
    reliability_df = loader.get_reliability_results(scope="clear10")

    assert not proxy_df.empty
    assert not drift_df.empty
    assert not localization_df.empty
    assert not rca_df.empty

    # Build proxy charts for each metric used by the CLEAR-10 app tab.
    for metric_name in sorted(proxy_df["metric"].unique().tolist()):
        metric_df = proxy_df[proxy_df["metric"] == metric_name]
        fig = viz.create_proxy_metric_step_chart(
            metric_df,
            metric_name=metric_name,
            lower_threshold=0.7,
            upper_threshold=1.0,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Estimated + Actual traces

    # Build detector trend charts for each detector in drift timeline.
    for detector_name in sorted(drift_df["detector"].unique().tolist()):
        detector_df = drift_df[drift_df["detector"] == detector_name]
        fig = viz.create_detector_step_chart(
            detector_df,
            detector_name=detector_name,
            threshold=0.1,
            alert_direction="above",
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    risk_fig = viz.create_reliability_risk_distribution(reliability_df)
    signal_fig = viz.create_reliability_signal_profile(reliability_df)
    gauge_fig = viz.create_reliability_risk_gauge(
        avg_risk_score=float(reliability_df["risk_score"].mean()) if not reliability_df.empty else 0.0,
        high_risk_ratio=float((reliability_df["risk_label"] == "HIGH").mean()) if not reliability_df.empty else 0.0,
    )
    availability_fig = viz.create_reliability_signal_availability(reliability_df)
    calibration_fig = viz.create_reliability_calibration_breakdown(reliability_df)
    assert isinstance(risk_fig, go.Figure)
    assert isinstance(signal_fig, go.Figure)
    assert isinstance(gauge_fig, go.Figure)
    assert isinstance(availability_fig, go.Figure)
    assert isinstance(calibration_fig, go.Figure)
