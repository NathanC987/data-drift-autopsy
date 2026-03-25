"""Visualization components for drift analysis dashboard."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional


# Color scheme for severity levels
SEVERITY_COLORS = {
    "none": "#90EE90",      # Light green
    "low": "#FFD700",       # Gold
    "medium": "#FFA500",    # Orange
    "high": "#FF6347",      # Tomato
    "critical": "#DC143C",  # Crimson
}


def create_drift_timeline(df: pd.DataFrame, title: Optional[str] = None) -> go.Figure:
    """
    Create time-series line chart of drift scores.
    
    Args:
        df: DataFrame with columns: year, detector, score, drift_detected
        title: Optional custom title
    
    Returns:
        Plotly Figure
    """
    fig = go.Figure()
    
    # Get unique detectors
    detectors = df["detector"].unique() if "detector" in df.columns else [None]
    
    # Add line traces for each detector
    for detector in detectors:
        if detector is None:
            detector_df = df
            name = "Score"
        else:
            detector_df = df[df["detector"] == detector]
            name = detector
        
        # Add main line trace
        fig.add_trace(
            go.Scatter(
                x=detector_df["year"],
                y=detector_df["score"],
                mode="lines+markers",
                name=name,
                hovertemplate=f"{name}<br>Year: %{{x}}<br>Score: %{{y:.6f}}<extra></extra>",
            )
        )
        
        # Add drift detection markers if column exists
        if "drift_detected" in detector_df.columns:
            drift_points = detector_df[detector_df["drift_detected"]]
            if not drift_points.empty:
                # Use legendgroup to group drift markers together
                fig.add_trace(
                    go.Scatter(
                        x=drift_points["year"],
                        y=drift_points["score"],
                        mode="markers",
                        marker=dict(size=12, symbol="x", color="red"),
                        name="Drift Detected",
                        legendgroup="drift",
                        showlegend=(detector == detectors[0]),  # Only show legend for first detector
                        hovertemplate=f"{name}<br>Year: %{{x}}<br>Score: %{{y:.6f}}<br><b>Drift Detected</b><extra></extra>",
                    )
                )
    
    fig.update_layout(
        title=title or "Drift Score Over Time",
        xaxis_title="Year",
        yaxis_title="Drift Score",
        hovermode="x unified",
        height=400,
        template="plotly_white",
    )
    
    return fig


def create_detector_comparison(df: pd.DataFrame, title: Optional[str] = None) -> go.Figure:
    """
    Create bar chart comparing detectors across years.
    
    Args:
        df: DataFrame with columns: year, detector, score, drift_detected
        title: Optional custom title
    
    Returns:
        Plotly Figure
    """
    fig = px.bar(
        df,
        x="year",
        y="score",
        color="detector",
        barmode="group",
        title=title or "Detector Comparison Across Years",
        labels={"score": "Drift Score", "year": "Year"},
        template="plotly_white",
    )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Drift Score",
        legend_title="Detector",
        height=450,
    )
    
    return fig


def create_feature_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create heatmap of feature drift scores over time.
    
    Args:
        df: DataFrame with columns: year, feature, drift_score
    
    Returns:
        Plotly Figure
    """
    # Pivot data for heatmap
    pivot_df = df.pivot_table(
        index="feature",
        columns="year",
        values="drift_score",
        aggfunc="mean"
    )
    
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale="Reds",
            colorbar=dict(title="Drift Score"),
            hoverongaps=False,
        )
    )
    
    fig.update_layout(
        title="Feature Drift Heatmap",
        xaxis_title="Year",
        yaxis_title="Feature",
        height=max(400, len(pivot_df) * 30),  # Dynamic height
        template="plotly_white",
    )
    
    return fig


def create_performance_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create line chart showing model performance over time.
    
    Args:
        df: DataFrame with columns: year, accuracy, accuracy_delta
    
    Returns:
        Plotly Figure  
    """
    fig = go.Figure()
    
    # Accuracy line
    fig.add_trace(
        go.Scatter(
            x=df["year"],
            y=df["accuracy"],
            mode="lines+markers",
            name="Accuracy",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=10),
        )
    )
    
    # Performance degradation area
    if "accuracy_delta" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["year"],
                y=df["accuracy_delta"],
                name="Accuracy Delta",
                marker_color="rgba(255, 0, 0, 0.3)",
                yaxis="y2",
            )
        )
    
    fig.update_layout(
        title="Model Performance Over Time",
        xaxis_title="Year",
        yaxis_title="Accuracy",
        yaxis=dict(tickformat=".1%"),
        yaxis2=dict(
            title="Accuracy Delta",
            overlaying="y",
            side="right",
            tickformat=".2%",
        ),
        template="plotly_white",
        hovermode="x unified",
        height=400,
    )
    
    return fig


def create_severity_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Create pie chart showing distribution of drift severity levels.
    
    Args:
        df: DataFrame with columns: severity, drift_detected
    
    Returns:
        Plotly Figure
    """
    severity_counts = df["severity"].value_counts().reset_index()
    severity_counts.columns = ["severity", "count"]
    
    # Map severity to colors
    colors = [SEVERITY_COLORS.get(sev, "#808080") for sev in severity_counts["severity"]]
    
    fig = go.Figure(
        data=[
            go.Pie(
                labels=severity_counts["severity"],
                values=severity_counts["count"],
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            )
        ]
    )
    
    fig.update_layout(
        title="Drift Severity Distribution",
        height=350,
        showlegend=True,
    )
    
    return fig


def create_drift_detection_timeline(df: pd.DataFrame) -> go.Figure:
    """
    Create timeline showing when drift was detected by each detector.
    
    Args:
        df: DataFrame with columns: year, detector, drift_detected
    
    Returns:
        Plotly Figure
    """
    # Filter only drift detections
    drift_df = df[df["drift_detected"]].copy()
    
    # Create scatter plot
    fig = px.scatter(
        drift_df,
        x="year",
        y="detector",
        color="detector",
        size_max=20,
        title="Drift Detection Timeline",
        labels={"year": "Year", "detector": "Detector"},
        template="plotly_white",
    )
    
    fig.update_traces(marker=dict(size=15, symbol="diamond"))
    
    fig.update_layout(
        height=350,
        showlegend=False,
        xaxis=dict(dtick=1),  # Show every year
    )
    
    return fig


def create_top_drifted_features(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Create bar chart of top drifted features.
    
    Args:
        df: DataFrame with columns: feature, drift_score, drift_detected
        top_n: Number of top features to show
    
    Returns:
        Plotly Figure
    """
    # Filter drifted features and get average score
    drifted = df[df["drift_detected"]].groupby("feature")["drift_score"].mean().reset_index()
    drifted = drifted.sort_values("drift_score", ascending=False).head(top_n)
    
    fig = px.bar(
        drifted,
        x="drift_score",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Drifted Features (Avg Score)",
        labels={"drift_score": "Average Drift Score", "feature": "Feature"},
        template="plotly_white",
        color="drift_score",
        color_continuous_scale="Reds",
    )
    
    fig.update_layout(
        height=max(300, top_n * 40),
        showlegend=False,
    )
    
    return fig


def create_feature_importance_comparison(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Create side-by-side bar chart comparing feature importances.
    
    Args:
        df: DataFrame with columns: year, feature, ref_importance, test_importance, change
        top_n: Number of top features to show (by absolute change)
    
    Returns:
        Plotly Figure
    """
    if df.empty:
        return go.Figure()
    
    # Get most recent year
    latest_year = df["year"].max()
    latest_df = df[df["year"] == latest_year]
    
    # Get top N by absolute change
    top_changes = latest_df.nlargest(top_n, "abs_change")
    
    fig = go.Figure()
    
    # Reference importances
    fig.add_trace(go.Bar(
        y=top_changes["feature"],
        x=top_changes["ref_importance"],
        name=f"Reference (2014)",
        orientation="h",
        marker_color="lightblue",
    ))
    
    # Test importances
    fig.add_trace(go.Bar(
        y=top_changes["feature"],
        x=top_changes["test_importance"],
        name=f"Test ({latest_year})",
        orientation="h",
        marker_color="coral",
    ))
    
    fig.update_layout(
        title=f"Feature Importance Changes ({latest_year})",
        xaxis_title="SHAP Importance",
        yaxis_title="Feature",
        barmode="group",
        template="plotly_white",
        height=max(400, top_n * 40),
    )
    
    return fig


def create_importance_change_timeline(df: pd.DataFrame, top_features: int = 5) -> go.Figure:
    """
    Create timeline showing how feature importances changed over years.
    
    Args:
        df: DataFrame with columns: year, feature, change
        top_features: Number of top features to track
    
    Returns:
        Plotly Figure
    """
    if df.empty:
        return go.Figure()
    
    # Get features with largest average absolute change
    avg_change = df.groupby("feature")["abs_change"].mean().nlargest(top_features)
    top_feature_names = avg_change.index.tolist()
    
    # Filter to top features
    filtered_df = df[df["feature"].isin(top_feature_names)]
    
    fig = px.line(
        filtered_df,
        x="year",
        y="change",
        color="feature",
        markers=True,
        title=f"Feature Importance Changes Over Time (Top {top_features} Features)",
        labels={"change": "Importance Change (Test - Reference)", "year": "Year"},
        template="plotly_white",
    )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="No Change")
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Importance Change",
        hovermode="x unified",
        height=400,
    )
    
    return fig


def create_rca_recommendations_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create formatted DataFrame of RCA recommendations.
    
    Args:
        df: DataFrame with columns: year, detector, recommendations
    
    Returns:
        Formatted DataFrame for display
    """
    if df.empty:
        return pd.DataFrame(columns=["Year", "Detector", "Recommendations"])
    
    # Expand recommendations
    rows = []
    for _, row in df.iterrows():
        recommendations = row["recommendations"]
        if recommendations:
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                rows.append({
                    "Year": row["year"],
                    "Detector": row["detector"].replace("_", " ").title(),
                    "Recommendation": rec,
                })
    
    result_df = pd.DataFrame(rows)
    return result_df


def create_feature_importance_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create heatmap of feature importance changes over time.
    
    Args:
        df: DataFrame with columns: year, feature, change
    
    Returns:
        Plotly Figure
    """
    if df.empty:
        return go.Figure()
    
    # Pivot data
    pivot_df = df.pivot_table(
        index="feature",
        columns="year",
        values="change",
        aggfunc="mean"
    )
    
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale="RdBu_r",
            zmid=0,  # Center colorscale at 0
            colorbar=dict(title="Importance<br>Change"),
            hoverongaps=False,
            text=np.round(pivot_df.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    
    fig.update_layout(
        title="Feature Importance Changes Heatmap",
        xaxis_title="Year",
        yaxis_title="Feature",
        height=max(400, len(pivot_df) * 30),
        template="plotly_white",
    )
    
    return fig


def create_slice_drift_heatmap(df: pd.DataFrame, title: Optional[str] = None) -> go.Figure:
    """
    Create heatmap of slice-level drift scores by detector.

    Args:
        df: DataFrame with columns: detector, score, and either slice_key_label or slice_key
        title: Optional custom title

    Returns:
        Plotly Figure
    """
    if df.empty:
        return go.Figure()

    slice_axis_col = "slice_key_label" if "slice_key_label" in df.columns else "slice_key"
    pivot_df = df.pivot_table(
        index=slice_axis_col,
        columns="detector",
        values="score",
        aggfunc="mean",
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale="YlOrRd",
            colorbar=dict(title="Drift Score"),
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title=title or "Slice-Level Drift Scores",
        xaxis_title="Detector",
        yaxis_title="Slice",
        height=max(350, len(pivot_df) * 28),
        template="plotly_white",
    )

    return fig


def create_proxy_metric_step_chart(
    df: pd.DataFrame,
    metric_name: str,
    lower_threshold: float,
    upper_threshold: float,
) -> go.Figure:
    """
    Create stepped chart for estimated vs actual proxy metric with alert markers.

    Args:
        df: DataFrame with columns: bucket, estimated, actual
        metric_name: Metric label for title
        lower_threshold: Lower alert threshold
        upper_threshold: Upper alert threshold

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(
            title=f"{metric_name.title()} (No Data)",
            template="plotly_white",
            height=320,
        )
        return fig

    plot_df = df.sort_values("bucket")

    fig.add_trace(
        go.Scatter(
            x=plot_df["bucket"],
            y=plot_df["estimated"],
            mode="lines+markers",
            line=dict(shape="hvh", width=2, color="#1f77b4"),
            marker=dict(size=8),
            name="Estimated",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["bucket"],
            y=plot_df["actual"],
            mode="lines+markers",
            line=dict(shape="hvh", width=2, color="#2ca02c"),
            marker=dict(size=8),
            name="Actual",
        )
    )

    est_alerts = plot_df[
        (plot_df["estimated"] < lower_threshold) | (plot_df["estimated"] > upper_threshold)
    ]
    if not est_alerts.empty:
        fig.add_trace(
            go.Scatter(
                x=est_alerts["bucket"],
                y=est_alerts["estimated"],
                mode="markers",
                marker=dict(color="red", size=10, symbol="x"),
                name="Estimated Alerts",
            )
        )

    actual_alerts = plot_df[
        (plot_df["actual"] < lower_threshold) | (plot_df["actual"] > upper_threshold)
    ]
    if not actual_alerts.empty:
        fig.add_trace(
            go.Scatter(
                x=actual_alerts["bucket"],
                y=actual_alerts["actual"],
                mode="markers",
                marker=dict(color="red", size=10, symbol="diamond"),
                name="Actual Alerts",
            )
        )

    fig.add_hline(y=lower_threshold, line_color="red", line_dash="dot")
    fig.add_hline(y=upper_threshold, line_color="red", line_dash="dot")

    fig.update_layout(
        title=f"{metric_name.title()} (Estimated vs Actual)",
        xaxis_title="Bucket",
        yaxis_title=metric_name.title(),
        template="plotly_white",
        height=320,
        hovermode="x unified",
    )
    fig.update_xaxes(dtick=1)

    return fig


def create_detector_step_chart(
    df: pd.DataFrame,
    detector_name: str,
    threshold: Optional[float] = None,
    alert_direction: str = "above",
) -> go.Figure:
    """
    Create stepped detector score trend chart with optional threshold overlay.

    Args:
        df: DataFrame with columns: bucket, score
        detector_name: Detector label
        threshold: Optional threshold value for alerting
        alert_direction: Alert rule direction, "above" or "below"

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    if df.empty:
        fig.update_layout(
            title=f"{detector_name} (No Data)",
            template="plotly_white",
            height=320,
        )
        return fig

    plot_df = df.sort_values("bucket")

    fig.add_trace(
        go.Scatter(
            x=plot_df["bucket"],
            y=plot_df["score"],
            mode="lines+markers",
            line=dict(shape="hvh", width=2, color="#ff7f0e"),
            marker=dict(size=8),
            name=detector_name,
        )
    )

    if threshold is not None:
        fig.add_hline(y=threshold, line_color="red", line_dash="dot")
        if alert_direction == "below":
            alerts = plot_df[plot_df["score"] <= threshold]
        else:
            alerts = plot_df[plot_df["score"] >= threshold]
        if not alerts.empty:
            fig.add_trace(
                go.Scatter(
                    x=alerts["bucket"],
                    y=alerts["score"],
                    mode="markers",
                    marker=dict(color="red", size=10, symbol="x"),
                    name="Alerts",
                )
            )

    fig.update_layout(
        title=f"{detector_name} Trend",
        xaxis_title="Bucket",
        yaxis_title="Score",
        template="plotly_white",
        height=320,
        hovermode="x unified",
    )
    fig.update_xaxes(dtick=1)

    return fig


def create_pca_bucket_3d_scatter(df: pd.DataFrame) -> go.Figure:
    """Create 3D scatter plot for PCA projection colored by bucket."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="PCA 3D Projection (No Data)",
            template="plotly_white",
            height=420,
        )
        return fig

    plot_df = df.copy()
    plot_df["bucket"] = plot_df["bucket"].astype(str)

    fig = px.scatter_3d(
        plot_df,
        x="pc1",
        y="pc2",
        z="pc3",
        color="bucket",
        title="PCA 3D Projection (PC1, PC2, PC3)",
        labels={
            "pc1": "PC1",
            "pc2": "PC2",
            "pc3": "PC3",
            "bucket": "Bucket",
        },
        template="plotly_white",
    )

    fig.update_traces(marker=dict(size=4, opacity=0.75))
    fig.update_layout(
        height=420,
        legend_title="Bucket",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
    )
    return fig
