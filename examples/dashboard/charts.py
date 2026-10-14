"""Plotly figures for the dashboard (reference categorical palette, fixed order)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MUTED = "#52514e"


def _layout(fig: go.Figure, title: str, y_title: str = "", x_title: str = "", height: int = 340) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="closest",
                      yaxis_title=y_title, xaxis_title=x_title, template="plotly_white")
    return fig


def accuracy_vs_estimate(summary: pd.DataFrame, reference_accuracy: float) -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=reference_accuracy, line_dash="dot", line_color=MUTED,
                  annotation_text="reference accuracy", annotation_position="bottom right")
    fig.add_scatter(x=summary["window"], y=summary["accuracy"], mode="markers+lines", name="measured accuracy",
                    marker=dict(size=9, color=PALETTE[0]), line=dict(width=2, color=PALETTE[0]),
                    hovertemplate="%{x}<br>measured %{y:.4f}<extra></extra>")
    fig.add_scatter(x=summary["window"], y=summary["CBPE estimate"], mode="markers+lines",
                    name="CBPE estimate (label free)",
                    marker=dict(size=9, symbol="square-open", color=PALETTE[1], line=dict(width=2)),
                    line=dict(width=2, dash="dash", color=PALETTE[1]),
                    hovertemplate="%{x}<br>estimated %{y:.4f}<extra></extra>")
    return _layout(fig, "Measured accuracy and the label-free estimate", "accuracy")


def localization_heatmap(loc: pd.DataFrame) -> go.Figure:
    pivot = loc.pivot(index="feature", columns="window", values="effect size")
    drifted = loc.pivot(index="feature", columns="window", values="drifted")
    text = drifted.map(lambda d: "*" if d else "")
    fig = go.Figure(go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index), colorscale="Blues",
                               text=text.values, texttemplate="%{text}", colorbar=dict(title="effect"),
                               hovertemplate="%{y} in %{x}<br>effect size %{z:.3f}<extra></extra>"))
    return _layout(fig, "Per-feature shift (KS statistic or Cramer's V; * = significant after Holm)", height=380)


def interval_bars(df: pd.DataFrame, label_col: str, value_col: str, lo_col: str, hi_col: str,
                  sig_col: str, title: str, x_title: str) -> go.Figure:
    colors = [PALETTE[0] if v > 0 else PALETTE[1] for v in df[value_col]]
    fig = go.Figure(go.Bar(
        y=df[label_col], x=df[value_col], orientation="h",
        marker=dict(color=[c if s else "white" for c, s in zip(colors, df[sig_col])],
                    line=dict(color=colors, width=1.5)),
        error_x=dict(type="data", symmetric=False, array=df[hi_col] - df[value_col],
                     arrayminus=df[value_col] - df[lo_col], color=MUTED, thickness=1),
        hovertemplate="%{y}<br>%{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    return _layout(fig, title, x_title=x_title, height=max(300, 22 * len(df) + 80))


def risk_bands(rel: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for color, band in zip(PALETTE, ("low", "medium", "high")):
        fig.add_bar(x=rel["window"], y=rel[f"error_rate_{band}"], name=f"{band} risk", marker_color=color,
                    hovertemplate="%{x}<br>" + band + " risk: error rate %{y:.3f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    return _layout(fig, "Observed error rate within each risk label", "error rate")


def confusion_heatmap(cm: pd.DataFrame) -> go.Figure:
    frac = cm.div(cm.sum(axis=1), axis=0)
    fig = go.Figure(go.Heatmap(z=frac.values, x=list(cm.columns), y=list(cm.index), colorscale="Blues", zmin=0,
                               zmax=1, text=cm.values, texttemplate="%{text}", showscale=False,
                               hovertemplate="injected %{y}<br>diagnosed %{x}<br>share %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Injected vs. diagnosed drift type (counts)", "injected", "diagnosed", height=380)


def audit_budget(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for color, (t, g) in zip(PALETTE, df.groupby("type", sort=False)):
        g = g.sort_values("labels")
        fig.add_scatter(x=g["labels"], y=g["correct"], mode="lines+markers", name=t,
                        line=dict(color=color, width=2), marker=dict(size=8),
                        error_y=dict(type="data", symmetric=False, array=g["ci high"] - g["correct"],
                                     arrayminus=g["correct"] - g["ci low"], color=color, thickness=1),
                        hovertemplate=t + "<br>%{x} labels: %{y:.2f}<extra></extra>")
    fig.update_yaxes(range=[0, 1.05])
    return _layout(fig, "Correct diagnosis rate by audit size", "correct rate", "labels in the audit")
