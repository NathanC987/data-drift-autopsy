"""Plotly figures for the dashboard.

Colour rules (reference palette, fixed order): categorical series take the
palette slots in order; magnitudes use one sequential blue ramp; gains and
losses use a blue <-> red diverging scale with a gray midpoint; filled vs
hollow markers carry yes / no so meaning never rests on colour alone. Charts
use the Streamlit theme (no template), so they follow light and dark mode.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#8c8b86"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_SCALE = [[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)]
DIVERGING = [[0.0, "#b3262a"], [0.25, "#e66767"], [0.5, "#d9d8d4"], [0.75, "#6da7ec"], [1.0, "#184f95"]]
ORDINAL3 = ["#86b6ef", "#2a78d6", "#104281"]          # low / medium / high
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
TYPE_COLOR = {"no drift": NEUTRAL, "covariate": PALETTE[0], "prior": PALETTE[1], "concept": PALETTE[2],
              "label noise": PALETTE[3]}


def _layout(fig: go.Figure, title: str = "", y_title: str = "", x_title: str = "", height: int = 360,
            legend: bool = True) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, xanchor="left", y=1, yref="container", yanchor="top",
                                 pad=dict(t=8), font=dict(size=15)), height=height,
                      margin=dict(l=10, r=10, t=70 if title else 30, b=10), hovermode="closest",
                      showlegend=legend, legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None),
                      font=dict(size=13))
    if y_title:
        fig.update_yaxes(title_text=y_title)
    if x_title:
        fig.update_xaxes(title_text=x_title)
    return fig


def _order(df: pd.DataFrame) -> List[str]:
    return list(dict.fromkeys(df["window"]))


def _group_panels(df: pd.DataFrame, y_title: str, title: str, height: int = 360, shared_y: bool = True):
    """Temporal windows on the left (joined by lines), geographic on the right (not joined)."""
    groups = [g for g in ("temporal", "geographic") if g in set(df["group"])]
    if len(groups) == 1:
        fig = go.Figure()
        return fig, {groups[0]: (None, None)}, groups
    n_t = df.loc[df["group"] == "temporal", "window"].nunique()
    n_g = df.loc[df["group"] == "geographic", "window"].nunique()
    fig = make_subplots(rows=1, cols=2, shared_yaxes=shared_y, horizontal_spacing=0.04,
                        column_widths=[n_t / (n_t + n_g), n_g / (n_t + n_g)],
                        subplot_titles=("over time (California)", "other states, 2014"))
    fig.update_annotations(font_size=12)
    return fig, {"temporal": (1, 1), "geographic": (1, 2)}, groups


def _segments(d: pd.DataFrame) -> List[pd.DataFrame]:
    """Split a temporal series where a period is missing (``gap_before``), so no line crosses the gap."""
    if "gap_before" not in d or not d["gap_before"].any():
        return [d]
    cut = np.cumsum(d["gap_before"].to_numpy())
    return [d[cut == k] for k in np.unique(cut)]


def _add(fig, trace, pos):
    r, c = pos
    if r is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=r, col=c)


# --------------------------------------------------------------------------- #
# 1. setup and results

def split_bar(split: pd.DataFrame, total_label: str) -> go.Figure:
    fig = go.Figure()
    for color, (_, r) in zip(PALETTE, split.iterrows()):
        fig.add_bar(y=[total_label], x=[r["rows"]], orientation="h", name=f"{r['part']} ({r['share']:.0%})",
                    marker=dict(color=color, line=dict(width=2, color="rgba(0,0,0,0)")),
                    text=f"{r['part']}  {r['share']:.0%}  ·  {r['rows']:,}", textposition="inside",
                    insidetextanchor="middle", hovertemplate=f"<b>{r['part']}</b> {r['rows']:,} rows<br>{r['role']}<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return _layout(fig, "", height=130, legend=False)


def window_sizes(summary: pd.DataFrame, unit: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=summary["window"], y=summary["n"], marker_color=PALETTE[0],
                           text=[f"{n:,}" for n in summary["n"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>%{y:,} " + unit + "<extra></extra>"))
    fig.update_yaxes(showgrid=False, showticklabels=False, range=[0, summary["n"].max() * 1.18])
    return _layout(fig, f"{unit.capitalize()} per window", height=300, legend=False)


def accuracy_with_ci(summary: pd.DataFrame, reference_accuracy: float) -> go.Figure:
    fig, pos, groups = _group_panels(summary, "accuracy", "")
    for g in groups:
        for d in _segments(summary[summary["group"] == g]):
            filled = d["drop significant"]
            _add(fig, go.Scatter(
                x=d["window"], y=d["accuracy"], mode="lines+markers" if g == "temporal" else "markers",
                line=dict(color=PALETTE[0], width=2),
                marker=dict(size=11, color=[PALETTE[0] if f else "rgba(0,0,0,0)" for f in filled],
                            line=dict(color=PALETTE[0], width=2)),
                error_y=dict(type="data", symmetric=False, array=d["acc_high"] - d["accuracy"],
                             arrayminus=d["accuracy"] - d["acc_low"], color=PALETTE[0], thickness=1.2, width=4),
                customdata=np.c_[d["acc_low"], d["acc_high"], d["accuracy change"], d["drop p-value"], d["n"]],
                hovertemplate=("<b>%{x}</b><br>accuracy %{y:.4f}<br>95% CI %{customdata[0]:.4f} to %{customdata[1]:.4f}"
                               "<br>change %{customdata[2]:+.4f} (one-sided p %{customdata[3]:.2g})"
                               "<br>%{customdata[4]:,} rows<extra></extra>"),
                showlegend=False), pos[g])
    fig.add_hline(y=reference_accuracy, line_dash="dot", line_color=NEUTRAL,
                  annotation_text=f"reference {reference_accuracy:.4f}", annotation_position="bottom left")
    fig.update_yaxes(title_text="measured accuracy", row=1, col=1) if len(groups) > 1 else fig.update_yaxes(
        title_text="measured accuracy")
    return _layout(fig, "Measured accuracy with 95% Wilson intervals (filled = significant drop)", height=380,
                   legend=False)


# --------------------------------------------------------------------------- #
# 2a. proxy performance

def estimate_vs_measured(summary: pd.DataFrame, reference_accuracy: float) -> go.Figure:
    fig, pos, groups = _group_panels(summary, "accuracy", "")
    for g in groups:
        mode = "lines+markers" if g == "temporal" else "markers"
        for i, d in enumerate(_segments(summary[summary["group"] == g])):
            first = g == "temporal" and i == 0
            _add(fig, go.Scatter(x=d["window"], y=d["accuracy"], mode=mode, name="measured (needs labels)",
                                 line=dict(color=PALETTE[0], width=2), marker=dict(size=10, color=PALETTE[0]),
                                 legendgroup="m", showlegend=first,
                                 hovertemplate="%{x}<br>measured %{y:.4f}<extra></extra>"), pos[g])
            _add(fig, go.Scatter(x=d["window"], y=d["CBPE estimate"], mode=mode, name="CBPE estimate (label free)",
                                 line=dict(color=PALETTE[1], width=2, dash="dash"),
                                 marker=dict(size=10, symbol="square-open", color=PALETTE[1], line=dict(width=2)),
                                 legendgroup="e", showlegend=first,
                                 hovertemplate="%{x}<br>estimated %{y:.4f}<extra></extra>"), pos[g])
    fig.add_hline(y=reference_accuracy, line_dash="dot", line_color=NEUTRAL,
                  annotation_text="reference", annotation_position="bottom left")
    _layout(fig, "Label-free estimate against measured accuracy", height=400)
    return fig.update_layout(legend=dict(y=1.13), margin=dict(t=100))


def estimate_error(summary: pd.DataFrame) -> go.Figure:
    err = summary["CBPE error"] * 100
    fig = go.Figure(go.Bar(x=summary["window"], y=err, marker_color=PALETTE[1],
                           text=[f"{e:+.1f}" for e in err], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>estimate minus measured %{y:+.2f} pp<extra></extra>"))
    fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    return _layout(fig, "Estimate error (percentage points; above 0 = optimistic)", "pp", height=300, legend=False)


def confidence_hist(hist: pd.DataFrame, window: str) -> go.Figure:
    fig = go.Figure()
    for pop, color, name in (("reference", NEUTRAL, "reference window"), ("window", PALETTE[0], window)):
        d = hist[hist["population"] == pop]
        fig.add_bar(x=d["bin"], y=d["share"], name=name, marker_color=color,
                    hovertemplate="confidence %{x}<br>%{y:.1%} of predictions<extra>" + name + "</extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, f"Model confidence: reference vs {window}", "share of predictions", "top-class confidence",
                   height=340)


def classwise_f1(df: pd.DataFrame, window: str) -> go.Figure:
    fig = go.Figure()
    for _, r in df.iterrows():
        fig.add_trace(go.Scatter(x=[r["actual F1"], r["estimated F1"]], y=[r["class"]] * 2, mode="lines",
                                 line=dict(color=NEUTRAL, width=1.5), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["actual F1"], y=df["class"], mode="markers", name="actual F1 (needs labels)",
                             marker=dict(size=10, color=PALETTE[0]), hovertemplate="%{y}<br>actual %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["estimated F1"], y=df["class"], mode="markers", name="estimated F1 (label free)",
                             marker=dict(size=10, symbol="square-open", color=PALETTE[1], line=dict(width=2)),
                             hovertemplate="%{y}<br>estimated %{x:.3f}<extra></extra>"))
    return _layout(fig, f"Per-class F1, {window}: estimate vs actual", x_title="F1", height=40 + 30 * len(df))


# --------------------------------------------------------------------------- #
# 2b. detection

def detector_small_multiples(det: pd.DataFrame, statistic_names: Dict[str, tuple], psi_threshold: float,
                             ncols: int = 2) -> go.Figure:
    detectors = list(dict.fromkeys(det["detector"]))
    nrows = int(np.ceil(len(detectors) / ncols))
    titles = [f"<b>{d}</b>  ·  {statistic_names.get(d, ('score', ''))[0]}" for d in detectors]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles, vertical_spacing=0.16 if nrows > 1 else 0.1,
                        horizontal_spacing=0.08)
    fig.update_annotations(font_size=12)
    order = _order(det)
    for i, name in enumerate(detectors):
        r, c = i // ncols + 1, i % ncols + 1
        d = det[det["detector"] == name]
        for g in ("temporal", "geographic"):
            for dd in _segments(d[d["group"] == g]):
                if dd.empty:
                    continue
                _small_multiple_trace(fig, dd, g, name, r, c)
        if name == "PSI":
            fig.add_hline(y=psi_threshold, line_dash="dot", line_color=NEUTRAL, row=r, col=c,
                          annotation_text=f"{psi_threshold}", annotation_position="top left")
        fig.update_xaxes(categoryorder="array", categoryarray=order, tickangle=-35, row=r, col=c)
    # legend proxies for filled / hollow
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="alarm",
                             marker=dict(size=10, color=PALETTE[0], line=dict(color=PALETTE[0], width=2))))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="no alarm",
                             marker=dict(size=10, color="rgba(0,0,0,0)", line=dict(color=PALETTE[0], width=2))))
    return _layout(fig, "", height=290 * nrows + 40)


def _small_multiple_trace(fig, dd: pd.DataFrame, g: str, name: str, r: int, c: int) -> None:
    fig.add_trace(go.Scatter(
        x=dd["window"], y=dd["score"], mode="lines+markers" if g == "temporal" else "markers",
        line=dict(color=PALETTE[0], width=2, shape="hv"),
        marker=dict(size=10, color=[PALETTE[0] if a else "rgba(0,0,0,0)" for a in dd["drift"]],
                    line=dict(color=PALETTE[0], width=2)),
        customdata=np.c_[dd["p-value"].astype(float).fillna(np.nan), dd["drift"].map({True: "alarm", False: "no alarm"})],
        hovertemplate="<b>%{x}</b><br>score %{y:.4g}<br>p %{customdata[0]:.3g}<br>%{customdata[1]}<extra>" + name + "</extra>",
        showlegend=False), row=r, col=c)


def alarm_matrix(det: pd.DataFrame) -> go.Figure:
    order = _order(det)
    detectors = list(dict.fromkeys(det["detector"]))[::-1]
    fig = go.Figure()
    for alarm, name in ((True, "alarm"), (False, "no alarm")):
        d = det[det["drift"] == alarm]
        fig.add_trace(go.Scatter(
            x=d["window"], y=d["detector"], mode="markers", name=name,
            marker=dict(size=18, symbol="circle", color=PALETTE[1] if alarm else "rgba(0,0,0,0)",
                        line=dict(color=PALETTE[1] if alarm else NEUTRAL, width=2)),
            customdata=d["p-value"].astype(float), hovertemplate="%{y} on %{x}: " + name + "<br>p %{customdata:.3g}<extra></extra>"))
    fig.update_xaxes(categoryorder="array", categoryarray=order, showgrid=False, tickangle=-35)
    fig.update_yaxes(categoryorder="array", categoryarray=detectors, showgrid=False)
    return _layout(fig, "Which detector raised an alarm, per window", height=130 + 48 * len(detectors))


# --------------------------------------------------------------------------- #
# 2c. localization

def feature_heatmaps(loc: pd.DataFrame) -> go.Figure:
    order = _order(loc)
    parts = [("numeric", "KS statistic D (numeric features)"), ("categorical", "Cramér's V (categorical features)")]
    parts = [(t, s) for t, s in parts if (loc["test"] == t).any()]
    sizes = [loc.loc[loc["test"] == t, "feature"].nunique() for t, _ in parts]
    fig = make_subplots(rows=len(parts), cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_heights=[s / sum(sizes) for s in sizes], subplot_titles=[s for _, s in parts])
    fig.update_annotations(font_size=12)
    for i, (t, _) in enumerate(parts):
        d = loc[loc["test"] == t]
        z = d.pivot(index="label", columns="window", values="effect size").reindex(columns=order)
        sig = d.pivot(index="label", columns="window", values="drifted").reindex(columns=order)
        text = [[f"{v:.2f}{'*' if s else ''}" for v, s in zip(zr, sr)] for zr, sr in zip(z.values, sig.values)]
        fig.add_trace(go.Heatmap(z=z.values, x=order, y=list(z.index), text=text, texttemplate="%{text}",
                                 colorscale=SEQ_SCALE, zmin=0, coloraxis=None, showscale=True,
                                 colorbar=dict(len=0.42, y=1 - (i + 0.5) / len(parts), thickness=12,
                                               title=dict(text="D" if t == "numeric" else "V", side="right")),
                                 hovertemplate="%{y} in %{x}<br>effect %{z:.3f}<extra></extra>"), row=i + 1, col=1)
    fig.update_xaxes(tickangle=-35)
    return _layout(fig, "", height=110 + 34 * sum(sizes), legend=False)


def unseen_categories(loc: pd.DataFrame, min_share: float = 0.01) -> Optional[go.Figure]:
    d = loc.dropna(subset=["unseen category share"])
    keep = d.groupby("feature")["unseen category share"].transform("max") >= min_share
    d = d[keep]
    if d.empty:
        return None
    order = _order(loc)
    fig = go.Figure()
    for color, (feat, g) in zip(PALETTE, d.groupby("label", sort=False)):
        g = g.set_index("window").reindex(order).reset_index()
        fig.add_bar(x=g["window"], y=g["unseen category share"], name=feat, marker_color=color,
                    text=[f"{v:.1%}" if v >= min_share else "" for v in g["unseen category share"].fillna(0)],
                    textposition="outside", cliponaxis=False,
                    hovertemplate="%{x}<br>%{y:.1%} of rows carry a code never seen in the reference<extra>" + feat + "</extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, "Rows whose category never appears in the reference", "share of rows", height=320)


def clear_dims(loc: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=loc["window"], y=loc["drifted dimensions"], marker_color=PALETTE[0],
                           text=loc["drifted dimensions"], textposition="outside", cliponaxis=False,
                           hovertemplate="%{x}<br>%{y} of 512 dimensions shift (Holm)<extra></extra>"))
    return _layout(fig, "Embedding dimensions that shift (all images)", "dimensions", height=300, legend=False)


def class_slice_heatmap(sl: pd.DataFrame) -> go.Figure:
    order = _order(sl)
    z = sl.pivot(index="class", columns="window", values="drifted dimensions").reindex(columns=order)
    fig = go.Figure(go.Heatmap(z=z.values, x=order, y=list(z.index), colorscale=SEQ_SCALE, zmin=0,
                               text=z.values, texttemplate="%{text}", colorbar=dict(thickness=12, title="dims"),
                               hovertemplate="predicted %{y} in %{x}<br>%{z} dimensions shift<extra></extra>"))
    return _layout(fig, "Shifted dimensions within each predicted class", height=80 + 30 * len(z), legend=False)


def sensor_heatmap(sl: pd.DataFrame) -> go.Figure:
    order = _order(sl)
    z = sl.pivot(index="sensor", columns="window", values="drifted features").reindex(columns=order)
    n_feat = int(sl["of"].max())
    fig = go.Figure(go.Heatmap(z=z.values, x=order, y=list(z.index), colorscale=SEQ_SCALE, zmin=0, zmax=n_feat,
                               text=z.values, texttemplate="%{text}",
                               colorbar=dict(thickness=12, title=f"of {n_feat}"),
                               hovertemplate="%{y} in %{x}<br>%{z} of " + str(n_feat) +
                                             " features shift<extra></extra>"))
    return _layout(fig, "Shifted features of each sensor", height=80 + 24 * len(z), legend=False)


def composition_bars(comp: pd.DataFrame) -> go.Figure:
    """100% stacked bars of the class mix per window."""
    fig = go.Figure()
    windows = list(dict.fromkeys(comp["window"]))
    for i, g in enumerate(dict.fromkeys(comp["gas"])):
        sub = comp[comp["gas"] == g].set_index("window").reindex(windows)
        fig.add_bar(x=windows, y=sub["share"].fillna(0), name=g, marker_color=PALETTE[i % len(PALETTE)],
                    hovertemplate=f"{g}<br>%{{x}}: %{{y:.1%}}<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.25)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return _layout(fig, "Gas mix per window (labels; evaluation only)", height=320)


def pca_points(proj: Dict, focus_bucket: int) -> go.Figure:
    pts = proj["points"]
    fig = go.Figure()
    for b, color, name, size in ((1, NEUTRAL, "bucket 1 (reference)", 3), (focus_bucket, PALETTE[1], f"bucket {focus_bucket}", 3)):
        d = pts[pts["bucket"] == b]
        fig.add_trace(go.Scatter3d(x=d["PC1"], y=d["PC2"], z=d["PC3"], mode="markers", name=name,
                                   marker=dict(size=size, color=color, opacity=0.75),
                                   customdata=d["class"], hovertemplate="%{customdata}<extra>" + name + "</extra>"))
    ev = proj["explained"]
    fig.update_layout(scene=dict(xaxis_title=f"PC1 ({ev[0]:.1%})", yaxis_title=f"PC2 ({ev[1]:.1%})",
                                 zaxis_title=f"PC3 ({ev[2]:.1%})"))
    return _layout(fig, "Images in the first three principal components", height=480)


def pca_centroids(proj: Dict, focus_bucket: int) -> go.Figure:
    c = proj["centroids"]
    n = len(c)
    ramp = [SEQ[1 + int(round(i * (len(SEQ) - 2) / max(n - 1, 1)))] for i in range(n)]
    fig = go.Figure(go.Scatter3d(
        x=c["PC1"], y=c["PC2"], z=c["PC3"], mode="lines+markers+text", text=[f"b{b}" for b in c["bucket"]],
        textposition="top center", line=dict(color=NEUTRAL, width=4),
        marker=dict(size=[10 if b == focus_bucket else 6 for b in c["bucket"]], color=ramp,
                    symbol=["diamond" if b == focus_bucket else "circle" for b in c["bucket"]]),
        customdata=c["n_images"], hovertemplate="bucket %{text}<br>centroid of %{customdata:,} images<extra></extra>",
        showlegend=False))
    ev = proj["explained"]
    fig.update_layout(scene=dict(xaxis_title=f"PC1 ({ev[0]:.1%})", yaxis_title=f"PC2 ({ev[1]:.1%})",
                                 zaxis_title=f"PC3 ({ev[2]:.1%})"))
    return _layout(fig, "Path of the bucket centroids (light = early, dark = late; diamond = focus)", height=480,
                   legend=False)


# --------------------------------------------------------------------------- #
# 3. root cause

def shap_heatmap(shap: pd.DataFrame) -> go.Figure:
    order = _order(shap)
    z = shap.pivot(index="label", columns="window", values="change").reindex(columns=order)
    sig = shap.pivot(index="label", columns="window", values="change_significant").reindex(columns=order)
    z = z.loc[z.abs().max(axis=1).sort_values().index]
    sig = sig.loc[z.index]
    lim = float(np.nanmax(np.abs(z.values)))
    text = [[f"{v:+.3f}{'*' if s else ''}" for v, s in zip(zr, sr)] for zr, sr in zip(z.values, sig.values)]
    fig = go.Figure(go.Heatmap(z=z.values, x=order, y=list(z.index), text=text, texttemplate="%{text}",
                               colorscale=DIVERGING, zmin=-lim, zmax=lim, colorbar=dict(thickness=12, title="Δ"),
                               hovertemplate="%{y} in %{x}<br>change in mean |SHAP| %{z:+.4f}<extra></extra>"))
    fig.update_xaxes(tickangle=-35)
    return _layout(fig, "Change in the model's reliance on each feature (* = significant)", height=90 + 34 * len(z),
                   legend=False)


def shap_dumbbell(shap: pd.DataFrame, window: str) -> go.Figure:
    fig = go.Figure()
    for _, r in shap.iterrows():
        fig.add_trace(go.Scatter(x=[r["ref_importance"], r["test_importance"]], y=[r["label"]] * 2, mode="lines",
                                 line=dict(color=PALETTE[1] if r["change_significant"] else NEUTRAL,
                                           width=3 if r["change_significant"] else 1.5,
                                           dash="solid" if r["change_significant"] else "dot"),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=shap["ref_importance"], y=shap["label"], mode="markers", name="reference",
                             marker=dict(size=11, color=NEUTRAL),
                             hovertemplate="%{y}<br>reference mean |SHAP| %{x:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=shap["test_importance"], y=shap["label"], mode="markers", name=window,
        marker=dict(size=11, color=PALETTE[0]),
        customdata=np.c_[shap["change"], shap["change_ci_low"], shap["change_ci_high"]],
        hovertemplate=("%{y}<br>" + window + " mean |SHAP| %{x:.4f}<br>change %{customdata[0]:+.4f} "
                       "[%{customdata[1]:+.4f}, %{customdata[2]:+.4f}]<extra></extra>")))
    return _layout(fig, f"Reliance per feature: reference → {window} (solid orange = significant change)",
                   x_title="mean |SHAP| on the predicted probability", height=90 + 34 * len(shap))


def interval_bars(df: pd.DataFrame, label_col: str, value_col: str, lo_col: str, hi_col: str,
                  sig_col: str, title: str, x_title: str) -> go.Figure:
    colors = [PALETTE[0] if v > 0 else PALETTE[7] for v in df[value_col]]
    fig = go.Figure(go.Bar(
        y=df[label_col], x=df[value_col], orientation="h",
        marker=dict(color=[c if s else "rgba(0,0,0,0)" for c, s in zip(colors, df[sig_col])],
                    line=dict(color=colors, width=1.5)),
        error_x=dict(type="data", symmetric=False, array=df[hi_col] - df[value_col],
                     arrayminus=df[value_col] - df[lo_col], color=NEUTRAL, thickness=1),
        hovertemplate="%{y}<br>%{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
    return _layout(fig, title, x_title=x_title, height=max(300, 26 * len(df) + 90), legend=False)


def concept_count_line(val: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=val["bucket"], y=val["significant concepts"], mode="lines+markers+text",
                               text=val["significant concepts"], textposition="top center",
                               line=dict(color=PALETTE[0], width=2), marker=dict(size=9, color=PALETTE[0]),
                               hovertemplate="bucket %{x}<br>%{y} concepts shift significantly<extra></extra>"))
    fig.update_xaxes(dtick=1, title_text="bucket")
    fig.update_yaxes(range=[0, val["significant concepts"].max() * 1.2])
    return _layout(fig, "Concepts whose share changes significantly vs bucket 1", "concepts", height=300, legend=False)


def concept_rho_line(val: pd.DataFrame, alpha: float) -> go.Figure:
    sig = val["p (all)"] < alpha
    fig = go.Figure(go.Scatter(
        x=val["bucket"], y=val["rho (all)"], mode="lines+markers", line=dict(color=PALETTE[0], width=2),
        marker=dict(size=10, color=[PALETTE[0] if s else "rgba(0,0,0,0)" for s in sig], line=dict(color=PALETTE[0], width=2)),
        customdata=np.c_[val["p (all)"], val["tags"]],
        hovertemplate="bucket %{x}<br>Spearman ρ %{y:.2f}<br>permutation p %{customdata[0]:.3f} (%{customdata[1]} tags)<extra></extra>"))
    fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    fig.update_xaxes(dtick=1, title_text="bucket")
    fig.update_yaxes(range=[-1, 1])
    return _layout(fig, "Agreement with the image tags per bucket (filled = significant)", "Spearman ρ", height=300,
                   legend=False)


def concept_tag_scatter(pairs: pd.DataFrame, rho: float, p: float, bucket: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=pairs["tag shift"], y=pairs["concept shift"], mode="markers+text", text=pairs["tag"],
                               textposition="top center", marker=dict(size=10, color=PALETTE[0]),
                               customdata=pairs["concepts"],
                               hovertemplate="tag %{text} ↔ %{customdata}<br>tag shift %{x:+.3f}<br>concept shift %{y:+.4f}<extra></extra>"))
    fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
    return _layout(fig, f"Bucket {bucket}: concept shift vs tag shift (ρ = {rho:.2f}, p = {p:.2f})",
                   "concept shift (CLIP)", "tag shift (CLEAR metadata)", height=420, legend=False)


def share_bars(ref_share: float, prod_share: float, ref_label: str, prod_label: str, concept: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=[ref_share, prod_share], y=[ref_label, prod_label], orientation="h",
                           marker_color=[NEUTRAL, PALETTE[1]], text=[f"{ref_share:.0%}", f"{prod_share:.0%}"],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>%{x:.1%} of images have '" + concept + "' as their best concept<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(ref_share, prod_share) * 1.25])
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "", height=150, legend=False)


def _wrap(label: str, width: int = 18) -> str:
    """Break a label into lines of at most about `width` characters (Plotly <br>)."""
    lines, cur = [], ""
    for word in label.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return "<br>".join(lines + [cur])


def decision_matrix(dp: pd.DataFrame) -> go.Figure:
    order = list(dict.fromkeys(dp["window"]))[::-1]
    questions = list(dict.fromkeys(dp["question"]))
    fig = go.Figure()
    style = {"yes": dict(color=PALETTE[0], line=dict(color=PALETTE[0], width=2), size=17),
             "no": dict(color="rgba(0,0,0,0)", line=dict(color=PALETTE[0], width=2), size=17),
             "not reached": dict(color=NEUTRAL, line=dict(width=0), size=5)}
    for ans, st in style.items():
        d = dp[dp["answer"] == ans]
        fig.add_trace(go.Scatter(x=d["question"], y=d["window"], mode="markers", name=ans, marker=dict(symbol="circle", **st),
                                 customdata=d["rationale"],
                                 hovertemplate="%{y}<br>%{x}: " + ans + "<extra></extra>"))
    v = dp.drop_duplicates("window")
    for verdict, g in v.groupby("verdict"):
        fig.add_trace(go.Scatter(
            x=["verdict"] * len(g), y=g["window"], mode="markers+text", text=[f"  {t}" for t in g["verdict"]],
            textposition="middle right", marker=dict(symbol="square", size=16, color=TYPE_COLOR.get(verdict, NEUTRAL)),
            customdata=np.c_[g["stage"], g["rationale"]], showlegend=False,
            hovertemplate="<b>%{y}: " + verdict + "</b> (%{customdata[0]} stage)<br>%{customdata[1]}<extra></extra>"))
    # wrap the long questions onto two lines so neighbouring headers do not overlap
    fig.update_xaxes(categoryorder="array", categoryarray=questions + ["verdict"], showgrid=False, side="top",
                     tickangle=0, range=[-0.5, len(questions) + 1.2], tickvals=questions + ["verdict"],
                     ticktext=[_wrap(q) for q in questions] + ["verdict"])
    fig.update_yaxes(categoryorder="array", categoryarray=order, showgrid=False)
    _layout(fig, "", height=130 + 40 * len(order))
    return fig.update_layout(legend=dict(orientation="h", y=-0.02, yanchor="top", x=0), margin=dict(b=40))


def prior_lines(df: pd.DataFrame) -> go.Figure:
    fig, pos, groups = _group_panels(df, "positive rate", "")
    styles = {"actual (needs labels)": (PALETTE[0], "circle", "solid"),
              "predicted by the model": (PALETTE[1], "square-open", "dash"),
              "BBSE estimate (label free)": (PALETTE[2], "diamond-open", "dot")}
    for g in groups:
        d = df[df["group"] == g]
        for series, (color, sym, dash) in styles.items():
            for i, seg in enumerate(_segments(d[d["series"] == series])):
                _add(fig, go.Scatter(x=seg["window"], y=seg["positive rate"], name=series, legendgroup=series,
                                     showlegend=g == groups[0] and i == 0,
                                     mode="lines+markers" if g == "temporal" else "markers",
                                     line=dict(color=color, width=2, dash=dash),
                                     marker=dict(size=10, symbol=sym, color=color, line=dict(width=2, color=color)),
                                     hovertemplate="%{x}<br>" + series + " %{y:.3f}<extra></extra>"), pos[g])
    fig.update_yaxes(tickformat=".0%")
    _layout(fig, "Share of positive cases: actual, predicted and estimated", height=400)
    return fig.update_layout(legend=dict(y=1.13), margin=dict(t=100))


# --------------------------------------------------------------------------- #
# 4. reliability

def auroc_dots(rel: pd.DataFrame) -> go.Figure:
    series = [("auroc_confidence_only", "confidence alone", PALETTE[0], "circle"),
              ("auroc_risk", "fused score", PALETTE[1], "diamond-open"),
              ("auroc_instability", "instability", PALETTE[2], "square-open"),
              ("auroc_ood", "out of distribution", PALETTE[3], "x-thin-open")]
    fig = go.Figure()
    for col, name, color, sym in series:
        fig.add_trace(go.Scatter(x=rel["window"], y=rel[col], mode="markers", name=name,
                                 marker=dict(size=12, symbol=sym, color=color, line=dict(width=2, color=color)),
                                 hovertemplate="%{x}<br>" + name + " AUROC %{y:.3f}<extra></extra>"))
    fig.add_hline(y=0.5, line_dash="dot", line_color=NEUTRAL, annotation_text="chance (0.5)",
                  annotation_position="bottom left")
    fig.update_xaxes(tickangle=-35)
    return _layout(fig, "How well each signal ranks wrong predictions above right ones", "AUROC", height=380)


def risk_shares(rel: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for color, band in zip(ORDINAL3, ("low", "medium", "high")):
        fig.add_bar(x=rel["window"], y=rel[f"share_{band}"], name=f"{band.upper()} risk", marker_color=color,
                    hovertemplate="%{x}<br>" + band + ": %{y:.1%} of predictions<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.3)
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickangle=-35)
    _layout(fig, "Share of predictions per risk label", "share", height=340)
    return fig.update_layout(legend=dict(traceorder="normal"))


def risk_error_rates(rel: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for color, band, sym in zip(ORDINAL3, ("low", "medium", "high"), ("circle", "square", "diamond")):
        fig.add_trace(go.Scatter(x=rel["window"], y=rel[f"error_rate_{band}"], mode="markers", name=f"{band.upper()} risk",
                                 marker=dict(size=11, symbol=sym, color=color),
                                 hovertemplate="%{x}<br>" + band + " risk: %{y:.1%} of these predictions are wrong<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(tickangle=-35)
    return _layout(fig, "Observed error rate inside each risk label", "error rate", height=340)


def risk_hist(records: pd.DataFrame, thresholds: Dict[str, float], window: str) -> go.Figure:
    fig = go.Figure()
    for outcome, color in (("correct", PALETTE[0]), ("wrong", PALETTE[1])):
        d = records[records["outcome"] == outcome]
        fig.add_trace(go.Histogram(x=d["risk_score"], name=f"{outcome} ({len(d)})", marker_color=color, opacity=0.7,
                                   histnorm="probability", xbins=dict(start=0, end=1, size=0.025),
                                   hovertemplate="risk %{x}<br>%{y:.1%} of " + outcome + " predictions<extra></extra>"))
    fig.update_layout(barmode="overlay")
    for name, x in (("LOW | MEDIUM", thresholds["LOW_below"]), ("MEDIUM | HIGH", thresholds["HIGH_from"])):
        fig.add_vline(x=x, line_dash="dot", line_color=NEUTRAL, annotation_text=name, annotation_position="top")
    fig.update_yaxes(tickformat=".0%")
    _layout(fig, f"Fused risk score of right and wrong predictions, {window}", "share of predictions",
            "estimated probability of error", height=360)
    return fig.update_layout(legend=dict(orientation="v", x=0.99, xanchor="right", y=0.95, yanchor="top"))


def fusion_bars(coefs: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=coefs["coefficient"], y=coefs["signal"], orientation="h", marker_color=PALETTE[0],
                           text=[f"{c:+.2f}" for c in coefs["coefficient"]], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>%{x:+.3f} log-odds per reference SD<extra></extra>"))
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
    return _layout(fig, "Learned fusion weights", x_title="log-odds of error per reference SD", height=230, legend=False)


# --------------------------------------------------------------------------- #
# 5. remediation

def remediation_heatmap(t: pd.DataFrame) -> go.Figure:
    settings = list(dict.fromkeys(t["setting"]))[::-1]
    strategies = list(dict.fromkeys(t["strategy"]))
    z = t.pivot(index="setting", columns="strategy", values="recovered").reindex(index=settings, columns=strategies) * 100
    sig = t.pivot(index="setting", columns="strategy", values="significant").reindex(index=settings, columns=strategies)
    reason = t.pivot(index="setting", columns="strategy", values="reason").reindex(index=settings, columns=strategies)
    # Colours saturate at the largest gain so small gains stay visible next to large losses;
    # the number in each cell is exact.
    gains = z.values[np.isfinite(z.values) & (z.values > 0)]
    lim = float(max(gains.max() if gains.size else 0.0, 0.5))

    def cell(v, s):
        if np.isnan(v):
            return "n/a"
        v = 0.0 if abs(v) < 0.005 else v
        return (f"{v:+.2f}" if abs(v) < 1 else f"{v:+.1f}") + ("★" if s else "")

    text = [[cell(v, s) for v, s in zip(zr, sr)] for zr, sr in zip(z.values, sig.values)]
    hover = [[(f"not applicable: {rs}" if np.isnan(v) else f"{v:+.2f} pp" + (" (significant)" if s else ""))
              for v, s, rs in zip(zr, sr, rr)] for zr, sr, rr in zip(z.values, sig.values, reason.values)]
    labels = [s.replace(" ", "<br>", 1) for s in strategies]
    fig = go.Figure(go.Heatmap(z=z.values, x=labels, y=settings, text=text, texttemplate="%{text}", customdata=hover,
                               colorscale=DIVERGING, zmin=-lim, zmax=lim, xgap=2, ygap=2,
                               colorbar=dict(thickness=12, title="pp", tickvals=[-lim, 0, lim],
                                             ticktext=[f"≤ −{lim:.1f}", "0", f"≥ +{lim:.1f}"]),
                               hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>"))
    fig.update_xaxes(side="top", showgrid=False)
    fig.update_yaxes(showgrid=False)
    return _layout(fig, "", height=110 + 36 * len(settings), legend=False)


def triage_scorecard(sc: pd.DataFrame) -> go.Figure:
    x = list(sc["setting"])
    fig = go.Figure()
    rows = [("triage says", sc["triage"] == "escalate", "escalate", "hold"),
            ("truth", sc["truth"] == "a probe truly helps", "a probe truly helps", "no probe helps")]
    for row, mask, yes, no in rows:
        fig.add_trace(go.Scatter(x=x, y=[row] * len(x), mode="markers", showlegend=False,
                                 marker=dict(size=18, symbol=["triangle-up" if m else "circle-open" for m in mask],
                                             color=PALETTE[0], line=dict(width=2, color=PALETTE[0])),
                                 customdata=[yes if m else no for m in mask],
                                 hovertemplate="%{x}<br>" + row + ": %{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=["triage right?"] * len(x), mode="text", showlegend=False,
                             text=["✓" if c else "✗" for c in sc["correct"]],
                             textfont=dict(size=20, color=[GOOD if c else CRITICAL for c in sc["correct"]]),
                             customdata=sc["rationale"], hovertemplate="%{x}<br>%{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="escalate / helps",
                             marker=dict(size=12, symbol="triangle-up", color=PALETTE[0])))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="hold / does not help",
                             marker=dict(size=12, symbol="circle-open", color=PALETTE[0], line=dict(width=2))))
    fig.update_yaxes(categoryorder="array", categoryarray=["triage right?", "truth", "triage says"], showgrid=False)
    fig.update_xaxes(showgrid=False)
    return _layout(fig, "", height=230)


def probe_scatter(pr: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for color, (probe, d) in zip(PALETTE, pr.groupby("probe", sort=False)):
        fig.add_trace(go.Scatter(
            x=d["predicted"] * 100, y=d["actual"] * 100, mode="markers", name=probe,
            marker=dict(size=11, color=[color if s else "rgba(0,0,0,0)" for s in d["predicted significant"]],
                        line=dict(width=2, color=color)),
            error_x=dict(type="data", symmetric=False, array=(d["pred high"] - d["predicted"]) * 100,
                         arrayminus=(d["predicted"] - d["pred low"]) * 100, color=color, thickness=1),
            error_y=dict(type="data", symmetric=False, array=(d["act high"] - d["actual"]) * 100,
                         arrayminus=(d["actual"] - d["act low"]) * 100, color=color, thickness=1),
            customdata=d["setting"],
            hovertemplate="%{customdata}<br>predicted %{x:+.2f} pp<br>actual %{y:+.2f} pp<extra>" + probe + "</extra>"))
    lo = float(min(pr["pred low"].min(), pr["act low"].min()) * 100)
    hi = float(max(pr["pred high"].max(), pr["act high"].max()) * 100)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=NEUTRAL, dash="dot", width=1),
                             name="predicted = actual", hoverinfo="skip"))
    fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
    return _layout(fig, "Probe gain: label-free prediction vs graded result (filled = predicted significant)",
                   "actual gain (pp, labelled holdout)", "predicted gain (pp, no labels)", height=420)


# --------------------------------------------------------------------------- #
# benchmark

def bench_true_drop(runs: pd.DataFrame, types: Sequence[str]) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(types), shared_yaxes=True, subplot_titles=list(types), horizontal_spacing=0.02)
    fig.update_annotations(font_size=12)
    for i, t in enumerate(types):
        d = runs[runs["injected"] == t].sort_values("intensity_order")
        fig.add_trace(go.Box(x=d["intensity"], y=d["true drop"] * 100, boxpoints="all", jitter=0.5, pointpos=0,
                             marker=dict(size=3, color=TYPE_COLOR[t]), line=dict(color=TYPE_COLOR[t], width=1.5),
                             fillcolor="rgba(0,0,0,0)", showlegend=False, name=t,
                             hovertemplate=t + " %{x}<br>true drop %{y:.2f} pp<extra></extra>"), row=1, col=i + 1)
    fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    fig.update_yaxes(title_text="true accuracy drop (pp)", row=1, col=1)
    return _layout(fig, "", height=340, legend=False)


def bench_alarm_heatmap(rates: pd.DataFrame, types: Sequence[str]) -> go.Figure:
    rates = rates.copy()
    rates["cell"] = rates["injected"] + " · " + rates["intensity"]
    rows = []
    for t in types:
        rows += list(rates[rates["injected"] == t].sort_values("intensity_order")["cell"].drop_duplicates())
    z = rates.pivot(index="cell", columns="detector", values="rate").reindex(rows[::-1])
    fig = go.Figure(go.Heatmap(z=z.values, x=list(z.columns), y=list(z.index), colorscale=SEQ_SCALE, zmin=0, zmax=1,
                               text=[[f"{v:.0%}" for v in r] for r in z.values], texttemplate="%{text}", xgap=2, ygap=2,
                               colorbar=dict(thickness=12, title="alarm rate", tickformat=".0%"),
                               hovertemplate="%{y}<br>%{x} alarms in %{z:.0%} of windows<extra></extra>"))
    fig.update_xaxes(side="top")
    return _layout(fig, "", height=80 + 28 * len(rows), legend=False)


def false_alarms(fa: pd.DataFrame, alpha: float) -> go.Figure:
    fig = go.Figure(go.Scatter(x=fa["rate"], y=fa["detector"], mode="markers", marker=dict(size=12, color=PALETTE[0]),
                               error_x=dict(type="data", symmetric=False, array=fa["ci high"] - fa["rate"],
                                            arrayminus=fa["rate"] - fa["ci low"], color=PALETTE[0], thickness=1.5),
                               customdata=np.c_[fa["k"], fa["n"]],
                               hovertemplate="%{y}: %{customdata[0]} of %{customdata[1]} control windows<extra></extra>"))
    fig.add_vline(x=alpha, line_dash="dot", line_color=NEUTRAL, annotation_text=f"α = {alpha}", annotation_position="top")
    fig.update_xaxes(tickformat=".0%", rangemode="tozero")
    return _layout(fig, "False alarms on no-drift control windows (95% CI)", x_title="false alarm rate", height=260,
                   legend=False)


def bench_estimate_scatter(runs: pd.DataFrame, types: Sequence[str]) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(types), shared_yaxes=True, shared_xaxes=True, subplot_titles=list(types),
                        horizontal_spacing=0.02)
    fig.update_annotations(font_size=12)
    lo = float(min(runs["true accuracy"].min(), runs["estimated accuracy"].min()))
    hi = float(max(runs["true accuracy"].max(), runs["estimated accuracy"].max()))
    for i, t in enumerate(types):
        d = runs[runs["injected"] == t]
        fig.add_trace(go.Scatter(x=d["true accuracy"], y=d["estimated accuracy"], mode="markers",
                                 marker=dict(size=4, color=TYPE_COLOR[t], opacity=0.6), showlegend=False,
                                 customdata=np.c_[d["intensity"], d["state"]],
                                 hovertemplate=t + " %{customdata[0]} (%{customdata[1]})<br>true %{x:.3f}<br>estimated %{y:.3f}<extra></extra>"),
                      row=1, col=i + 1)
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=NEUTRAL, dash="dot", width=1),
                                 showlegend=False, hoverinfo="skip"), row=1, col=i + 1)
    fig.update_yaxes(title_text="CBPE estimate", row=1, col=1)
    fig.update_xaxes(title_text="true accuracy")
    return _layout(fig, "", height=330, legend=False)


def confusion_heatmap(cm: pd.DataFrame) -> go.Figure:
    frac = cm.div(cm.sum(axis=1), axis=0)
    fig = go.Figure(go.Heatmap(z=frac.values, x=list(cm.columns), y=list(cm.index), colorscale=SEQ_SCALE, zmin=0,
                               zmax=1, text=cm.values, texttemplate="%{text}", showscale=False, xgap=2, ygap=2,
                               hovertemplate="injected %{y}<br>diagnosed %{x}<br>%{z:.0%} of this row<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    return _layout(fig, "", "injected", "diagnosed", height=360, legend=False)


def bench_per_intensity(pi: pd.DataFrame, types: Sequence[str]) -> go.Figure:
    fig = make_subplots(rows=1, cols=len(types), shared_yaxes=True, subplot_titles=list(types), horizontal_spacing=0.02)
    fig.update_annotations(font_size=12)
    for i, t in enumerate(types):
        d = pi[pi["injected"] == t]
        fig.add_trace(go.Scatter(x=d["intensity"], y=d["correct"], mode="lines+markers",
                                 line=dict(color=TYPE_COLOR[t], width=2), marker=dict(size=9, color=TYPE_COLOR[t]),
                                 error_y=dict(type="data", symmetric=False, array=d["ci high"] - d["correct"],
                                              arrayminus=d["correct"] - d["ci low"], color=TYPE_COLOR[t]),
                                 customdata=np.c_[d["k"], d["n"]], showlegend=False,
                                 hovertemplate=t + " %{x}<br>%{customdata[0]} of %{customdata[1]} correct<extra></extra>"),
                      row=1, col=i + 1)
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_yaxes(title_text="correct diagnosis", row=1, col=1)
    return _layout(fig, "", height=300, legend=False)


def audit_budget(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for t, g in df.groupby("type", sort=False):
        g = g.sort_values("labels")
        color = TYPE_COLOR.get(t, NEUTRAL)
        fig.add_scatter(x=g["labels"], y=g["correct"], mode="lines+markers", name=t,
                        line=dict(color=color, width=2), marker=dict(size=9, color=color),
                        error_y=dict(type="data", symmetric=False, array=g["ci high"] - g["correct"],
                                     arrayminus=g["correct"] - g["ci low"], color=color, thickness=1),
                        hovertemplate=t + "<br>%{x} labels: %{y:.0%} correct<extra></extra>")
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_xaxes(tickvals=sorted(df["labels"].unique()))
    return _layout(fig, "Correct diagnosis by audit size", "correct", "labels in the audit", height=330)


def stage_share(st: pd.DataFrame, types: Sequence[str]) -> go.Figure:
    fig = go.Figure()
    for color, stage in ((PALETTE[0], "label-free"), (PALETTE[1], "audit")):
        d = st[st["stage"] == stage].set_index("injected").reindex(types).reset_index()
        fig.add_bar(y=d["injected"], x=d["share"], orientation="h", name=f"{stage} stage", marker_color=color,
                    customdata=d["count"], hovertemplate="%{y}: %{customdata} windows decided at the " + stage + " stage<extra></extra>")
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(autorange="reversed")
    _layout(fig, "Where the verdict was reached", height=280)
    return fig.update_layout(legend=dict(traceorder="normal"))


def bench_reliability_gain(bt: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=bt["fused minus confidence"], y=bt["injected"], mode="markers", marker=dict(size=12, color=PALETTE[0]),
        error_x=dict(type="data", symmetric=False, array=bt["fused minus confidence high"] - bt["fused minus confidence"],
                     arrayminus=bt["fused minus confidence"] - bt["fused minus confidence low"], color=PALETTE[0]),
        hovertemplate="%{y}<br>fused minus confidence AUROC %{x:+.4f}<extra></extra>"))
    fig.add_vline(x=0, line_color=NEUTRAL, line_width=1)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Fused score minus confidence alone (AUROC, 95% CI)", x_title="AUROC difference", height=280,
                   legend=False)


def bench_triage_strip(runs: pd.DataFrame, types: Sequence[str]) -> go.Figure:
    fig = go.Figure()
    rng = np.random.default_rng(0)       # jitter only (display)
    pos = {t: i for i, t in enumerate(types)}
    for decision, color, sym in (("hold", NEUTRAL, "circle-open"), ("escalate", PALETTE[1], "triangle-up")):
        d = runs[runs["triage"] == decision]
        x = d["injected"].map(pos) + rng.uniform(-0.3, 0.3, len(d))
        fig.add_trace(go.Scatter(x=x, y=d["actual gain"] * 100, mode="markers", name=f"triage: {decision}",
                                 marker=dict(size=6, color=color, symbol=sym, line=dict(width=1.2, color=color)),
                                 customdata=np.c_[d["injected"], d["intensity"], d["retraining helps"].map({True: "yes", False: "no"})],
                                 hovertemplate="%{customdata[0]} %{customdata[1]}<br>actual gain %{y:+.2f} pp<br>"
                                               "retraining significantly helps: %{customdata[2]}<extra></extra>"))
    fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    fig.update_xaxes(tickvals=list(pos.values()), ticktext=list(pos.keys()), showgrid=False)
    return _layout(fig, "Actual gain from retraining, coloured by the triage decision", "actual gain (pp)", height=380)


def triage_confusion(conf: Dict[str, int]) -> go.Figure:
    z = [[conf["tp"], conf["fn"]], [conf["fp"], conf["tn"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["triage: escalate", "triage: hold"], y=["retraining helps", "retraining does not help"],
                               text=z, texttemplate="%{text}", colorscale=SEQ_SCALE, showscale=False, xgap=2, ygap=2,
                               hovertemplate="%{y} · %{x}: %{z} windows<extra></extra>"))
    fig.update_xaxes(side="top")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "", height=230, legend=False)


# --------------------------------------------------------------------------- #
# baselines and ablations

KIND_COLOR = {"ours": PALETTE[0], "baseline": PALETTE[1], "learned": PALETTE[2], "ablation": NEUTRAL}


def method_dots(df: pd.DataFrame) -> go.Figure:
    """Share of windows typed correctly per method, with 95% Wilson intervals."""
    d = df.sort_values("accuracy")
    fig = go.Figure()
    for kind, color in KIND_COLOR.items():
        k = d[d["kind"] == kind]
        if k.empty:
            continue
        p = k["p-value"].astype(float)
        fig.add_trace(go.Scatter(
            x=k["accuracy"], y=k["method"], mode="markers", name=kind,
            marker=dict(size=13, color=color, symbol="diamond" if kind == "ours" else "circle"),
            error_x=dict(type="data", symmetric=False, array=k["ci high"] - k["accuracy"],
                         arrayminus=k["accuracy"] - k["ci low"], color=color, thickness=1.5, width=5),
            customdata=np.c_[k["macro F1"], p.fillna(np.nan), k["ours minus method"].astype(float).fillna(np.nan)],
            hovertemplate=("<b>%{y}</b><br>correct %{x:.1%}<br>macro F1 %{customdata[0]:.3f}"
                           "<br>ours minus this method %{customdata[2]:+.3f} (paired p %{customdata[1]:.2g})"
                           "<extra></extra>")))
    fig.update_xaxes(tickformat=".0%", range=[0, 1.02], title_text="share of windows whose drift type is named correctly")
    fig.update_yaxes(categoryorder="array", categoryarray=list(d["method"]))
    return _layout(fig, "Drift type: ours against baselines, a learned rule and ablations", height=120 + 42 * len(d))


def estimator_mae(df: pd.DataFrame) -> go.Figure:
    """Mean absolute error of each label-free accuracy estimate by injected type."""
    types = list(dict.fromkeys(df["type"]))
    fig = go.Figure()
    for i, (name, g) in enumerate(df.groupby("estimator", sort=False)):
        g = g.set_index("type").reindex(types)
        fig.add_trace(go.Scatter(
            x=g["MAE"] * 100, y=types, mode="markers", name=name,
            marker=dict(size=11, color=PALETTE[i % len(PALETTE)], symbol="diamond" if "ours" in name else "circle"),
            error_x=dict(type="data", symmetric=False, array=(g["ci high"] - g["MAE"]) * 100,
                         arrayminus=(g["MAE"] - g["ci low"]) * 100, color=PALETTE[i % len(PALETTE)], thickness=1.2),
            hovertemplate=name + "<br>%{y}: %{x:.2f} pp<extra></extra>"))
    fig.update_xaxes(title_text="mean absolute error of the accuracy estimate (pp)", rangemode="tozero")
    fig.update_yaxes(categoryorder="array", categoryarray=types[::-1])
    return _layout(fig, "Label-free accuracy estimates: CBPE against three baselines", height=140 + 40 * len(types))


def mechanism_lines(df: pd.DataFrame) -> go.Figure:
    """Design B: correct-type rate by mechanism and intensity."""
    fig = go.Figure()
    order = ["mild", "moderate", "strong"]
    for i, (m, g) in enumerate(df.groupby("mechanism", sort=False)):
        g = g.set_index("intensity").reindex(order)
        fig.add_trace(go.Scatter(x=order, y=g["correct"], mode="lines+markers", name=m,
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=2), marker=dict(size=10),
                                 customdata=g["true drop"] * 100,
                                 hovertemplate=m + "<br>%{x}: %{y:.0%} correct<br>mean true accuracy drop "
                                                   "%{customdata:.1f} pp<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], title_text="correct drift type")
    return _layout(fig, "Held-out mechanisms: correct type by intensity", height=380)
