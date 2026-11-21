"""Chest X-ray Drift Detection & Root Cause Analysis (RCA) dashboard tab."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from drift_autopsy.core.chest_xray_pipeline import XrayAnalysisResult, run_xray_analysis
from drift_autopsy.rca.grad_cam import (
    compute_gradcam,
    gradcam_overlay,
    weighted_focus_score,
)
from drift_autopsy.reporting.report_generator import generate_drift_report

_STAGES = [
    "Setup and\\nsplit",
    "Proxy estimate (CBPE)\\n+ detection\\n+ localization",
    "Grad-CAM RCA\\n+ drift type",
    "Reliability\\n(OOD)",
]


def _pipeline_dot() -> str:
    nodes = "\n".join(f'  s{i} [label="{i + 1}. {s}"];' for i, s in enumerate(_STAGES))
    edges = " -> ".join(f"s{i}" for i in range(len(_STAGES)))
    return f"""digraph {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#cde2fb", color="#2a78d6", fontcolor="#0d366b",
        fontname="Helvetica", fontsize=10, margin="0.12,0.06"];
  edge [color="#8c8b86", arrowsize=0.7];
{nodes}
  {edges};
}}"""


def _shows(text: str) -> None:
    with st.container(border=True):
        st.markdown(f"**What the data shows.** {text}")


def _resolve_local_dataset() -> Tuple[Optional[Path], Optional[Path]]:
    """Auto-resolve local Chest X-ray dataset directory and metadata CSV."""
    root = Path.cwd()
    candidate_dirs = [
        root / "data" / "chest X ray" / "sample" / "images",
        root / "data" / "chest X ray" / "images",
        root / "data" / "chest X ray" / "sample",
        root / "data" / "chest X ray",
        root / "dataset" / "sample" / "images",
        root / "dataset" / "images",
        root / "dataset" / "sample",
        root / "dataset",
    ]
    candidate_csvs = [
        root / "data" / "chest X ray" / "sample_labels.csv",
        root / "data" / "chest X ray" / "sample" / "sample_labels.csv",
        root / "dataset" / "sample_labels.csv",
        root / "dataset" / "sample" / "sample_labels.csv",
    ]
    dataset_dir = next((p for p in candidate_dirs if p.exists() and p.is_dir()), None)
    metadata_csv = next((p for p in candidate_csvs if p.exists() and p.is_file()), None)
    return dataset_dir, metadata_csv


def _detector_bar_chart(result: XrayAnalysisResult) -> go.Figure:
    """Grouped bar chart comparing detector decision values against thresholds."""
    names = ["MMD (p-value vs α)", "KS Test (p-value vs α)", "PSI (score vs thresh)"]
    scores = [
        float(result.mmd_result.p_value if result.mmd_result.p_value is not None else result.mmd_result.score),
        float(result.ks_result.p_value if result.ks_result.p_value is not None else result.ks_result.score),
        float(result.psi_result.score),
    ]
    thresholds = [
        float(result.mmd_result.threshold),
        float(result.ks_result.threshold),
        float(result.psi_result.threshold),
    ]
    fig = go.Figure(data=[
        go.Bar(name="Observed", x=names, y=scores, marker_color="#2a78d6"),
        go.Bar(name="Threshold", x=names, y=thresholds, marker_color="#e05a47"),
    ])
    fig.update_layout(
        barmode="group",
        title="Detector Decision Values vs Thresholds",
        yaxis_title="Value",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def _recon_histogram(result: XrayAnalysisResult, recon_thresh: float) -> go.Figure:
    """Overlay histogram of Baseline vs Drifted autoencoder reconstruction MSE."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=result.baseline_errors, name="Baseline", opacity=0.65, marker_color="#2a78d6"))
    fig.add_trace(go.Histogram(x=result.drifted_errors, name="Drifted", opacity=0.65, marker_color="#e05a47"))
    fig.add_vline(
        x=recon_thresh,
        line_dash="dash",
        line_color="#111111",
        annotation_text="2.5σ threshold",
        annotation_position="top right",
    )
    fig.update_layout(
        barmode="overlay",
        title="Autoencoder Reconstruction Error (MSE)",
        xaxis_title="MSE per image",
        yaxis_title="Count",
        height=340,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def _localization_bar_chart(result: XrayAnalysisResult) -> go.Figure:
    """Bar chart of per-dimension KS statistics across the 32-D latent space."""
    rows = [
        {
            "dimension": fd.feature_name,
            "ks_stat": fd.score,
            "drifted": "Holm-rejected" if fd.drift_detected else "Not rejected",
        }
        for fd in result.localization_result.feature_drifts
    ]
    df = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
    fig = px.bar(
        df,
        x="dimension",
        y="ks_stat",
        color="drifted",
        color_discrete_map={"Holm-rejected": "#e05a47", "Not rejected": "#9ebce6"},
        title=f"Latent Dimension Localization ({len(result.localization_result.drifted_features)}/32 shifted)",
        labels={"ks_stat": "KS statistic", "dimension": "Latent dimension"},
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def _pca_centroids_chart(result: XrayAnalysisResult) -> go.Figure:
    """Plot Baseline vs Drifted centroids in PC1-PC2 space."""
    cent = result.embedding_df.groupby("set", as_index=False)[["pc1", "pc2", "pc3"]].mean()
    ev = result.pca_explained_variance * 100.0
    fig = px.scatter(
        cent,
        x="pc1",
        y="pc2",
        color="set",
        text="set",
        size=[18, 18],
        color_discrete_map={"Baseline": "#2a78d6", "Drifted": "#e05a47"},
        title=f"Latent Centroid Shift (PC1 {ev[0]:.1f}%, PC2 {ev[1]:.1f}% var)",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def _confidence_histogram(result: XrayAnalysisResult) -> go.Figure:
    """Histogram of CNN top-class confidence on Baseline val vs Drifted."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=result.baseline_confidences, name="Baseline (val)", opacity=0.65, marker_color="#2a78d6"))
    fig.add_trace(go.Histogram(x=result.drifted_confidences, name="Drifted", opacity=0.65, marker_color="#e05a47"))
    fig.update_layout(
        barmode="overlay",
        title="Model Confidence Distribution (Baseline Val vs Drifted)",
        xaxis_title="Top-class confidence",
        yaxis_title="Count",
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def _ood_histogram(result: XrayAnalysisResult) -> go.Figure:
    """Histogram of OOD scores in [0, 1] on Baseline vs Drifted latent embeddings."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=result.baseline_ood_scores, name="Baseline", opacity=0.65, marker_color="#2a78d6"))
    fig.add_trace(go.Histogram(x=result.drifted_ood_scores, name="Drifted", opacity=0.65, marker_color="#e05a47"))
    fig.update_layout(
        barmode="overlay",
        title="Per-Prediction OOD Score Distribution (Latent Space)",
        xaxis_title="OOD score [0, 1]",
        yaxis_title="Count",
        height=320,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def render() -> None:
    """Render the Chest X-ray Drift Detection & RCA tab."""
    st.markdown(
        "**Chest X-ray (NIH sample)**: grayscale radiographs monitored with a convolutional "
        "autoencoder (32-D latent bottleneck), statistical + kernel drift tests (MMD, KS, PSI), "
        "CBPE proxy accuracy, OOD scoring, and CNN Grad-CAM saliency maps."
    )

    default_dir, default_csv = _resolve_local_dataset()

    with st.sidebar:
        st.divider()
        st.subheader("Chest X-ray Drift Controls")
        dataset_dir_str = st.text_input(
            "Dataset directory",
            value=str(default_dir) if default_dir is not None else "data/chest X ray/sample/images",
            key="xray_dataset_dir",
        )
        metadata_csv_str = st.text_input(
            "Metadata CSV (optional)",
            value=str(default_csv) if default_csv is not None else "",
            key="xray_metadata_csv",
        )

        max_samples = st.slider("Max samples", 40, 600, 160, step=20, key="xray_max_samples")
        image_size = st.selectbox("Image size", [64, 96, 128], index=0, key="xray_image_size")
        baseline_ratio = st.slider("Baseline ratio", 0.4, 0.85, 0.65, step=0.05, key="xray_baseline_ratio")
        batch_size = st.selectbox("Batch size", [16, 32, 64], index=1, key="xray_batch_size")
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1, key="xray_seed")

        st.markdown("**Drift Simulation (Drifted Split)**")
        brightness = st.slider("Brightness", 0.5, 2.0, 1.25, step=0.05, key="xray_brightness")
        contrast = st.slider("Contrast", 0.5, 2.0, 1.15, step=0.05, key="xray_contrast")
        noise_std = st.slider("Gaussian noise std", 0.0, 0.25, 0.04, step=0.01, key="xray_noise_std")

        st.markdown("**Training & Detector Thresholds**")
        ae_epochs = st.slider("Autoencoder epochs", 2, 25, 8, step=1, key="xray_ae_epochs")
        cls_epochs = st.slider("CNN classifier epochs", 2, 25, 8, step=1, key="xray_cls_epochs")
        mmd_threshold = st.number_input("MMD α threshold", 0.001, 0.25, 0.05, step=0.01, format="%.3f", key="xray_mmd_t")
        ks_threshold = st.number_input("KS α threshold", 0.001, 0.25, 0.05, step=0.01, format="%.3f", key="xray_ks_t")
        psi_threshold = st.number_input("PSI threshold", 0.05, 1.0, 0.20, step=0.05, format="%.2f", key="xray_psi_t")

        use_metadata_split = st.checkbox("Use metadata split", value=False, key="xray_meta_split")
        split_column = st.text_input("Split column", value="View Position", key="xray_split_col")
        baseline_value = st.text_input("Baseline value", value="PA", key="xray_base_val")

        run_clicked = st.button("Run Chest X-ray Drift Analysis", type="primary", key="xray_run_btn")

    if run_clicked:
        csv_arg = metadata_csv_str.strip() or None
        with st.spinner("Running autoencoder training, drift detection, localization, and CNN setup..."):
            try:
                res = run_xray_analysis(
                    dataset_dir=dataset_dir_str,
                    metadata_csv=csv_arg,
                    max_samples=int(max_samples),
                    image_size=int(image_size),
                    baseline_ratio=float(baseline_ratio),
                    batch_size=int(batch_size),
                    seed=int(seed),
                    brightness=float(brightness),
                    contrast=float(contrast),
                    noise_std=float(noise_std),
                    ae_epochs=int(ae_epochs),
                    cls_epochs=int(cls_epochs),
                    mmd_threshold=float(mmd_threshold),
                    ks_threshold=float(ks_threshold),
                    psi_threshold=float(psi_threshold),
                    use_metadata_split=bool(use_metadata_split),
                    split_column=split_column,
                    baseline_value=baseline_value,
                )
                st.session_state["xray_result"] = res
                st.session_state["xray_params"] = {
                    "brightness": float(brightness),
                    "contrast": float(contrast),
                    "noise_std": float(noise_std),
                }
            except Exception as exc:
                st.error(f"Chest X-ray analysis failed: {exc}")
                return

    if "xray_result" not in st.session_state:
        st.info(
            "Configure parameters in the sidebar and click **Run Chest X-ray Drift Analysis** "
            "to inspect reconstruction drift, latent-space detectors, localization, and Grad-CAM RCA."
        )
        return

    result: XrayAnalysisResult = st.session_state["xray_result"]
    if not hasattr(result, "localization_result") or not hasattr(result, "drifted_actual_acc"):
        st.warning("Cached analysis result is from an older schema. Click **Run Chest X-ray Drift Analysis** to refresh.")
        return
    params = st.session_state.get("xray_params", {"brightness": 1.0, "contrast": 1.0, "noise_std": 0.0})

    base_mean = float(np.mean(result.baseline_errors))
    base_std = float(np.std(result.baseline_errors))
    drift_mean = float(np.mean(result.drifted_errors))
    recon_thresh = base_mean + 2.5 * base_std
    affected_pct = float(np.mean(result.drifted_errors > recon_thresh) * 100.0)

    # Top-level overview metrics & pipeline diagram (matching CLEAR-10)
    ov = st.columns(4)
    ov[0].metric(
        "Reference accuracy (val)",
        f"{result.classifier_val_acc:.3f}" if result.classifier_val_acc is not None else "n/a",
    )
    ov[1].metric(
        "Drifted accuracy (eval)",
        f"{result.drifted_actual_acc:.3f}" if result.drifted_actual_acc is not None else "n/a",
    )
    ov[2].metric(
        "Detectors triggered",
        f"{sum(d.drift_detected for d in (result.mmd_result, result.ks_result, result.psi_result))} of 3",
    )
    ov[3].metric("Latent dims shifted", f"{len(result.localization_result.drifted_features)} of 32")
    st.graphviz_chart(_pipeline_dot(), width="content")

    # Step 1: Initial setup and results
    st.markdown("### 1. Initial setup and results")
    st.caption("How the Chest X-ray subset is split into baseline and drifted windows, and how the CNN behaves.")
    s_left, s_right = st.columns([3, 2])
    with s_left:
        split_df = pd.DataFrame({
            "Split": ["Baseline", "Drifted"],
            "Images": [len(result.baseline_indices), len(result.drifted_indices)],
        })
        fig_split = px.bar(
            split_df,
            x="Images",
            y="Split",
            orientation="h",
            color="Split",
            color_discrete_map={"Baseline": "#2a78d6", "Drifted": "#e05a47"},
            title="Baseline vs Drifted Split Size",
        )
        fig_split.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_split, width="stretch", key="xray_split_bar")
    with s_right:
        n_pos = int((result.subset.labels == 1).sum())
        n_neg = int((result.subset.labels == 0).sum())
        st.markdown(
            "\n".join([
                f"- **Total images**: {len(result.subset.images)}",
                f"- **Resolution**: {result.baseline_images.shape[-1]}×{result.baseline_images.shape[-2]} grayscale",
                f"- **Labels**: {n_neg} No Finding / {n_pos} Disease",
                f"- **Photometric drift**: brightness={params['brightness']:.2f}, contrast={params['contrast']:.2f}, noise={params['noise_std']:.3f}",
                f"- **Device**: `{result.device}`",
            ])
        )

    st.divider()

    # Step 2: Proxy performance, drift detection, and localization
    st.markdown("### 2. Proxy performance estimation, drift detection and localization")
    if result.cbpe_result is not None and result.baseline_confidences is not None:
        st.markdown("#### 2a. Label-free accuracy estimate (CBPE)")
        est_acc = result.cbpe_result.metadata.get("estimated_accuracy")
        p1, p2, p3 = st.columns(3)
        p1.metric("CBPE Estimated Accuracy", f"{est_acc:.3f}" if est_acc is not None else "n/a")
        p2.metric(
            "Measured Drifted Accuracy",
            f"{result.drifted_actual_acc:.3f}" if result.drifted_actual_acc is not None else "n/a",
        )
        p3.metric(
            "Confidence Shift (chi-square)",
            f"{result.cbpe_result.score:.2f}",
            delta=f"p={result.cbpe_result.p_value:.4f} ({'DRIFT' if result.cbpe_result.drift_detected else 'OK'})",
            delta_color="inverse" if result.cbpe_result.drift_detected else "normal",
        )
        st.plotly_chart(_confidence_histogram(result), width="stretch", key="xray_conf_hist")
        _shows(
            f"CBPE estimates drifted accuracy at {est_acc:.3f} (vs {result.drifted_actual_acc:.3f} measured) "
            f"with confidence-shift p={result.cbpe_result.p_value:.4f}."
            if est_acc is not None and result.drifted_actual_acc is not None
            else f"Confidence-shift test p={result.cbpe_result.p_value:.4f}."
        )

    st.markdown("#### 2b. Drift detection (Reconstruction MSE & Latent Space)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Reconstruction MSE (Drifted)",
        f"{drift_mean:.5f}",
        delta=f"{drift_mean - base_mean:+.5f} vs baseline",
        delta_color="inverse",
    )
    c2.metric(
        "MMD",
        f"{result.mmd_result.score:.4f}",
        delta=f"p={result.mmd_result.p_value:.4f} ({'DRIFT' if result.mmd_result.drift_detected else 'OK'})",
        delta_color="inverse" if result.mmd_result.drift_detected else "normal",
    )
    c3.metric(
        "KS Test",
        f"{result.ks_result.score:.4f}",
        delta=f"p={result.ks_result.p_value:.4f} ({'DRIFT' if result.ks_result.drift_detected else 'OK'})",
        delta_color="inverse" if result.ks_result.drift_detected else "normal",
    )
    c4.metric(
        "PSI",
        f"{result.psi_result.score:.4f}",
        delta=f"{'DRIFT' if result.psi_result.drift_detected else 'OK'} (thresh {result.psi_result.threshold:.2f})",
        delta_color="inverse" if result.psi_result.drift_detected else "normal",
    )

    d_left, d_right = st.columns(2)
    with d_left:
        st.plotly_chart(_recon_histogram(result, recon_thresh), width="stretch", key="xray_recon_hist")
    with d_right:
        st.plotly_chart(_detector_bar_chart(result), width="stretch", key="xray_det_bar")

    st.markdown("#### 2c. Drift localization: which latent dimensions moved & PCA space")
    st.caption("Each of the 32 autoencoder latent dimensions is tested with KS and Holm correction.")
    st.plotly_chart(_localization_bar_chart(result), width="stretch", key="xray_loc_bar")

    pca_dim = st.radio("PCA view", ["2D", "3D"], horizontal=True, key="xray_pca_dim")
    p_left, p_right = st.columns(2)
    with p_left:
        if pca_dim == "3D":
            fig_pca = px.scatter_3d(
                result.embedding_df,
                x="pc1",
                y="pc2",
                z="pc3",
                color="set",
                color_discrete_map={"Baseline": "#2a78d6", "Drifted": "#e05a47"},
                title="32-D Latent Embeddings (3D PCA)",
                opacity=0.75,
            )
        else:
            fig_pca = px.scatter(
                result.embedding_df,
                x="pc1",
                y="pc2",
                color="set",
                color_discrete_map={"Baseline": "#2a78d6", "Drifted": "#e05a47"},
                title="32-D Latent Embeddings (2D PCA)",
                opacity=0.75,
            )
        fig_pca.update_layout(height=340, margin=dict(l=20, r=20, t=45, b=20))
        st.plotly_chart(fig_pca, width="stretch", key="xray_pca_plot")
    with p_right:
        st.plotly_chart(_pca_centroids_chart(result), width="stretch", key="xray_pca_cent")

    _shows(
        f"{len(result.localization_result.drifted_features)} of 32 latent dimensions shifted significantly "
        f"(Holm-corrected KS), and {affected_pct:.1f}% of drifted scans exceed the 2.5σ reconstruction MSE threshold."
    )

    st.divider()

    # Step 3: Grad-CAM RCA & Drift Type
    st.markdown("### 3. Root cause analysis (Grad-CAM) and drift type")
    baseline_focus_avg: Optional[float] = None
    drifted_focus_avg: Optional[float] = None

    if result.classifier is None:
        st.info("CNN classifier was not trained (requires >= 30 labeled baseline images with > 1 class).")
    else:
        st.markdown("#### 3a. Where the model looks (Grad-CAM on highest-MSE drifted scans)")
        st.caption("Side-by-side original X-ray and Grad-CAM overlay (red = highest evidence for predicted class).")
        sample_n = min(12, len(result.baseline_images), len(result.drifted_images))
        base_scores = [
            weighted_focus_score(compute_gradcam(result.classifier, result.baseline_images[i], result.device)[0])
            for i in range(sample_n)
        ]
        drift_scores = [
            weighted_focus_score(compute_gradcam(result.classifier, result.drifted_images[i], result.device)[0])
            for i in range(sample_n)
        ]
        baseline_focus_avg = float(np.mean(base_scores))
        drifted_focus_avg = float(np.mean(drift_scores))
        focus_shift = drifted_focus_avg - baseline_focus_avg

        f1, f2, f3 = st.columns(3)
        f1.metric("Avg Baseline Top-10% Focus", f"{baseline_focus_avg:.4f}")
        f2.metric("Avg Drifted Top-10% Focus", f"{drifted_focus_avg:.4f}")
        f3.metric("Focus Shift (Drifted - Baseline)", f"{focus_shift:+.4f}")

        exceed_idx = np.where(result.drifted_errors > recon_thresh)[0]
        if len(exceed_idx) == 0:
            exceed_idx = np.argsort(result.drifted_errors)[::-1][:3]
        else:
            exceed_idx = exceed_idx[np.argsort(result.drifted_errors[exceed_idx])[::-1][:3]]

        cols = st.columns(2 * len(exceed_idx), gap="small")
        for j, idx in enumerate(exceed_idx):
            img_t = result.drifted_images[int(idx)]
            heatmap, prob = compute_gradcam(result.classifier, img_t, result.device)
            overlay = gradcam_overlay(img_t, heatmap, alpha=0.45)
            orig_rgb = np.repeat(np.clip(img_t[0].cpu().numpy(), 0.0, 1.0)[..., None], 3, axis=-1)
            err = float(result.drifted_errors[int(idx)])
            ood_u = float(result.drifted_ood_unclipped[int(idx)])
            pred_lbl = "Disease" if prob >= 0.5 else "No Finding"
            true_raw = int(result.subset.labels[result.drifted_indices[int(idx)]])
            true_lbl = "Disease" if true_raw == 1 else ("No Finding" if true_raw == 0 else "Unknown")
            cols[2 * j].image(orig_rgb, width="stretch")
            cols[2 * j + 1].image(overlay, width="stretch")
            cols[2 * j].caption(f"**pred {pred_lbl}** (P={prob:.2f})  \ntrue {true_lbl}")
            cols[2 * j + 1].caption(f"MSE={err:.5f} | OOD={ood_u:.2f}")

        st.markdown(
            "<div style='display:flex;align-items:center;gap:0.5rem;max-width:420px'>low"
            "<div style='flex:1;height:12px;border-radius:6px;"
            "background:linear-gradient(to right,rgb(0,0,128),rgb(0,255,255),rgb(255,255,0),rgb(255,0,0))'></div>"
            "high</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### 3b. Structured Drift Report")
    report = generate_drift_report(
        baseline_errors=result.baseline_errors,
        drifted_errors=result.drifted_errors,
        mmd_result=result.mmd_result,
        ks_result=result.ks_result,
        psi_result=result.psi_result,
        baseline_focus_score=baseline_focus_avg,
        drifted_focus_score=drifted_focus_avg,
        brightness=params["brightness"],
        contrast=params["contrast"],
        noise_std=params["noise_std"],
        classifier_val_acc=result.classifier_val_acc,
    )
    st.markdown(
        f"**Status:** {'Drift Detected' if report.drift_detected else 'No Drift'} | "
        f"**Type:** {report.drift_type} | **Severity:** `{report.severity}` | "
        f"**Confidence:** `{report.confidence:.2f}`"
    )
    st.markdown("**Root Causes:** " + "; ".join(report.root_causes))
    st.markdown("**Recommendations:** " + "; ".join(report.recommendations))
    st.download_button(
        "Download Drift Report (JSON)",
        data=report.to_json(),
        file_name="chest_xray_drift_report.json",
        mime="application/json",
        key="xray_download_report",
    )

    st.divider()

    # Step 4: Reliability (OOD)
    st.markdown("### 4. Per-prediction reliability (OOD)")
    st.caption("Isolation Forest + latent-space Euclidean distance fitted on baseline embeddings.")
    ood_base_pct = float(np.mean(result.baseline_ood_scores >= 0.5) * 100.0)
    ood_drift_pct = float(np.mean(result.drifted_ood_scores >= 0.5) * 100.0)
    r1, r2 = st.columns(2)
    r1.metric("Baseline High-OOD (>= 0.5)", f"{ood_base_pct:.1f}%")
    r2.metric("Drifted High-OOD (>= 0.5)", f"{ood_drift_pct:.1f}%", delta=f"{ood_drift_pct - ood_base_pct:+.1f} pp")
    st.plotly_chart(_ood_histogram(result), width="stretch", key="xray_ood_hist")
    _shows(
        f"High-OOD share shifted from {ood_base_pct:.1f}% (baseline) to {ood_drift_pct:.1f}% (drifted)."
    )
