"""
Drift Autopsy Dashboard - Interactive drift analysis visualization

Run with: streamlit run examples/dashboard/app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from examples.dashboard.data_loader import DriftResultsLoader
from examples.dashboard import visualizations as viz
from examples.dashboard import story
from examples.dashboard.remediation import render_remediation_dashboard

# Page configuration
st.set_page_config(
    page_title="Drift Autopsy Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_data(results_path: str) -> DriftResultsLoader:
    """Load drift results with caching."""
    loader = DriftResultsLoader(results_path)
    loader.load()
    return loader


def render_folktables_dashboard(
    loader: DriftResultsLoader,
    selected_years,
    selected_detectors,
    show_raw_data: bool,
) -> None:
    """Folktables (ACS Income) view, told as the seven-step pipeline story."""
    if not selected_years or not selected_detectors:
        st.warning("Select at least one year and one detector in the sidebar.")
        return

    all_detectors_df = loader.get_all_detectors_timeline()
    all_detectors_df = all_detectors_df[all_detectors_df["year"].isin(selected_years)]
    detector_name_map = {d.replace("_", " ").title(): d for d in selected_detectors}
    all_detectors_df = all_detectors_df[all_detectors_df["detector"].isin(detector_name_map.keys())]

    perf_df = loader.get_performance_metrics()
    perf_df = perf_df[perf_df["year"].isin(selected_years)]
    feature_df = loader.get_feature_drift_timeline()
    feature_df = feature_df[feature_df["year"].isin(selected_years)]
    slice_df = loader.get_slice_analysis_results()
    if not slice_df.empty:
        slice_df = slice_df[slice_df["detector"].isin(detector_name_map.keys())]
    rca_df = loader.get_rca_results()
    importance_changes_df = loader.get_feature_importance_changes()

    story.render_executive_summary(story.folktables_summary(loader))
    st.markdown("---")

    # ---- Step 0 -----------------------------------------------------------
    story.step_header(0)
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Model**\n\nStandardised logistic regression (tabular, binary: income > \\$50k)")
    c2.markdown("**Reference window**\n\nCalifornia 2014")
    c3.markdown(
        "**Production windows**\n\nCalifornia " + ", ".join(str(y) for y in selected_years)
        + " (temporal) and other US states in 2014 (geographic)"
    )
    st.markdown("---")

    # ---- Step 1 -----------------------------------------------------------
    story.step_header(1)
    cbpe_df = all_detectors_df[all_detectors_df["detector"] == "Cbpe"].copy()
    col1, col2 = st.columns(2)
    with col1:
        if not perf_df.empty:
            st.plotly_chart(viz.create_performance_chart(perf_df), width="stretch")
        else:
            st.info("No measured-accuracy series (labels held out only for validation).")
    with col2:
        if not cbpe_df.empty:
            st.plotly_chart(viz.create_drift_timeline(cbpe_df, title="CBPE confidence-shift statistic"), width="stretch")
        else:
            st.info("No CBPE data available")
    st.caption(
        "CBPE compares the model's confidence histogram now against the reference. A significant "
        "shift is a label-free signal that performance has moved; here it grows every year."
    )
    st.markdown("---")

    # ---- Step 2 -----------------------------------------------------------
    story.step_header(2)
    drift_detectors_df = all_detectors_df[all_detectors_df["detector"].isin(["Ks Test", "Psi", "Mmd"])].copy()
    col1, col2 = st.columns(2)
    with col1:
        if not drift_detectors_df.empty:
            st.plotly_chart(viz.create_drift_timeline(drift_detectors_df, title="Detector score over time"), width="stretch")
    with col2:
        if not all_detectors_df.empty:
            st.plotly_chart(viz.create_severity_distribution(all_detectors_df), width="stretch")
    if not all_detectors_df.empty:
        st.plotly_chart(viz.create_drift_detection_timeline(all_detectors_df), width="stretch")
    st.caption(
        "KS and CBPE call the shift critical every year; PSI and MMD stay quiet - the change is real "
        "but spread thinly, no single feature moves far on its own."
    )
    st.markdown("---")

    # ---- Step 3 -----------------------------------------------------------
    story.step_header(3, "features and slices")
    if not feature_df.empty:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(viz.create_feature_heatmap(feature_df), width="stretch")
        with col2:
            top_n = st.slider("Top features", 5, 20, 10)
            st.plotly_chart(viz.create_top_drifted_features(feature_df, top_n=top_n), width="stretch")
    if not slice_df.empty:
        st.subheader("Slice localisation (by US state)")
        slice_label_col = "slice_key_label" if "slice_key_label" in slice_df.columns else "slice_key"
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(viz.create_slice_drift_heatmap(slice_df, title="Slice drift by detector"), width="stretch")
        with col2:
            st.metric("Slices evaluated", slice_df[slice_label_col].nunique())
            st.metric("Slice drift events", int(slice_df["drift_detected"].sum()))
            st.metric("Avg slice score", f"{slice_df['score'].mean():.4f}")
        st.dataframe(
            slice_df[["analysis_key", "detector", slice_label_col, "drift_detected", "severity",
                      "score", "reference_samples", "test_samples"]],
            width="stretch", hide_index=True,
        )
        st.caption(
            "Cross-state comparison isolates the drift to place-of-birth and race code - deploying the "
            "California model to another state is a much larger shift than any single year."
        )
    st.markdown("---")

    # ---- Step 4 -----------------------------------------------------------
    story.step_header(4, "SHAP attribution")
    if not rca_df.empty and not importance_changes_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            n_imp = st.slider("Features to compare", 5, 15, 10, key="importance_slider")
            st.plotly_chart(viz.create_feature_importance_comparison(importance_changes_df, top_n=n_imp), width="stretch")
        with col2:
            n_trk = st.slider("Features to track", 3, 10, 5, key="timeline_slider")
            st.plotly_chart(viz.create_importance_change_timeline(importance_changes_df, top_features=n_trk), width="stretch")
        st.plotly_chart(viz.create_feature_importance_heatmap(importance_changes_df), width="stretch")
        rec_df = viz.create_rca_recommendations_table(rca_df)
        if not rec_df.empty:
            st.subheader("Generated recommendations")
            st.dataframe(rec_df, width="stretch", hide_index=True)
        st.caption(
            "SHAP shows the model swinging its reliance between occupation and place-of-birth as the "
            "surveys re-code those fields - the features that both drifted and changed importance are "
            "flagged as the likely root cause."
        )
    else:
        st.info("No SHAP RCA data. Enable RCA in the pipeline (KS Test path in the demo).")
    st.markdown("---")

    # ---- Step 5 -----------------------------------------------------------
    story.step_header(5)
    story.render_reliability_step(loader, "ACS Income")
    st.markdown("---")

    # ---- Step 6 -----------------------------------------------------------
    render_remediation_dashboard(loader=loader, scope="acs")

    if show_raw_data:
        st.markdown("---")
        with st.expander("Raw data tables"):
            tab1, tab2, tab3, tab4 = st.tabs(["Detectors", "Feature drift", "Performance", "RCA"])
            tab1.dataframe(all_detectors_df, width="stretch")
            tab2.dataframe(feature_df, width="stretch")
            tab3.dataframe(perf_df, width="stretch")
            tab4.dataframe(rca_df, width="stretch")


def render_clear10_dashboard(loader: DriftResultsLoader) -> None:
    """CLEAR-10 (image) view, told as the seven-step pipeline story."""
    story.render_executive_summary(story.clear10_summary(loader))
    st.markdown("---")

    story.step_header(0)
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Model**\n\nLinear head on frozen ImageNet ResNet-18 embeddings (11-way image classifier)")
    c2.markdown("**Reference window**\n\nBucket 1 (earliest photos)")
    c3.markdown("**Production windows**\n\nBuckets 2-10, each compared against the one before it")
    st.subheader("Reference model performance")
    baseline = loader.get_clear10_baseline_performance()

    if baseline:
        metric_aliases = {
            "accuracy": ["accuracy"],
            "precision": ["precision", "precision_macro"],
            "recall": ["recall", "recall_macro"],
            "f1": ["f1", "f1_macro"],
        }
        cols = st.columns(4)
        for idx, metric_name in enumerate(["accuracy", "precision", "recall", "f1"]):
            value = None
            for alias in metric_aliases[metric_name]:
                if alias in baseline and baseline.get(alias) is not None:
                    value = baseline.get(alias)
                    break
            if value is not None:
                cols[idx].metric(metric_name.title(), f"{float(value):.3f}")
            else:
                cols[idx].metric(metric_name.title(), "N/A")
    else:
        st.info("No baseline performance block found.")

    st.markdown("---")

    story.step_header(1)
    proxy_df = loader.get_clear10_proxy_metrics()
    if proxy_df.empty:
        st.info("No proxy metric data available.")
    else:
        metric_order = ["accuracy", "precision", "recall", "f1"]
        available_metrics = [m for m in metric_order if m in proxy_df["metric"].unique().tolist()]
        if not available_metrics:
            available_metrics = sorted(proxy_df["metric"].unique().tolist())

        metric_layout = [
            ["accuracy", "precision"],
            ["recall", "f1"],
        ]
        rendered_metrics = set()

        for row_metrics in metric_layout:
            left_col, right_col = st.columns(2)
            row_cols = [left_col, right_col]
            for col_idx, metric_name in enumerate(row_metrics):
                if metric_name not in available_metrics:
                    continue
                rendered_metrics.add(metric_name)
                metric_df = proxy_df[proxy_df["metric"] == metric_name]

                with row_cols[col_idx]:
                    st.subheader(metric_name.title())
                    t_col1, t_col2 = st.columns(2)
                    with t_col1:
                        lower_threshold = st.number_input(
                            f"{metric_name.title()} Lower Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.83,
                            step=0.01,
                            key=f"clear10_proxy_lower_{metric_name}",
                        )
                    with t_col2:
                        upper_threshold = st.number_input(
                            f"{metric_name.title()} Upper Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.87,
                            step=0.01,
                            key=f"clear10_proxy_upper_{metric_name}",
                        )

                    fig = viz.create_proxy_metric_step_chart(
                        metric_df,
                        metric_name=metric_name,
                        lower_threshold=lower_threshold,
                        upper_threshold=upper_threshold,
                    )
                    st.plotly_chart(fig, width="stretch")

        remaining_metrics = [m for m in available_metrics if m not in rendered_metrics]
        for metric_name in remaining_metrics:
            st.subheader(metric_name.title())
            metric_df = proxy_df[proxy_df["metric"] == metric_name]
            col1, col2 = st.columns(2)
            with col1:
                lower_threshold = st.number_input(
                    f"{metric_name.title()} Lower Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.83,
                    step=0.01,
                    key=f"clear10_proxy_lower_{metric_name}",
                )
            with col2:
                upper_threshold = st.number_input(
                    f"{metric_name.title()} Upper Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.87,
                    step=0.01,
                    key=f"clear10_proxy_upper_{metric_name}",
                )

            fig = viz.create_proxy_metric_step_chart(
                metric_df,
                metric_name=metric_name,
                lower_threshold=lower_threshold,
                upper_threshold=upper_threshold,
            )
            st.plotly_chart(fig, width="stretch")

        classwise_df = loader.get_clear10_proxy_metrics_classwise()
        if not classwise_df.empty:
            st.subheader("Class-wise Proxy vs Actual")
            class_metrics = [m for m in ["precision", "recall", "f1"] if m in classwise_df["metric"].unique().tolist()]
            if not class_metrics:
                class_metrics = sorted(classwise_df["metric"].unique().tolist())

            controls_col, graph_col = st.columns([1, 2])

            with controls_col:
                selected_metric = st.selectbox(
                    "Class-wise Metric",
                    options=class_metrics,
                    index=0,
                    key="clear10_classwise_metric",
                )

                class_options = (
                    classwise_df[["class_id", "class_name"]]
                    .drop_duplicates()
                    .sort_values("class_id")
                )
                class_labels = [
                    f"{int(row.class_id)} - {row.class_name}"
                    for row in class_options.itertuples(index=False)
                ]

                selected_class_label = st.selectbox(
                    "Class",
                    options=class_labels,
                    index=0,
                    key="clear10_classwise_class",
                )
                selected_class_id = int(selected_class_label.split(" - ", maxsplit=1)[0])

                class_lower_threshold = st.number_input(
                    "Class-wise Lower Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.70,
                    step=0.01,
                    key="clear10_classwise_proxy_lower",
                )
                class_upper_threshold = st.number_input(
                    "Class-wise Upper Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.00,
                    step=0.01,
                    key="clear10_classwise_proxy_upper",
                )

            selected_df = classwise_df[
                (classwise_df["metric"] == selected_metric)
                & (classwise_df["class_id"] == selected_class_id)
            ]

            class_name = class_options[class_options["class_id"] == selected_class_id]["class_name"].iloc[0]
            fig = viz.create_proxy_metric_step_chart(
                selected_df,
                metric_name=f"{selected_metric} ({class_name})",
                lower_threshold=class_lower_threshold,
                upper_threshold=class_upper_threshold,
            )

            with graph_col:
                st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    story.step_header(2)
    drift_df = loader.get_clear10_drift_timeline()
    pca_projection_df = loader.get_clear10_pca_3d_projection()
    if drift_df.empty:
        st.info("No drift timeline data available.")
    else:
        detectors = sorted(drift_df["detector"].unique().tolist())
        standard_thresholds = {
            "Ks Test": 0.05,
            "Psi": 0.2,
            "Mmd": 0.1,
            "Pca Reconstruction": 0.1,
            "Fid Distance": 0.1,
        }

        ordered_detector_rows = [
            ["Pca Reconstruction"],
            ["Fid Distance", "Mmd"],
            ["Ks Test", "Psi"],
        ]

        rendered_detectors = set()

        for detector_row in ordered_detector_rows:
            if len(detector_row) == 1:
                detector_name = detector_row[0]
                if detector_name not in detectors:
                    continue
                rendered_detectors.add(detector_name)
                detector_df = drift_df[drift_df["detector"] == detector_name]
                default_threshold = standard_thresholds.get(detector_name, 0.1)
                if detector_df["threshold"].notna().any():
                    default_threshold = float(detector_df["threshold"].dropna().iloc[0])

                threshold = st.number_input(
                    f"{detector_name} Threshold",
                    min_value=0.0,
                    value=float(default_threshold),
                    step=0.01,
                    key=f"clear10_drift_threshold_{detector_name}",
                )

                alert_direction = "below" if "ks" in detector_name.lower() else "above"
                semantics = "lower-than-threshold triggers alert" if alert_direction == "below" else "higher-than-threshold triggers alert"
                if detector_name == "Pca Reconstruction" and not pca_projection_df.empty:
                    left_col, right_col = st.columns(2)
                    with left_col:
                        st.subheader(detector_name)
                        st.caption(f"Alert semantics: {semantics}")
                        fig = viz.create_detector_step_chart(
                            detector_df,
                            detector_name=detector_name,
                            threshold=threshold,
                            alert_direction=alert_direction,
                        )
                        st.plotly_chart(fig, width="stretch")

                    with right_col:
                        st.subheader("PCA 3D Bucket Projection")
                        pca_fig = viz.create_pca_bucket_3d_scatter(pca_projection_df)
                        st.plotly_chart(pca_fig, width="stretch")
                else:
                    st.subheader(detector_name)
                    st.caption(f"Alert semantics: {semantics}")
                    fig = viz.create_detector_step_chart(
                        detector_df,
                        detector_name=detector_name,
                        threshold=threshold,
                        alert_direction=alert_direction,
                    )
                    st.plotly_chart(fig, width="stretch")
            else:
                left_col, right_col = st.columns(2)
                for col_idx, detector_name in enumerate(detector_row):
                    if detector_name not in detectors:
                        continue
                    rendered_detectors.add(detector_name)
                    current_col = left_col if col_idx == 0 else right_col
                    with current_col:
                        st.subheader(detector_name)
                        detector_df = drift_df[drift_df["detector"] == detector_name]
                        default_threshold = standard_thresholds.get(detector_name, 0.1)
                        if detector_df["threshold"].notna().any():
                            default_threshold = float(detector_df["threshold"].dropna().iloc[0])

                        threshold = st.number_input(
                            f"{detector_name} Threshold",
                            min_value=0.0,
                            value=float(default_threshold),
                            step=0.01,
                            key=f"clear10_drift_threshold_{detector_name}",
                        )

                        alert_direction = "below" if "ks" in detector_name.lower() else "above"
                        semantics = "lower-than-threshold triggers alert" if alert_direction == "below" else "higher-than-threshold triggers alert"
                        st.caption(f"Alert semantics: {semantics}")

                        fig = viz.create_detector_step_chart(
                            detector_df,
                            detector_name=detector_name,
                            threshold=threshold,
                            alert_direction=alert_direction,
                        )
                        st.plotly_chart(fig, width="stretch")

        remaining_detectors = [d for d in detectors if d not in rendered_detectors]
        for detector_name in remaining_detectors:
            st.subheader(detector_name)
            detector_df = drift_df[drift_df["detector"] == detector_name]
            default_threshold = standard_thresholds.get(detector_name, 0.1)
            if detector_df["threshold"].notna().any():
                default_threshold = float(detector_df["threshold"].dropna().iloc[0])

            threshold = st.number_input(
                f"{detector_name} Threshold",
                min_value=0.0,
                value=float(default_threshold),
                step=0.01,
                key=f"clear10_drift_threshold_{detector_name}",
            )

            alert_direction = "below" if "ks" in detector_name.lower() else "above"
            semantics = "lower-than-threshold triggers alert" if alert_direction == "below" else "higher-than-threshold triggers alert"
            st.caption(f"Alert semantics: {semantics}")

            fig = viz.create_detector_step_chart(
                detector_df,
                detector_name=detector_name,
                threshold=threshold,
                alert_direction=alert_direction,
            )
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    story.step_header(3, "embedding dimensions and class slices")
    localization_df = loader.get_clear10_localization_summary()
    if localization_df.empty:
        st.info("No localization summary data available.")
    else:
        st.dataframe(localization_df, width="stretch")

    st.markdown("---")

    story.step_header(4, "embedding-shift attribution and Grad-CAM")
    rca_df = loader.get_clear10_rca_summary()
    visual_rca_df = loader.get_clear10_visual_rca()
    if rca_df.empty:
        st.info("No RCA summary data available.")
    else:
        st.dataframe(rca_df, width="stretch")

    st.subheader("Visual RCA (Grad-CAM)")
    if visual_rca_df.empty:
        st.info("No visual RCA artifacts available.")
    else:
        enabled_df = visual_rca_df[visual_rca_df["enabled"] == True].copy()
        if enabled_df.empty:
            status_cols = ["bucket", "bucket_severity", "status", "reason"]
            st.dataframe(
                visual_rca_df[status_cols].drop_duplicates().sort_values("bucket"),
                width="stretch",
            )
        else:
            for bucket in sorted(enabled_df["bucket"].dropna().unique().tolist()):
                bucket_rows = enabled_df[enabled_df["bucket"] == bucket].sort_values("rank")
                if bucket_rows.empty:
                    continue
                severity = str(bucket_rows["bucket_severity"].iloc[0])
                st.markdown(f"**Bucket {int(bucket)} | Severity: {severity}**")

                sample_rows = []
                for _, row in bucket_rows.iterrows():
                    input_image_path = row.get("input_image_path")
                    gradcam_path = row.get("gradcam_path")
                    if not input_image_path and not gradcam_path:
                        continue

                    try:
                        score_value = float(row.get("drift_score", float("nan")))
                    except (TypeError, ValueError):
                        score_value = float("nan")

                    sample_rows.append(
                        {
                            "input_image_path": input_image_path,
                            "gradcam_path": gradcam_path,
                            "caption": (
                                f"sample={row.get('sample_id', '-')}, "
                                f"class={row.get('class_name', '-')}, "
                                f"score={score_value:.3f}"
                            ),
                        }
                    )

                for idx in range(0, len(sample_rows), 2):
                    pair = sample_rows[idx : idx + 2]
                    col_a_input, col_a_cam, col_b_input, col_b_cam = st.columns(4)

                    first = pair[0]
                    with col_a_input:
                        if first["input_image_path"] and Path(first["input_image_path"]).exists():
                            st.image(
                                first["input_image_path"],
                                caption=f"Input image | {first['caption']}",
                                use_container_width=True,
                            )
                    with col_a_cam:
                        if first["gradcam_path"] and Path(first["gradcam_path"]).exists():
                            st.image(
                                first["gradcam_path"],
                                caption=f"Grad-CAM overlay | {first['caption']}",
                                use_container_width=True,
                            )

                    if len(pair) > 1:
                        second = pair[1]
                        with col_b_input:
                            if second["input_image_path"] and Path(second["input_image_path"]).exists():
                                st.image(
                                    second["input_image_path"],
                                    caption=f"Input image | {second['caption']}",
                                    use_container_width=True,
                                )
                        with col_b_cam:
                            if second["gradcam_path"] and Path(second["gradcam_path"]).exists():
                                st.image(
                                    second["gradcam_path"],
                                    caption=f"Grad-CAM overlay | {second['caption']}",
                                    use_container_width=True,
                                )

    st.markdown("---")
    st.subheader("Concept-level cause (what changed, in words)")
    story.render_concept_rca_step(bucket="10")

    st.markdown("---")
    story.step_header(5)
    story.render_reliability_step(loader, "CLEAR-10")

    st.markdown("---")
    render_remediation_dashboard(loader=loader, scope="clear10")


def main():
    """Main dashboard application."""
    st.title("Data Drift Autopsy Dashboard")
    st.markdown("Interactive visualization of drift detection and drift autopsy results")

    # Sidebar - mode data configuration and filters
    with st.sidebar:
        st.header("Configuration")

        folktables_path = st.text_input(
            "Folktables Results File",
            value="outputs/folktables_drift_results.json",
            help="Path to Folktables JSON results file",
        )

        clear10_path = st.text_input(
            "CLEAR-10 Results File",
            value="outputs/clear10_drift_results.json",
            help="Path to CLEAR-10 JSON results file",
        )

        folktables_loader = None
        if Path(folktables_path).exists():
            try:
                folktables_loader = load_data(folktables_path)
                st.success("Folktables results loaded")
            except Exception as e:
                st.error(f"Folktables load error: {e}")
        else:
            st.warning(f"Folktables file not found: {folktables_path}")

        clear10_loader = None
        if Path(clear10_path).exists():
            try:
                clear10_loader = load_data(clear10_path)
                st.success("CLEAR-10 results loaded")
            except Exception as e:
                st.error(f"CLEAR-10 load error: {e}")
        else:
            st.info(f"CLEAR-10 file not found yet: {clear10_path}")

        st.markdown("---")
        st.subheader("Folktables Filters")

        available_years = folktables_loader.get_available_years() if folktables_loader else []
        available_detectors = folktables_loader.get_available_detectors() if folktables_loader else []

        selected_years = st.multiselect(
            "Years",
            options=available_years,
            default=available_years,
            help="Select Folktables years to display",
        )

        selected_detectors = st.multiselect(
            "Detectors",
            options=available_detectors,
            default=available_detectors,
            help="Select Folktables detectors to display",
        )

        st.markdown("---")
        st.subheader("Display Options")
        show_raw_data = st.checkbox("Show Raw Data Tables", value=False)
    # Top-tab mode switch
    folktables_tab, clear10_tab = st.tabs(["Folktables Demo", "CLEAR-10 Demo"])

    with folktables_tab:
        if folktables_loader is None:
            st.info("Provide a valid Folktables results file path in the sidebar to view this tab.")
        else:
            render_folktables_dashboard(
                loader=folktables_loader,
                selected_years=selected_years,
                selected_detectors=selected_detectors,
                show_raw_data=show_raw_data,
            )

    with clear10_tab:
        if clear10_loader is None:
            st.info("Provide a valid CLEAR-10 results file path in the sidebar to view this tab.")
        else:
            render_clear10_dashboard(clear10_loader)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>Drift Autopsy Dashboard v0.2.0 | Built with Streamlit and Plotly</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
