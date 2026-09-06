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
    """Render the existing Folktables dashboard view."""
    if not selected_years:
        st.warning("Please select at least one year")
        return

    if not selected_detectors:
        st.warning("Please select at least one detector")
        return

    # Summary metrics
    st.header("Summary Metrics")

    summary = loader.get_summary_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Years Analyzed",
            summary["total_years"],
        )

    with col2:
        st.metric(
            "Drift Events Detected",
            summary["total_drift_events"],
        )

    with col3:
        st.metric(
            "Average Accuracy",
            f"{summary['avg_accuracy']:.1%}",
            delta=None,
        )

    with col4:
        st.metric(
            "Drifted Features",
            summary["unique_drifted_features"],
        )

    st.markdown("---")

    # Load filtered data
    all_detectors_df = loader.get_all_detectors_timeline()
    all_detectors_df = all_detectors_df[
        (all_detectors_df["year"].isin(selected_years))
    ]

    detector_name_map = {d.replace("_", " ").title(): d for d in selected_detectors}
    all_detectors_df = all_detectors_df[
        all_detectors_df["detector"].isin(detector_name_map.keys())
    ]

    perf_df = loader.get_performance_metrics()
    perf_df = perf_df[perf_df["year"].isin(selected_years)]

    feature_df = loader.get_feature_drift_timeline()
    feature_df = feature_df[feature_df["year"].isin(selected_years)]

    slice_df = loader.get_slice_analysis_results()
    if not slice_df.empty:
        slice_df = slice_df[slice_df["detector"].isin(detector_name_map.keys())]

    # Main visualizations
    st.header("Drift Analysis")

    drift_detectors_df = all_detectors_df[all_detectors_df["detector"].isin(["Ks Test", "Psi", "Mmd"])].copy()
    cbpe_df = all_detectors_df[all_detectors_df["detector"] == "Cbpe"].copy()

    # Row 1: CBPE
    st.subheader("Performance Estimator (CBPE)")
    col1, col2 = st.columns(2)

    with col1:
        if not cbpe_df.empty:
            fig = viz.create_drift_timeline(cbpe_df, title="CBPE Score Timeline")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No CBPE data available")

    with col2:
        if not cbpe_df.empty:
            fig = viz.create_detector_comparison(cbpe_df, title="CBPE Comparison")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No CBPE data available")

    # Row 2: Drift detectors
    st.subheader("Drift Detectors (KS Test, PSI, MMD)")
    col1, col2 = st.columns(2)

    with col1:
        if not drift_detectors_df.empty:
            fig = viz.create_drift_timeline(drift_detectors_df, title="Drift Score Timeline")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No drift detector data available")

    with col2:
        if not drift_detectors_df.empty:
            fig = viz.create_detector_comparison(drift_detectors_df, title="Detector Comparison")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No drift detector data available")

    # Row 3: Performance and severity
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance Over Time")
        if not perf_df.empty:
            fig = viz.create_performance_chart(perf_df)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No performance data available")

    with col2:
        st.subheader("Drift Severity Distribution")
        if not all_detectors_df.empty:
            fig = viz.create_severity_distribution(all_detectors_df)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No severity data available")

    st.markdown("---")

    # Feature-level analysis
    st.header("Feature-Level Analysis")

    if not feature_df.empty:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Feature Drift Heatmap")
            fig = viz.create_feature_heatmap(feature_df)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Top Drifted Features")
            top_n = st.slider("Number of features", 5, 20, 10)
            fig = viz.create_top_drifted_features(feature_df, top_n=top_n)
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("No feature drift data available")

    st.markdown("---")

    # Detection timeline
    st.header("Drift Detection Timeline")
    if not all_detectors_df.empty:
        fig = viz.create_drift_detection_timeline(all_detectors_df)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No drift detection data available")

    st.markdown("---")

    # Slice-level analysis
    st.header("Slice-Level Drift Analysis")
    if not slice_df.empty:
        col1, col2 = st.columns([2, 1])
        slice_label_col = "slice_key_label" if "slice_key_label" in slice_df.columns else "slice_key"

        with col1:
            fig = viz.create_slice_drift_heatmap(slice_df, title="Slice Drift Score by Detector")
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Slice Summary")
            st.metric("Slices Evaluated", slice_df[slice_label_col].nunique())
            st.metric("Slice Drift Events", int(slice_df["drift_detected"].sum()))
            st.metric("Avg Slice Score", f"{slice_df['score'].mean():.4f}")

        st.subheader("Slice-Level Details")
        st.dataframe(
            slice_df[[
                "analysis_key",
                "detector",
                slice_label_col,
                "drift_detected",
                "severity",
                "score",
                "reference_samples",
                "test_samples",
            ]],
            width="stretch",
        )
    else:
        st.info("No slice analysis data available. Run a pipeline with slice_config enabled.")

    st.markdown("---")

    # Root Cause Analysis
    st.header("Root Cause Analysis")
    rca_df = loader.get_rca_results()
    importance_changes_df = loader.get_feature_importance_changes()

    if not rca_df.empty and not importance_changes_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Feature Importance Comparison")
            top_n_importance = st.slider("Number of features to compare", 5, 15, 10, key="importance_slider")
            fig = viz.create_feature_importance_comparison(importance_changes_df, top_n=top_n_importance)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.subheader("Importance Changes Over Time")
            top_features = st.slider("Number of features to track", 3, 10, 5, key="timeline_slider")
            fig = viz.create_importance_change_timeline(importance_changes_df, top_features=top_features)
            st.plotly_chart(fig, width="stretch")

        st.subheader("Feature Importance Changes Heatmap")
        fig = viz.create_feature_importance_heatmap(importance_changes_df)
        st.plotly_chart(fig, width="stretch")

        st.subheader("Recommendations")
        rec_df = viz.create_rca_recommendations_table(rca_df)
        if not rec_df.empty:
            st.dataframe(rec_df, width="stretch")
        else:
            st.info("No recommendations available")

        st.subheader("RCA Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyses", len(rca_df))
        with col2:
            total_recs = rca_df["n_recommendations"].sum()
            st.metric("Total Recommendations", int(total_recs))
        with col3:
            avg_recs = rca_df["n_recommendations"].mean()
            st.metric("Avg Recommendations/Analysis", f"{avg_recs:.1f}")
    else:
        st.info("No RCA data available. Enable RCA in your drift detection pipeline to see root cause analysis.")

    render_reliability_section(loader=loader, scope="folktables")

    st.markdown("---")

    render_remediation_dashboard(loader=loader)
    
    if show_raw_data:
        st.markdown("---")
        st.header("Raw Data Tables")

        tab1, tab2, tab3, tab4 = st.tabs(["Detector Results", "Feature Drift", "Performance Metrics", "RCA Data"])

        with tab1:
            st.subheader("Detector Results")
            st.dataframe(all_detectors_df, width="stretch")

        with tab2:
            st.subheader("Feature Drift")
            st.dataframe(feature_df, width="stretch")

        with tab3:
            st.subheader("Performance Metrics")
            st.dataframe(perf_df, width="stretch")

        with tab4:
            st.subheader("RCA Results")
            if not rca_df.empty:
                st.dataframe(rca_df, width="stretch")
            else:
                st.info("No RCA data available")


def render_clear10_dashboard(loader: DriftResultsLoader) -> None:
    """Render CLEAR-10 dashboard view in required section order."""
    st.header("1. Baseline Model Performance")
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

    st.header("2. Proxy Performance Estimation")
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

    st.header("3. Drift Detection")
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

    st.header("4. Localization")
    localization_df = loader.get_clear10_localization_summary()
    if localization_df.empty:
        st.info("No localization summary data available.")
    else:
        st.dataframe(localization_df, width="stretch")

    st.markdown("---")

    st.header("5. RCA")
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
