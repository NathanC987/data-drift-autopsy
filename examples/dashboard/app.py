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
        metric_keys = ["accuracy", "precision", "recall", "f1"]
        cols = st.columns(4)
        for idx, key in enumerate(metric_keys):
            value = baseline.get(key)
            if value is not None:
                cols[idx].metric(key.title(), f"{float(value):.3f}")
            else:
                cols[idx].metric(key.title(), "N/A")
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

        for metric_name in available_metrics:
            st.subheader(metric_name.title())
            metric_df = proxy_df[proxy_df["metric"] == metric_name]

            col1, col2 = st.columns(2)
            with col1:
                lower_threshold = st.number_input(
                    f"{metric_name.title()} Lower Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.70,
                    step=0.01,
                    key=f"clear10_proxy_lower_{metric_name}",
                )
            with col2:
                upper_threshold = st.number_input(
                    f"{metric_name.title()} Upper Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.00,
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

            selected_df = classwise_df[
                (classwise_df["metric"] == selected_metric)
                & (classwise_df["class_id"] == selected_class_id)
            ]

            c1, c2 = st.columns(2)
            with c1:
                class_lower_threshold = st.number_input(
                    "Class-wise Lower Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.70,
                    step=0.01,
                    key="clear10_classwise_proxy_lower",
                )
            with c2:
                class_upper_threshold = st.number_input(
                    "Class-wise Upper Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.00,
                    step=0.01,
                    key="clear10_classwise_proxy_upper",
                )

            class_name = class_options[class_options["class_id"] == selected_class_id]["class_name"].iloc[0]
            fig = viz.create_proxy_metric_step_chart(
                selected_df,
                metric_name=f"{selected_metric} ({class_name})",
                lower_threshold=class_lower_threshold,
                upper_threshold=class_upper_threshold,
            )
            st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    st.header("3. Drift Detection")
    drift_df = loader.get_clear10_drift_timeline()
    if drift_df.empty:
        st.info("No drift timeline data available.")
    else:
        detectors = sorted(drift_df["detector"].unique().tolist())
        standard_thresholds = {
            "Ks Test": 0.05,
            "Psi": 0.2,
            "Mmd": 0.1,
            "Cbpe": 0.05,
        }

        for detector_name in detectors:
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
    if rca_df.empty:
        st.info("No RCA summary data available.")
    else:
        st.dataframe(rca_df, width="stretch")


def main():
    """Main dashboard application."""
    st.title("Data Drift Autopsy Dashboard")
    st.markdown("Interactive visualization of drift detection and drift autopsy results")
    st.markdown("---")

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
