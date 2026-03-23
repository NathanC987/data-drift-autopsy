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


def main():
    """Main dashboard application."""
    
    # Header
    st.title("🔍 Data Drift Autopsy Dashboard")
    st.markdown("**Interactive visualization of drift detection results**")
    st.markdown("---")
    
    # Sidebar - File selector and filters
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Results file path
        default_path = "outputs/folktables_drift_results.json"
        results_path = st.text_input(
            "Results File Path",
            value=default_path,
            help="Path to JSON results file from drift analysis"
        )
        
        # Load data
        if not Path(results_path).exists():
            st.error(f"❌ File not found: {results_path}")
            st.info("💡 Run the Folktables demo to generate results:\n```\npython examples/quickstart/folktables_demo.py\n```")
            st.stop()
        
        try:
            loader = load_data(results_path)
            st.success("✅ Results loaded successfully")
        except Exception as e:
            st.error(f"❌ Error loading results: {e}")
            st.stop()
        
        # Filters
        st.markdown("---")
        st.subheader("🔍 Filters")
        
        available_years = loader.get_available_years()
        available_detectors = loader.get_available_detectors()
        
        selected_years = st.multiselect(
            "Years",
            options=available_years,
            default=available_years,
            help="Select years to display"
        )
        
        selected_detectors = st.multiselect(
            "Detectors",
            options=available_detectors,
            default=available_detectors,
            help="Select detectors to display"
        )
        
        st.markdown("---")
        st.subheader("📊 Display Options")
        show_raw_data = st.checkbox("Show Raw Data Tables", value=False)
    
    # Main content
    if not selected_years:
        st.warning("⚠️ Please select at least one year")
        return
    
    if not selected_detectors:
        st.warning("⚠️ Please select at least one detector")
        return
    
    # Summary metrics
    st.header("📈 Summary Metrics")
    
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
    
    # Filter by detector names (convert back to original format)
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
    st.header("📊 Drift Analysis")
    
    # Separate drift detectors from performance estimators
    drift_detectors_df = all_detectors_df[all_detectors_df["detector"].isin(["Ks Test", "Psi", "Mmd"])].copy()
    cbpe_df = all_detectors_df[all_detectors_df["detector"] == "Cbpe"].copy()
    
    # Row 1: CBPE (Performance Estimator)
    st.subheader("📈 Performance Estimator (CBPE)")
    col1, col2 = st.columns(2)
    
    with col1:
        if not cbpe_df.empty:
            fig = viz.create_drift_timeline(cbpe_df, title="CBPE Score Timeline")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CBPE data available")
    
    with col2:
        if not cbpe_df.empty:
            fig = viz.create_detector_comparison(cbpe_df, title="CBPE Comparison")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No CBPE data available")
    
    # Row 2: Drift Detectors (KS, PSI, MMD)
    st.subheader("🎯 Drift Detectors (KS Test, PSI, MMD)")
    col1, col2 = st.columns(2)
    
    with col1:
        if not drift_detectors_df.empty:
            fig = viz.create_drift_timeline(drift_detectors_df, title="Drift Score Timeline")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No drift detector data available")
    
    with col2:
        if not drift_detectors_df.empty:
            fig = viz.create_detector_comparison(drift_detectors_df, title="Detector Comparison")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No drift detector data available")
    
    # Row 3: Performance chart and severity distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance Over Time")
        if not perf_df.empty:
            fig = viz.create_performance_chart(perf_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available")
    
    with col2:
        st.subheader("Drift Severity Distribution")
        if not all_detectors_df.empty:
            fig = viz.create_severity_distribution(all_detectors_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No severity data available")
    
    st.markdown("---")
    
    # Feature-level analysis
    st.header("🎯 Feature-Level Analysis")
    
    if not feature_df.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Feature Drift Heatmap")
            fig = viz.create_feature_heatmap(feature_df)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Top Drifted Features")
            top_n = st.slider("Number of features", 5, 20, 10)
            fig = viz.create_top_drifted_features(feature_df, top_n=top_n)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feature drift data available")
    
    st.markdown("---")
    
    # Detection timeline
    st.header("⏱️ Drift Detection Timeline")
    if not all_detectors_df.empty:
        fig = viz.create_drift_detection_timeline(all_detectors_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No drift detection data available")
    
    st.markdown("---")

    # Slice-level analysis (geographic/sub-slice)
    st.header("🧩 Slice-Level Drift Analysis")
    if not slice_df.empty:
        col1, col2 = st.columns([2, 1])
        slice_label_col = "slice_key_label" if "slice_key_label" in slice_df.columns else "slice_key"

        with col1:
            fig = viz.create_slice_drift_heatmap(slice_df, title="Slice Drift Score by Detector")
            st.plotly_chart(fig, use_container_width=True)

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
            use_container_width=True,
        )
    else:
        st.info("No slice analysis data available. Run a pipeline with slice_config enabled.")

    st.markdown("---")
    
    # Root Cause Analysis
    st.header("🔬 Root Cause Analysis")
    rca_df = loader.get_rca_results()
    importance_changes_df = loader.get_feature_importance_changes()
    
    if not rca_df.empty and not importance_changes_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Feature Importance Comparison")
            top_n_importance = st.slider("Number of features to compare", 5, 15, 10, key="importance_slider")
            fig = viz.create_feature_importance_comparison(importance_changes_df, top_n=top_n_importance)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Importance Changes Over Time")
            top_features = st.slider("Number of features to track", 3, 10, 5, key="timeline_slider")
            fig = viz.create_importance_change_timeline(importance_changes_df, top_features=top_features)
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        st.subheader("Feature Importance Changes Heatmap")
        fig = viz.create_feature_importance_heatmap(importance_changes_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.subheader("📋 Recommendations")
        rec_df = viz.create_rca_recommendations_table(rca_df)
        if not rec_df.empty:
            st.dataframe(rec_df, use_container_width=True)
        else:
            st.info("No recommendations available")
        
        # Show RCA summary metrics
        st.subheader("📊 RCA Summary")
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
    
    # Raw data tables (optional)
    if show_raw_data:
        st.markdown("---")
        st.header("📋 Raw Data Tables")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Detector Results", "Feature Drift", "Performance Metrics", "RCA Data"])
        
        with tab1:
            st.subheader("Detector Results")
            st.dataframe(all_detectors_df, use_container_width=True)
            
        with tab2:
            st.subheader("Feature Drift")
            st.dataframe(feature_df, use_container_width=True)
            
        with tab3:
            st.subheader("Performance Metrics")
            st.dataframe(perf_df, use_container_width=True)
        
        with tab4:
            st.subheader("RCA Results")
            if not rca_df.empty:
                st.dataframe(rca_df, use_container_width=True)
            else:
                st.info("No RCA data available")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>Drift Autopsy Dashboard v0.1.0 | Built with Streamlit & Plotly</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
