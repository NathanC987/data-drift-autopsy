import streamlit as st
import pandas as pd
import json
import subprocess
import time
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Try to import google genai; handle gracefully if not installed
try:
    from google import genai
except ImportError:
    genai = None


def render_remediation_dashboard(loader) -> None:
    """Render the Remediation operations and Agentic AI reasoning view."""

    st.header("Agentic AI Model Remediation")
    st.markdown(
        "Use an LLM to analyze Root Cause Analysis (RCA) and autonomously determine the best remediation strategy."
    )

    rca_df = loader.get_rca_results()
    if rca_df.empty:
        st.warning("No RCA data available. Please run the model pipelines with RCA enabled.")
        return

    st.subheader("1. Agent Initialization")

    if genai is None:
        st.error(
            "The `google-genai` library is not installed. "
            "Please run `python -m uv pip install google-genai` to enable real Agentic AI."
        )
        return

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or api_key == "your_api_key_here":
        st.error("Missing Gemini API Key! Please paste your key into the `.env` file in the project directory.")
        return

    try:
        client = genai.Client(api_key=api_key)
        st.success("Agent securely connected to Gemini APIs via `.env` file!")
    except Exception as e:
        st.error(f"Failed to initialize Agent: {e}")
        return

    st.markdown("---")
    st.subheader("2. Analyze Data Drift & Trigger Diagnosis")

    # Prepare the RCA context
    rca_summary_json = rca_df[["year", "detector", "n_recommendations"]].to_dict(orient="records")

    # Get top shifted features from importance changes
    importance_df = loader.get_feature_importance_changes()
    top_shifted_features = []
    if not importance_df.empty:
        shifted = importance_df.sort_values(by="abs_change", ascending=False).head(5)
        top_shifted_features = shifted["feature"].tolist()

    context_string = f"""
    You are a Senior MLOps Engineer Agent. Analyzed Model Drift Data:
    - RCA Summaries: {json.dumps(rca_summary_json)}
    - Top Shifted Features Causing Drift: {', '.join(top_shifted_features)}

    Task:
    1. Read the provided Root Cause Analysis.
    2. Identify the core fault.
    3. Recommend a data-centric or model-centric Remedial Strategy (e.g., Feature Dropping, Sample Weighting, Incremental Retraining, or Full Retraining). Explain WHY.
    4. End your message with the exact button the user should click below.

    Keep your reasoning concise (3-4 sentences max).
    """

    if st.button("Run Agentic Analysis"):
        with st.spinner("Agent is reasoning about the data drift..."):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=context_string,
                )
                st.session_state["agent_recommendation"] = response.text
            except Exception as e:
                st.error(f"Agent failed to respond: {e}")

    if "agent_recommendation" in st.session_state:
        st.info("🤖 **Agent Insight:**")
        st.write(st.session_state["agent_recommendation"])

        st.markdown("---")
        st.subheader("3. Execute Remediation Action")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔨 Execute: Retrain Whole Model", use_container_width=True):
                execute_retrain(strategy="full_retrain")

        with col2:
            if st.button("🛠️ Execute: Apply Feature Drop & Retrain", use_container_width=True):
                st.warning("Executing modified pipeline without drifted features...")
                execute_retrain(strategy="feature_drop", drop_features=top_shifted_features)


def execute_retrain(drop_features=None, strategy="full_retrain"):
    """Trigger the backend remediation pipeline on the current system."""
    project_root = Path(__file__).parent.parent.parent
    script_path = project_root / "examples" / "quickstart" / "retrain_demo.py"
    results_path = project_root / "outputs" / "remediation_results.json"

    python_exe = sys.executable

    cmd = [python_exe, str(script_path), "--strategy", strategy]
    if drop_features:
        cmd += ["--drop-features", ",".join(drop_features)]

    st.info(f"🚀 Starting remediation ({strategy}) using: `{python_exe}`")
    st.info("📊 **Strategy:** localisation-driven remediation on the ACS temporal shift")

    progress_bar = st.progress(0, text="Starting retraining pipeline...")
    status_placeholder = st.empty()

    try:
        start_time = time.time()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_root),
            env=env,
        )

        stages = [
            "Loading datasets (2014-2018)...",
            "Training original model (2014 only)...",
            "Training retrained model (2014+2015+2016)...",
            "Evaluating both models...",
            "Comparing results...",
            "Saving comparison...",
        ]
        timeout_seconds = 1800  # 30 minutes max

        while process.poll() is None:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                process.kill()
                st.error("⏰ Retraining timed out after 30 minutes.")
                return

            progress = min(elapsed / 600, 0.95)  # Estimate ~10 min
            stage_idx = min(int(progress * len(stages)), len(stages) - 1)
            progress_bar.progress(progress, text=stages[stage_idx])
            status_placeholder.text(f"⏱️ Elapsed: {elapsed:.0f}s | {stages[stage_idx]}")
            time.sleep(2)

        stdout, stderr = process.communicate()
        duration = time.time() - start_time

        if process.returncode == 0:
            progress_bar.progress(1.0, text="Complete!")
            status_placeholder.empty()
            st.success(f"✅ Retraining completed successfully in {duration:.1f} seconds.")
            st.balloons()

            # Show the comparison results
            if results_path.exists():
                with open(results_path) as f:
                    results = json.load(f)

                st.markdown("---")
                st.subheader("4. Before vs After Comparison")

                st.markdown(f"**Original Model** trained on: `{results['original_training_years']}`")
                st.markdown(f"**Retrained Model** trained on: `{results['retrained_training_years']}`")

                comparison = results.get("comparison", {})
                for year, data in comparison.items():
                    col1, col2, col3 = st.columns(3)
                    improvement = data["improvement"]
                    with col1:
                        st.metric(f"Year {year} - Original", f"{data['original_accuracy']:.2%}")
                    with col2:
                        st.metric(f"Year {year} - Retrained", f"{data['retrained_accuracy']:.2%}")
                    with col3:
                        delta_color = "normal" if improvement > 0 else "inverse"
                        st.metric(
                            f"Year {year} - Change",
                            f"{improvement:+.2%}",
                            delta=f"{improvement:+.2%}",
                            delta_color=delta_color,
                        )
        else:
            progress_bar.empty()
            status_placeholder.empty()
            error_output = stderr.decode("utf-8", errors="replace")
            st.error(f"❌ Retraining failed after {duration:.1f} seconds.")
            st.code(error_output[-2000:] if len(error_output) > 2000 else error_output)
    except Exception as e:
        st.error(f"❌ Failed to execute training process: {e}")
