"""Data Drift Autopsy dashboard.

Reads the study outputs written by the scripts in examples/quickstart/ and
shows each study in the order the pipeline runs: setup, label-free accuracy
estimate, detection, localisation, root cause, diagnosis, reliability and
remediation. Run with:  streamlit run examples/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts  # noqa: E402
import loaders as L  # noqa: E402

FIG_PNG = Path("outputs/figures")

st.set_page_config(page_title="Data Drift Autopsy", layout="wide")


def fmt_p(p):
    if p is None or pd.isna(p):
        return "--"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def missing(name: str, script: str) -> None:
    st.info(f"{name} not found. Run `python {script}` first.")


# --------------------------------------------------------------------------- #

def window_study(study, title: str, remediation_group, rem) -> None:
    summary = L.window_summary(study)
    ref_acc = study["reference"]["test_accuracy"]
    cfg = study["config"]

    st.subheader("1. Setup")
    c = st.columns(4)
    c[0].metric("Reference accuracy (held out)", f"{ref_acc:.4f}")
    c[1].metric("Training rows", f"{cfg['split_sizes']['train']:,}")
    c[2].metric("Reference window rows", f"{cfg['split_sizes']['calibration']:,}")
    c[3].metric("Significance level", f"{cfg['alpha']}")
    st.caption(f"Model: {cfg['model']}. Every detector decision is a test at the significance level above.")

    st.subheader("2. Accuracy and the label-free estimate")
    st.plotly_chart(charts.accuracy_vs_estimate(summary, ref_acc), use_container_width=True)
    show = summary[["window", "type", "accuracy", "accuracy change", "drop p-value", "CBPE estimate", "CBPE error"]].copy()
    show["drop p-value"] = show["drop p-value"].map(fmt_p)
    st.dataframe(show.style.format({"accuracy": "{:.4f}", "accuracy change": "{:+.4f}", "CBPE estimate": "{:.4f}",
                                    "CBPE error": "{:+.4f}"}), hide_index=True, use_container_width=True)

    st.subheader("3. Detection")
    det = L.detector_table(study)
    pivot = det.pivot(index="window", columns="detector", values="drift").reindex(summary["window"])
    st.dataframe(pivot.map(lambda d: "alert" if d else "--"), use_container_width=True)
    with st.expander("Scores and p-values"):
        det["p-value"] = det["p-value"].map(fmt_p)
        st.dataframe(det, hide_index=True, use_container_width=True)

    return summary


def acs_tab(acs, rem) -> None:
    summary = window_study(acs, "ACS Income", "acs", rem)
    windows = list(acs["windows"])

    st.subheader("4. Localization")
    loc = L.acs_localization(acs)
    st.plotly_chart(charts.localization_heatmap(loc), use_container_width=True)
    unseen = loc.dropna(subset=["unseen category share"])
    unseen = unseen[unseen["unseen category share"] > 0.01]
    if not unseen.empty:
        st.caption("Categories never seen in the reference (share of production rows): "
                   + "; ".join(f"{r.feature} in {r.window}: {r['unseen category share']:.1%}" for _, r in unseen.iterrows()))

    st.subheader("5. Root cause: change in SHAP attribution")
    w = st.selectbox("Window", windows, index=len(windows) - 5 if len(windows) > 5 else 0, key="acs_shap")
    shap = L.acs_shap(acs, w)
    if not shap.empty:
        st.plotly_chart(charts.interval_bars(shap, "feature", "change", "change_ci_low", "change_ci_high",
                                             "change_significant", f"Change in mean |SHAP| ({w} minus reference)",
                                             "change in predicted probability"), use_container_width=True)
        st.caption("Filled bars: the Bonferroni-adjusted bootstrap interval excludes zero.")
        for rec in acs["windows"][w]["pipelines"]["KS Test"]["rca"]["recommendations"]:
            st.write("- " + rec)

    st.subheader("6. Diagnosis")
    st.dataframe(summary[["window", "diagnosis", "stage", "rationale"]], hide_index=True, use_container_width=True)
    bb = pd.DataFrame([{"window": k, "actual positive rate": v["positive_rate"],
                        "predicted positive rate": v["predicted_positive_rate"],
                        "BBSE estimated positive rate": v["diagnosis"]["bbse"]["estimated_production_prior"][-1]}
                       for k, v in acs["windows"].items()])
    st.dataframe(bb.style.format("{:.3f}", subset=bb.columns[1:]), hide_index=True, use_container_width=True)

    reliability_section(acs, "7")
    remediation_section(rem, ("acs_temporal", "acs_geographic"), "8")


def clear_tab(clear, concepts, rem) -> None:
    window_study(clear, "CLEAR-10", "clear10", rem)

    st.subheader("4. Localization")
    loc = pd.DataFrame([{"bucket": k, "drifted dimensions (of 512)": w["localization"]["n_drifted_dimensions"],
                         **{f"{c} (dims)": v["n_drifted_dimensions"] for c, v in w["predicted_class_slices"].items()}}
                        for k, w in clear["windows"].items()])
    st.dataframe(loc, hide_index=True, use_container_width=True)
    st.caption("Per predicted class (label free): embedding dimensions that drift within the class, Holm-corrected.")

    st.subheader("5. Root cause: concept probe and class activation maps")
    if concepts is None:
        missing("Concept probe results", "examples/quickstart/clip_concept_probe_demo.py")
    else:
        st.dataframe(L.concept_validation(concepts), hide_index=True, use_container_width=True)
        b = st.selectbox("Bucket", list(concepts["buckets"]),
                         index=list(concepts["buckets"]).index(str(concepts["config"]["figure_bucket"])))
        focus = st.toggle(f"Focus class ({concepts['buckets'][b]['focus_class']['class']})", value=False)
        df = L.concept_shifts(concepts, b, focus)
        df = pd.concat([df.head(6), df.tail(8)])
        st.plotly_chart(charts.interval_bars(df, "concept", "delta", "ci_low", "ci_high", "significant",
                                             f"Concept mass shift, bucket {b} vs. bucket 1", "shift"),
                        use_container_width=True)
        ex = concepts["buckets"][str(concepts["config"]["figure_bucket"])].get("exemplars")
        if ex:
            st.caption(f"Most “{ex['concept']}”-like images (same rule for both rows)")
            for side in ("reference", "production"):
                cols = st.columns(len(ex[side]["images"]))
                for col, path, p in zip(cols, ex[side]["images"], ex[side]["probabilities"]):
                    if Path(path).exists():
                        col.image(path, caption=f"{side}, p={p:.2f}", use_container_width=True)
    cam = FIG_PNG / "gen_cam.png"
    if cam.exists():
        st.image(str(cam), caption="Class activation maps of the most out-of-distribution images "
                                   "(built by examples/paper/build_artifacts.py)")

    st.subheader("6. Diagnosis")
    st.dataframe(L.window_summary(clear)[["window", "diagnosis", "stage", "rationale"]], hide_index=True,
                 use_container_width=True)
    reliability_section(clear, "7")
    remediation_section(rem, ("clear10",), "8")


def reliability_section(study, n: str) -> None:
    st.subheader(f"{n}. Per-prediction reliability")
    rel = L.reliability_summary(study)
    st.dataframe(rel.style.format("{:.3f}", subset=rel.columns[1:], na_rep="--"), hide_index=True,
                 use_container_width=True)
    st.plotly_chart(charts.risk_bands(rel), use_container_width=True)
    coefs = study["reliability_model"]["fusion"]["coefficients"]
    st.caption("Fusion coefficients (log-odds per reference SD): "
               + ", ".join(f"{k} {v:+.2f}" for k, v in coefs.items())
               + ". Labels: low below the reference error rate, high from 0.5.")


def remediation_section(rem, groups, n: str) -> None:
    st.subheader(f"{n}. Remediation and triage")
    if rem is None:
        missing("Remediation results", "examples/quickstart/remediation_demo.py")
        return
    t = L.remediation_table(rem)
    t = t[t["group"].isin(groups)]
    if t.empty:
        return
    piv = t.pivot_table(index=["setting", "gap", "triage", "truly helps"], columns="strategy",
                        values="recovered").reset_index()
    st.dataframe(piv.style.format({c: "{:+.4f}" for c in piv.columns if c not in ("setting", "triage", "truly helps")}),
                 hide_index=True, use_container_width=True)
    st.caption("Recovered accuracy on a labelled holdout of each production window, against the deployed model.")


def benchmark_tab(bench) -> None:
    s = bench["summary"]
    ti = s["type_identification"]
    c = st.columns(3)
    c[0].metric("Windows", s["n_runs"])
    c[1].metric("Replicates (state x seed)", s["n_replicates"])
    lo, hi = ti["overall"]["ci"]
    c[2].metric("Correct drift type", f"{ti['overall']['rate']:.1%}", help=f"95% CI {lo:.1%} to {hi:.1%}")
    st.plotly_chart(charts.confusion_heatmap(L.benchmark_confusion(bench)), use_container_width=True)
    types = L.benchmark_types(bench)
    st.dataframe(types.style.format({c: "{:.3f}" for c in types.columns if c != "injected"}), hide_index=True,
                 use_container_width=True)
    st.plotly_chart(charts.audit_budget(L.benchmark_audit_budget(bench)), use_container_width=True)
    fa = pd.DataFrame([{"detector": k, "false alarm rate on no-drift windows": v["rate"]}
                       for k, v in s["detector_false_alarms"].items()])
    st.dataframe(fa, hide_index=True)
    with st.expander("Injection recipes"):
        st.json(bench["config"]["intensity_ladder"])


def main() -> None:
    st.title("Data Drift Autopsy")
    data = {k: L.load(p) for k, p in L.DEFAULT_PATHS.items()}
    tabs = st.tabs(["ACS Income", "CLEAR-10", "Injection benchmark"])
    with tabs[0]:
        acs_tab(data["acs"], data["remediation"]) if data["acs"] else missing(
            "ACS results", "examples/quickstart/folktables_demo.py")
    with tabs[1]:
        clear_tab(data["clear"], data["concepts"], data["remediation"]) if data["clear"] else missing(
            "CLEAR-10 results", "examples/quickstart/clear10_drift_study.py")
    with tabs[2]:
        benchmark_tab(data["benchmark"]) if data["benchmark"] else missing(
            "Benchmark results", "examples/quickstart/drift_injection_benchmark.py")


main()
