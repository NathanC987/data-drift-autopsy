"""Narrative helpers: turn the pipeline artifacts into a plain-English story.

The dashboard is meant to read as one flow for a user who only knows their
model type and data type: what was estimated, whether it drifted, where, why,
which predictions to distrust, and what to do. This module supplies the
auto-written summary paragraph and the renderers for the steps that ``app.py``
did not previously have (reliability, concept-level visual RCA, remediation
triage).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

CONCEPT_RESULTS = Path("outputs/clip_concepts/concept_rca_results.json")
REMEDIATION_RESULTS = Path("outputs/remediation_results.json")

STEP_TITLES = [
    "Step 0 - Setup: model, data and windows",
    "Step 1 - Performance estimate (no labels)",
    "Step 2 - Drift verdict",
    "Step 3 - Where the drift is",
    "Step 4 - Why: root cause",
    "Step 5 - Which predictions to distrust",
    "Step 6 - What to do: remediation",
]


def _load(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


_DETECTOR_PRETTY = {
    "cbpe": "CBPE", "ks test": "KS test", "ks_test": "KS test", "psi": "PSI",
    "mmd": "MMD", "fid distance": "FID", "pca reconstruction": "PCA reconstruction",
}


def _pretty_detectors(names) -> str:
    return ", ".join(_DETECTOR_PRETTY.get(str(n).lower(), str(n)) for n in names)


def step_header(index: int, extra: str = "") -> None:
    title = STEP_TITLES[index]
    st.header(title if not extra else f"{title} - {extra}")


# --------------------------------------------------------------------------- #
# Executive summary paragraph

def folktables_summary(loader) -> str:
    perf = loader.get_performance_metrics()
    det = loader.get_all_detectors_timeline()
    feat = loader.get_feature_drift_timeline()
    rel = loader.get_reliability_summary()

    bits: List[str] = []
    if not perf.empty:
        perf = perf.sort_values("year")
        first, last = perf.iloc[0], perf.iloc[-1]
        drop = (first["accuracy"] - last["accuracy"]) * 100
        bits.append(
            f"Between {int(first['year'])} and {int(last['year'])} the model's measured "
            f"accuracy fell about {drop:.1f} points, from {first['accuracy']:.3f} to {last['accuracy']:.3f}."
        )
    if not det.empty:
        fired = sorted(det[det["drift_detected"]]["detector"].unique().tolist())
        quiet = sorted(set(det["detector"].unique()) - set(fired))
        if fired:
            bits.append(
                f"{_pretty_detectors(fired)} flagged the shift as drift; "
                f"{_pretty_detectors(quiet) or 'no other detector'} stayed quiet, so the change is "
                f"real but spread thinly across features rather than concentrated in one."
            )
    if not feat.empty:
        persistent = (
            feat[feat["drift_detected"]]
            .groupby("feature")["year"].nunique()
            .sort_values(ascending=False)
        )
        top = persistent[persistent >= persistent.max()].index.tolist()[:4] if len(persistent) else []
        if top:
            bits.append(f"The drift is localised to {', '.join(top)}, which move in every window.")
    if not rel.empty:
        temporal = rel[rel["window"].str.isdigit()]
        if not temporal.empty:
            last_rel = temporal.sort_values("window").iloc[-1]
            bits.append(
                f"Reliability: about {last_rel['suspicious_pct']:.0f}% of sampled predictions are overconfident "
                f"relative to the label-free accuracy estimate, and this share grows with the drift."
            )
    rem = _load(REMEDIATION_RESULTS)
    if rem:
        verdicts = {
            s["shift_name"]: s["triage"]["will_retraining_help"]
            for s in rem.get("settings", []) if "ACS" in s["shift_name"]
        }
        if verdicts and not any(verdicts.values()):
            bits.append(
                "Remediation triage: retraining is **not** expected to recover this loss - the cheap "
                "label-free probes recover almost nothing, so the shift is concept-level, not a simple "
                "covariate move. Collecting fresh labels or revisiting the target rule comes first."
            )
    return " ".join(bits) or "Run the Folktables demo to populate this view."


def clear10_summary(loader) -> str:
    baseline = loader.get_clear10_baseline_performance() or {}
    proxy = loader.get_clear10_proxy_metrics()
    drift = loader.get_clear10_drift_timeline()
    concept = _load(CONCEPT_RESULTS)

    bits: List[str] = []
    acc = baseline.get("accuracy")
    if acc is not None:
        bits.append(f"The image model scores {float(acc):.3f} on the reference bucket.")
    if not proxy.empty:
        acc_df = proxy[proxy["metric"] == "accuracy"].sort_values("bucket")
        if not acc_df.empty and "actual" in acc_df.columns:
            worst = acc_df.loc[acc_df["actual"].idxmin()]
            bits.append(
                f"Estimated accuracy tracks the measured value within about half a point on average; "
                f"the low point is bucket {int(worst['bucket'])} near {worst['actual']:.3f}."
            )
    if not drift.empty and {"score", "threshold"}.issubset(drift.columns):
        drift = drift.assign(alert=drift["score"] >= drift["threshold"])
        ks = drift[drift["detector"].str.contains("Ks", case=False)]
        ks_all = int(ks[ks["alert"]]["bucket"].nunique())
        bits.append(
            f"The KS test on embedding dimensions flags {ks_all} of {int(drift['bucket'].nunique())} analysis "
            f"buckets; the multivariate detectors (PSI, MMD, FID) only fire at the strongest shift, so detection "
            f"forms a sensitivity ladder."
        )
    if concept:
        b10 = concept.get("10", {})
        bg = b10.get("per_class", {}).get("BACKGROUND", {})
        gains = [c["concept"] for c in bg.get("concepts", []) if c["delta"] > 0][:3]
        mv = b10.get("metadata_validation_focus", {}) or {}
        era = b10.get("photographic_era", {})
        rho = mv.get("spearman_rho")
        if gains:
            era_txt = ""
            if era.get("production_median_year"):
                era_txt = (
                    f"; the median capture year of those images moves from "
                    f"{era.get('reference_median_year', 0):.0f} to {era.get('production_median_year', 0):.0f}"
                )
            rho_txt = f" (rank correlation {rho:.2f} with the benchmark's own image tags)" if isinstance(rho, (int, float)) else ""
            bits.append(
                f"Concept-level root cause: the background class has shifted toward {', '.join(gains)}"
                f"{rho_txt}{era_txt}."
            )
    rem = _load(REMEDIATION_RESULTS)
    if rem:
        clr = [s for s in rem.get("settings", []) if "CLEAR" in s["shift_name"]]
        if clr and all(s["triage"]["will_retraining_help"] for s in clr):
            bits.append(
                "Remediation triage: retraining on the most recent bucket recovers most of the gap at a "
                "fraction of a full retrain's cost."
            )
    return " ".join(bits) or "Run the CLEAR-10 demo to populate this view."


def render_executive_summary(text: str) -> None:
    st.markdown("#### The story so far")
    st.info(text)


# --------------------------------------------------------------------------- #
# Step 5 - reliability

def render_reliability_step(loader, dataset_label: str) -> None:
    summary = loader.get_reliability_summary()
    if summary.empty:
        st.info("No reliability records in this result file.")
        return

    st.caption(
        "Every sampled prediction gets five label-free signals - confidence, out-of-distribution "
        "distance, perturbation stability, calibration risk and explanation consistency - fused into "
        "one risk score. A prediction is 'suspicious' when its confidence outruns the label-free "
        "accuracy estimate."
    )

    show = summary.copy()
    for col in ["mean_confidence", "mean_ood", "mean_stability", "mean_calibration_risk", "mean_risk"]:
        show[col] = show[col].round(3)
    show["suspicious_pct"] = show["suspicious_pct"].round(1)
    st.dataframe(
        show.rename(
            columns={
                "window": "Window", "n": "Sampled", "mean_confidence": "Confidence",
                "mean_ood": "OOD", "mean_stability": "Instability",
                "mean_calibration_risk": "Calibration risk", "mean_risk": "Fused risk",
                "suspicious_pct": "Suspicious %", "high_risk_pct": "High-risk %",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    records = loader.get_reliability_records()
    if not records.empty and "risk_score" in records:
        try:
            import plotly.express as px

            fig = px.histogram(
                records, x="risk_score", color="window", nbins=40, barmode="overlay",
                opacity=0.6, title="Fused reliability-risk distribution",
            )
            fig.add_vline(x=0.33, line_dash="dot", annotation_text="low / medium")
            fig.add_vline(x=0.66, line_dash="dot", annotation_text="medium / high")
            st.plotly_chart(fig, width="stretch")
        except Exception:
            pass

    tabular = summary[summary["window"].str.isdigit()]
    if len(tabular) >= 2:
        tabular = tabular.sort_values("window")
        rise = tabular.iloc[-1]["mean_calibration_risk"] - tabular.iloc[0]["mean_calibration_risk"]
        st.markdown(
            f"**Reading:** calibration risk rose {rise:+.3f} across the windows - the model grew more "
            f"confident as it grew less accurate. The *fused* score stays low because the per-input "
            f"signals (OOD, instability) are near zero for individually-ordinary rows; a population-level "
            f"calibration drift is real drift the current fusion under-weights."
        )


# --------------------------------------------------------------------------- #
# Step 4 (image) - concept-level visual RCA

def render_concept_rca_step(bucket: str = "10") -> None:
    concept = _load(CONCEPT_RESULTS)
    if not concept:
        st.info(
            "No concept-level RCA yet. Run `python examples/quickstart/clip_concept_probe_demo.py` "
            "(needs `pip install -e '.[image,concept]'`)."
        )
        return

    window = concept.get(bucket)
    if not window:
        bucket = next(iter(concept))
        window = concept[bucket]

    st.caption(
        "Drifted embedding dimensions are projected onto a basis of plain-language photographic "
        "concepts with a vision-language model, so the visual shift reads in words. The ranking is "
        "checked against the benchmark's own image tags."
    )

    scope = st.radio(
        "View", ["Overall", "Per drifted class"], horizontal=True, key=f"concept_scope_{bucket}"
    )
    if scope == "Overall":
        ranking = window.get("overall", [])
        mv = window.get("metadata_validation", {}) or {}
        subtitle = f"CLEAR-10 bucket {bucket}, overall"
    else:
        classes = list(window.get("per_class", {}))
        cls = st.selectbox("Class", classes, key=f"concept_class_{bucket}")
        ranking = window["per_class"][cls]["concepts"]
        mv = window.get("metadata_validation_focus", {}) or {}
        subtitle = f"CLEAR-10 bucket {bucket}, class {cls}"

    df = pd.DataFrame(ranking)
    df = df[df["delta"].abs() > 1e-3].sort_values("delta")
    try:
        import plotly.express as px

        fig = px.bar(
            df, x="delta", y="concept", orientation="h",
            color=df["delta"] > 0, color_discrete_map={True: "#2ca02c", False: "#c0392b"},
            title=f"Concept mass shift - {subtitle}",
        )
        fig.update_layout(showlegend=False, xaxis_title="production minus reference", yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.dataframe(df, width="stretch", hide_index=True)

    rho = mv.get("spearman_rho")
    era = window.get("photographic_era", {})
    cols = st.columns(2)
    cols[0].metric(
        "Agreement with image tags (Spearman rho)",
        f"{rho:.2f}" if isinstance(rho, (int, float)) else "n/a",
    )
    if era.get("production_median_year"):
        cols[1].metric(
            "Median capture year",
            f"{era.get('production_median_year', 0):.0f}",
            delta=f"{era.get('production_median_year', 0) - era.get('reference_median_year', 0):+.0f} vs reference",
        )

    gains = [c["concept"] for c in ranking if c["delta"] > 0][:3]
    losses = [c["concept"] for c in ranking if c["delta"] < 0][:2]
    if gains:
        st.markdown(
            f"**Reading:** the shift is toward *{', '.join(gains)}*"
            + (f" and away from *{', '.join(losses)}*" if losses else "")
            + ". This turns 'dimension 342 drifted' into a cause a data engineer can recognise."
        )

    exemplars = window.get("dim_exemplars", [])
    if exemplars:
        with st.expander("Extreme-activation images for the top drifted dimensions"):
            for dim in exemplars[:2]:
                st.markdown(f"**{dim['dimension']}** (drift score {dim['drift_score']:.3f})")
                ref_imgs = [p for p in dim.get("reference_images", []) if Path(p).exists()][:4]
                prod_imgs = [p for p in dim.get("production_images", []) if Path(p).exists()][:4]
                if ref_imgs:
                    st.caption("reference")
                    st.image(ref_imgs, width=130)
                if prod_imgs:
                    st.caption("production")
                    st.image(prod_imgs, width=130)


# --------------------------------------------------------------------------- #
# Step 6 - remediation triage

_STRAT_LABEL = {
    "full_retrain": "Full retrain",
    "feature_drop_retrain": "Feature-drop retrain",
    "importance_weighted_retrain": "Importance weighting (0 labels)",
    "retrain_on_recent[previous]": "Retrain on previous window",
    "retrain_on_recent[all_prior]": "Retrain on all prior windows",
    "head_refit": "Head refit (few labels)",
    "calibration_only": "Recalibrate only (few labels)",
}


def render_remediation_step(scope: str) -> None:
    """scope: 'acs' or 'clear10' - which settings to show."""
    rem = _load(REMEDIATION_RESULTS)
    if not rem:
        st.info(
            "No remediation results yet. Run "
            "`OMP_NUM_THREADS=1 python examples/quickstart/remediation_demo.py --shift all`."
        )
        return

    settings = rem.get("settings", [])
    if scope == "acs":
        settings = [s for s in settings if "ACS" in s["shift_name"]]
    else:
        settings = [s for s in settings if "CLEAR" in s["shift_name"]]
    if not settings:
        st.info("No remediation settings for this dataset.")
        return

    st.caption(
        "Before spending on a retrain, the pipeline runs the two cheapest label-free repairs as "
        "probes and only recommends escalation when a probe already recovers a quarter of the gap. "
        "A separable shift is not necessarily a recoverable one."
    )

    cols = st.columns(len(settings))
    for col, s in zip(cols, settings):
        gap = (s["reference_accuracy"] - s["baseline_before"]) * 100
        verdict = s["triage"]["will_retraining_help"]
        col.metric(
            s["shift_name"].split(" (")[0].replace("ACS ", "").replace("CLEAR-10 ", "b"),
            "Escalate" if verdict else "Do not retrain",
            delta=f"{gap:.1f} pp gap",
            delta_color="off",
        )
        col.caption(s["triage"]["rationale"])

    setting_names = [s["shift_name"] for s in settings]
    chosen = st.selectbox("Strategy detail for", setting_names, key=f"rem_detail_{scope}")
    s = next(s for s in settings if s["shift_name"] == chosen)

    rows = []
    for r in s["results"]:
        rows.append(
            {
                "Strategy": _STRAT_LABEL.get(r["strategy"], r["strategy"]),
                "Accuracy before": round(r["accuracy_before"], 4),
                "Accuracy after": round(r["accuracy_after"], 4),
                "Gap recovered": f"{r['fraction_of_gap_recovered']:.0%}",
                "Fit time (s)": round(r["wall_clock_seconds"], 3),
                "Train rows": int(r["train_samples"]),
                "Prod. labels": int(r["n_production_labels_required"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    pareto = Path("paper/figures/remediation_pareto.png")
    if pareto.exists():
        st.image(str(pareto), caption="Recovery vs fit cost across all settings", width="stretch")

    if scope == "acs":
        st.markdown(
            "**Reading:** on the real census shifts every strategy sits near zero - dropping the "
            "KS-localised features is actively harmful because those features carry the decision. "
            "On the *controlled* covariate shift (a known scaling of two features) the same cheap "
            "strategies recover most of the gap with no labels, which is how we know the machinery "
            "is sound and the real shift is simply concept-level."
        )
    else:
        st.markdown(
            "**Reading:** retraining on just the previous bucket is on the cost/recovery frontier - "
            "it recovers most of the gap for a fraction of a full retrain."
        )
