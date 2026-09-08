"""Dashboard tab: the controlled drift-injection benchmark.

Self-contained, for a reader who knows nothing about the dataset or the
pipeline. Three sections:

  1. what the four injected drifts are and exactly what was done to the data,
  2. how the pipeline identified each one and how close its read was to the
     ground truth,
  3. what remediation it advised and whether that was the right call.

Reads ``outputs/drift_injection_benchmark.json`` (its ``presentation`` block
carries the dataset card, the drift recipes, and the raw material for the
before/after visuals).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BENCHMARK_RESULTS = Path("outputs/drift_injection_benchmark.json")

_ORDER = ["covariate", "prior", "concept", "label_noise"]
_PRETTY = {
    "none": "no drift", "covariate": "covariate shift", "prior": "prior shift",
    "concept": "concept drift", "label_noise": "label noise",
}
_COLOR = {"covariate": "#2ca02c", "prior": "#9467bd", "concept": "#d62728",
          "label_noise": "#1f77b4", "none": "#7f7f7f"}
# a retrain is the right call only for covariate shift
_SHOULD_ESCALATE = {"covariate": True, "prior": False, "concept": False, "label_noise": False}


def _load() -> Optional[dict]:
    try:
        return json.loads(BENCHMARK_RESULTS.read_text())
    except Exception:
        return None


def _runs_of(result: dict, dt: str) -> List[dict]:
    return sorted((r for r in result["runs"] if r["drift_type"] == dt),
                  key=lambda r: r["intensity_order"])


def _strongest(result: dict, dt: str) -> Optional[dict]:
    runs = _runs_of(result, dt)
    return runs[-1] if runs else None


# ======================================================================= #
# entry point

def render_injection_benchmark() -> None:
    result = _load()
    if not result:
        st.info(
            "No benchmark results yet. Run "
            "`python examples/quickstart/drift_injection_benchmark.py` "
            "(add `--quick` for a fast check)."
        )
        return

    pres = result.get("presentation", {})
    card = pres.get("dataset_card", {})
    summ = result["summary"]
    ti = summ["type_identification"]

    st.header("Verification benchmark - does the pipeline name the drift correctly?")
    st.markdown(
        "The other tabs run the pipeline on **real** drift, where nobody knows the ground truth. "
        "Here we **inject** drift we designed - four kinds, three severities each, plus a no-drift "
        "control - and check every stage of the pipeline against the recipe it never saw."
    )
    _dataset_card(card)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drift type identified", f"{ti['accuracy']:.0%}", f"{ti['n_runs']} windows", delta_color="off")
    c2.metric("Label-free estimate optimistic", f"{summ['estimator_optimism_rate']:.0%}",
              "under-reports every drop", delta_color="off")
    hit = summ.get("audit_localisation_hit_rate")
    c3.metric("Affected feature localised", f"{hit:.0%}" if hit is not None else "n/a",
              "covariate + concept", delta_color="off")
    rem_ok = sum(
        1 for dt in _ORDER
        if (s := _strongest(result, dt)) is not None
        and bool(s["triage"]["will_retraining_help"]) == _SHOULD_ESCALATE[dt]
    )
    c4.metric("Remediation verdict correct", f"{rem_ok} / {len(_ORDER)}",
              "vs the fix each drift needs", delta_color="off")

    st.divider()
    st.subheader("1 - What the four injected drifts are, and exactly what was done")
    _decomposition_matrix(pres)
    ref_pr = card.get("reference_positive_rate")
    for dt in _ORDER:
        _drift_card(result, pres, dt, ref_pr)

    st.divider()
    st.subheader("2 - How the pipeline identified each drift, and how close it got")
    _confusion(result)
    _estimate_accuracy_comparison(result)
    _per_type_identification(result)

    st.divider()
    st.subheader("3 - What remediation the pipeline advised, and whether it was right")
    _remediation(result)

    st.divider()
    _full_table(result)


# ======================================================================= #
# top matter

def _dataset_card(card: Dict[str, Any]) -> None:
    if not card:
        return
    with st.container(border=True):
        st.markdown(f"**Dataset — {card.get('name', 'ACS Income')}**")
        cols = st.columns([2, 1, 1, 1])
        cols[0].markdown(f"*Task:* {card.get('task', '')}")
        cols[1].metric("Reference rows", f"{card.get('n_reference', 0):,}")
        cols[2].metric("Reference accuracy", f"{card.get('reference_accuracy', 0):.3f}")
        cols[3].metric("Earn > $50k", f"{card.get('reference_positive_rate', 0):.0%}")
        st.caption(
            f"Model: {card.get('model', '')}. Reference window: {card.get('reference_window', '')}. "
            + card.get("how_it_works", "")
        )
        feats = card.get("features", [])
        if feats:
            with st.expander("The 10 input features"):
                st.dataframe(
                    pd.DataFrame(feats).rename(columns={"name": "feature", "meaning": "what it is"}),
                    width="stretch", hide_index=True,
                )


def _decomposition_matrix(pres: Dict[str, Any]) -> None:
    catalog = pres.get("drift_catalog", {})
    cols = ["P(X)\ninput distribution", "P(Y)\noutcome mix", "P(Y|X)\nthe rule"]
    moved = {
        "covariate": [1, 0, 0], "prior": [0, 1, 0],
        "concept": [0, 0, 1], "label_noise": [0, 0, 1],
    }
    rows = [dt for dt in _ORDER if dt in catalog] or _ORDER
    z = [moved[dt] for dt in rows]
    text = [["changed" if v else "unchanged" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=[_PRETTY[dt] for dt in rows],
        text=text, texttemplate="%{text}", showscale=False,
        colorscale=[[0, "#eef2f6"], [1, "#f2b8b5"]], xgap=3, ygap=3,
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                      title="Which part of the joint distribution each drift moves")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Covariate and prior shift move something a monitor can measure from inputs and predictions. "
        "Concept drift and label noise change only P(Y|X) or the labels - the inputs and the model's "
        "outputs are identical to the reference, so a label-free monitor is blind to them by "
        "construction."
    )


# ======================================================================= #
# section 1 - per drift-type card

def _drift_card(result: dict, pres: Dict[str, Any], dt: str, ref_positive_rate: Optional[float]) -> None:
    entry = pres.get("drift_catalog", {}).get(dt)
    if not entry:
        return
    with st.container(border=True):
        st.markdown(f"#### {entry['title']}")
        st.markdown(entry["plain"])
        a, b = st.columns(2)
        a.markdown(f"**What moves**  \n{entry['moves']}")
        b.markdown(f"**What stays fixed**  \n{entry['keeps']}")
        st.markdown(f"**Exactly what was done to the data:**  \n{entry['recipe']}")
        if entry.get("focus_feature"):
            st.caption(f"Focus feature below: **{entry['focus_feature']}** — {entry['focus_feature_meaning']}.")

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Injected settings and the resulting true accuracy drop**")
            it = entry.get("intensity_table", [])
            if it:
                st.dataframe(pd.DataFrame(it), width="stretch", hide_index=True)
        with right:
            _drift_signature_chart(dt, entry, ref_positive_rate)

    st.write("")


def _drift_signature_chart(dt: str, entry: Dict[str, Any], ref_positive_rate: Optional[float]) -> None:
    if dt == "covariate":
        _hist_overlay(entry, "How the age (AGEP) distribution shifts")
    elif dt == "prior":
        _prior_balance_chart(entry, ref_positive_rate)
    elif dt == "concept":
        _accuracy_by_region_chart(entry, "Model accuracy by education level (SCHL)")
    elif dt == "label_noise":
        _label_flip_chart(entry)


def _hist_overlay(entry: Dict[str, Any], title: str) -> None:
    edges = entry.get("focus_hist_bin_edges")
    ref = entry.get("focus_hist_reference")
    if not edges or not ref:
        return
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    fig = go.Figure()
    fig.add_bar(x=mids, y=_pct(ref), name="reference", marker_color="#c7cdd6")
    for it in entry.get("intensities", []):
        h = it.get("focus_hist")
        if h:
            fig.add_scatter(x=mids, y=_pct(h), mode="lines", name=it["label"])
    fig.update_layout(height=300, title=title, barmode="overlay",
                      xaxis_title=entry.get("focus_feature", ""), yaxis_title="share of rows (%)",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    st.caption("The reference bars are the population the model was built on; each line is a production "
               "window. Stronger injection pushes mass into ranges the model saw little of.")


def _prior_balance_chart(entry: Dict[str, Any], ref_positive_rate: Optional[float]) -> None:
    intensities = entry.get("intensities", [])
    labels = (["reference"] if ref_positive_rate is not None else []) + [it["label"] for it in intensities]
    vals = (([ref_positive_rate]) if ref_positive_rate is not None else []) + \
           [it["positive_rate"] for it in intensities]
    colors = (["#c7cdd6"] if ref_positive_rate is not None else []) + [_COLOR["prior"]] * len(intensities)
    fig = go.Figure(go.Bar(x=labels, y=[v * 100 for v in vals], marker_color=colors,
                           text=[f"{v:.0%}" for v in vals], textposition="outside"))
    if ref_positive_rate is not None:
        fig.add_hline(y=ref_positive_rate * 100, line_dash="dot",
                      annotation_text=f"reference {ref_positive_rate:.0%}")
    fig.update_layout(height=300, title="Share of the population earning over $50k",
                      yaxis_title="positive rate (%)", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Only the mix of outcomes changes; each person's features are untouched, so a model "
               "tuned to the old base rate mislabels more of the now-larger class.")


def _accuracy_by_region_chart(entry: Dict[str, Any], title: str) -> None:
    ref = entry.get("accuracy_by_decile_reference")
    if not ref:
        return
    fig = go.Figure()

    def _series(rows, name, color, dash=None):
        xs = [r["decile"] for r in rows if r["accuracy"] is not None]
        ys = [r["accuracy"] for r in rows if r["accuracy"] is not None]
        fig.add_scatter(x=xs, y=ys, mode="lines+markers", name=name,
                        line=dict(color=color, dash=dash))

    _series(ref, "reference (rule intact)", "#7f7f7f")
    for it in entry.get("intensities", []):
        if it.get("accuracy_by_decile"):
            _series(it["accuracy_by_decile"], f"{it['label']} injection", _COLOR["concept"],
                    dash={"mild": "dot", "moderate": "dash"}.get(it["label"]))

    fig.update_layout(height=300, title=title, xaxis_title="education (SCHL) decile - low to high",
                      yaxis_title="model accuracy", yaxis_range=[0, 1],
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")
    st.caption("The input distribution is byte-identical to the reference. Accuracy holds everywhere "
               "except the top education deciles, where the rule was inverted - a real loss that an "
               "input-only monitor cannot see, growing as more of the range is flipped.")


def _label_flip_chart(entry: Dict[str, Any]) -> None:
    rows = []
    for it in entry.get("intensities", []):
        rows.append({
            "intensity": it["label"],
            "labels left alone": it.get("unchanged_labels", 0),
            "low -> high earner": it.get("flips_neg_to_pos", 0),
            "high -> low earner": it.get("flips_pos_to_neg", 0),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for col, colr in [("labels left alone", "#c7cdd6"),
                      ("low -> high earner", "#1f77b4"),
                      ("high -> low earner", "#ff7f0e")]:
        fig.add_bar(x=df["intensity"], y=df[col], name=col, marker_color=colr)
    fig.update_layout(barmode="stack", height=300, title="How many production labels were flipped",
                      yaxis_title="rows", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    st.caption("Flips are uniformly random - they do not depend on any feature - so no region of the "
               "data can be blamed. The model's predictions never change; only the yardstick does.")


# ======================================================================= #
# section 2 - identification

def _confusion(result: dict) -> None:
    ti = result["summary"]["type_identification"]
    labels = ti["labels"]
    cm = ti["confusion_matrix"]
    z = [[cm[a].get(b, 0) for b in labels] for a in labels]
    fig = go.Figure(go.Heatmap(
        z=z, x=[_PRETTY[l] for l in labels], y=[_PRETTY[l] for l in labels],
        text=z, texttemplate="%{text}", colorscale="Blues", showscale=False, xgap=2, ygap=2,
        hovertemplate="injected %{y}<br>verdict %{x}<br>%{z} runs<extra></extra>",
    ))
    fig.update_layout(height=340, xaxis_title="pipeline verdict", yaxis_title="injected drift",
                      title=f"Injected drift vs the pipeline's verdict  (accuracy {ti['accuracy']:.0%})",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    if ti["accuracy"] >= 1.0:
        st.success(
            f"All {ti['n_runs']} injected windows were labelled with their true drift type. Covariate "
            "and prior are resolved from label-free signals alone; concept drift and label noise are "
            "silent to the label-free monitors and are pinned down by a small delayed-label audit."
        )


def _estimate_accuracy_comparison(result: dict) -> None:
    st.markdown("**How close was the pipeline's accuracy estimate to the truth?**")
    fig = go.Figure()
    diag = [0.4, 0.85]
    fig.add_scatter(x=diag, y=diag, mode="lines", line=dict(dash="dash", color="#999"),
                    name="perfect estimate", hoverinfo="skip")
    lf_x, lf_y, au_x, au_y, txt = [], [], [], [], []
    for r in result["runs"]:
        e = r["grading"]["estimator"]
        true_acc = e["true_accuracy"]
        lf_x.append(true_acc); lf_y.append(e["estimated_accuracy"])
        au_x.append(true_acc); au_y.append(r["labelled_audit"]["measured_accuracy"])
        txt.append(f"{_PRETTY[r['drift_type']]} / {r['intensity_label']}")
    fig.add_scatter(x=lf_x, y=lf_y, mode="markers", name="label-free estimate (no labels)",
                    marker=dict(symbol="x", size=9, color="#d62728"), text=txt,
                    hovertemplate="%{text}<br>true %{x:.3f}<br>label-free est %{y:.3f}<extra></extra>")
    fig.add_scatter(x=au_x, y=au_y, mode="markers", name="delayed 600-label audit",
                    marker=dict(size=9, color="#2ca02c"), text=txt,
                    hovertemplate="%{text}<br>true %{x:.3f}<br>audit est %{y:.3f}<extra></extra>")
    fig.update_layout(height=420, xaxis_title="true accuracy (measured with held-out labels)",
                      yaxis_title="pipeline's accuracy estimate",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Each point is one production window. The label-free estimate (red x) sits above the diagonal "
        f"in {result['summary']['estimator_optimism_rate']:.0%} of windows - it under-reports the drop "
        f"every time, and for covariate shift it even reports an accuracy *gain*. The delayed audit "
        f"(green dot) lands on the diagonal: a few hundred real labels resolve what the label-free "
        f"stack cannot."
    )


def _per_type_identification(result: dict) -> None:
    st.markdown("**Per drift type: the evidence, the verdict, and the gap to ground truth**")
    tabs = st.tabs([_PRETTY[dt] for dt in _ORDER])
    for tab, dt in zip(tabs, _ORDER):
        with tab:
            _identification_detail(result, dt)


def _identification_detail(result: dict, dt: str) -> None:
    runs = _runs_of(result, dt)
    strong = runs[-1]
    sig = strong["signature"]
    v = strong["verdict"]
    aud = strong["labelled_audit"]
    e = strong["grading"]["estimator"]
    loc = strong["grading"]["localisation"]

    # --- evidence bars -------------------------------------------------
    signals = [
        ("peak feature KS statistic", sig["max_localised_ks_statistic"] / 0.5),
        ("reference/production separability (AUC-0.5)", max(sig["domain_classifier_auc"] - 0.5, 0) / 0.5),
        ("confidence-shift (CBPE)", sig["cbpe_confidence_shift"] / 0.15),
        ("model's own positive-rate shift", sig["predicted_prior_shift"] / 0.35),
        ("reliability out-of-distribution", _none0(sig["reliability_mean_ood"])),
        ("reliability high-risk share", _none0(sig["reliability_high_risk_pct"]) / 20.0),
        ("delayed audit: measured drop", max(aud["measured_drop"], 0) / 0.35),
        ("delayed audit: loss concentrated in a feature", max(aud["structure_score"], 0) / 0.35),
    ]
    names = [s[0] for s in signals][::-1]
    vals = [min(max(s[1], 0), 1) for s in signals][::-1]
    is_audit = [("audit" in s[0]) for s in signals][::-1]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=["#8c6bb1" if a else _COLOR[dt] for a in is_audit],
    ))
    fig.add_vline(x=0.3, line_dash="dot", line_color="#999", annotation_text="noise floor")
    fig.update_layout(height=340, xaxis_range=[0, 1], xaxis_title="signal strength (normalised)",
                      title=f"What the pipeline saw at the strongest {_PRETTY[dt]}",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Purple bars are the delayed-label audit; coloured bars are label-free. For concept "
               "drift and label noise every label-free bar is on the floor - only the audit fires.")

    box = st.success if v["correct"] else st.error
    box(f"**Verdict: {_PRETTY[v['predicted_type']]}** "
        f"({'correct' if v['correct'] else 'WRONG - injected ' + _PRETTY[dt]}), "
        f"resolved from the {v['stage']}.  \n{v['rationale']}")

    # --- closeness to ground truth ----------------------------------
    inj_feats = ", ".join(strong["ground_truth"]["affected_features"]) or "(none - not feature-localised)"
    ks_feats = ", ".join(loc.get("matched", [])) or "(none)"
    audit_feat = loc.get("audit_feature") or "-"
    lf_off = e["estimated_accuracy"] - e["true_accuracy"]
    au_off = aud["measured_accuracy"] - e["true_accuracy"]
    rows = [
        ["drift type", _PRETTY[dt], _PRETTY[v["predicted_type"]],
         "match" if v["correct"] else "MISS"],
        ["affected feature(s)", inj_feats,
         f"KS: {ks_feats}   |   audit: {audit_feat}",
         "match" if (loc.get("f1") == 1.0 or loc.get("audit_match")) else
         ("n/a" if not strong["ground_truth"]["affected_features"] else "partial")],
        ["accuracy at strongest intensity", f"{e['true_accuracy']:.3f} (true)",
         f"label-free {e['estimated_accuracy']:.3f} ({lf_off:+.3f})   |   "
         f"audit {aud['measured_accuracy']:.3f} ({au_off:+.3f})",
         "audit within 1pp" if abs(au_off) <= 0.015 else "see gap"],
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["dimension", "ground truth", "pipeline's read", "outcome"]),
        width="stretch", hide_index=True,
    )

    # --- intensity tracking ---------------------------------------
    xs = [r["intensity_label"] for r in runs]
    fig2 = go.Figure()
    fig2.add_scatter(x=xs, y=[r["ground_truth"]["true_accuracy_drop"] for r in runs],
                     mode="lines+markers", name="true drop (labels)", line=dict(color="#333"))
    fig2.add_scatter(x=xs, y=[r["labelled_audit"]["measured_drop"] for r in runs],
                     mode="lines+markers", name="delayed audit", line=dict(color="#2ca02c"))
    fig2.add_scatter(x=xs, y=[abs(r["grading"]["estimator"]["implied_drop"]) for r in runs],
                     mode="lines+markers", name="label-free implied drop",
                     line=dict(color="#d62728", dash="dash"))
    fig2.update_layout(height=300, title="Does the pipeline's read scale with the injected severity?",
                       yaxis_title="accuracy drop", margin=dict(l=10, r=10, t=40, b=10),
                       legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig2, width="stretch")


# ======================================================================= #
# section 3 - remediation

def _remediation(result: dict) -> None:
    rows = []
    for dt in _ORDER:
        strong = _strongest(result, dt)
        tri = strong["triage"]
        ok = bool(tri["will_retraining_help"]) == _SHOULD_ESCALATE[dt]
        rows.append({
            "injected drift": _PRETTY[dt],
            "pipeline's advice": "escalate a retrain" if tri["will_retraining_help"] else "do not retrain",
            "the fix this drift actually needs": strong["ground_truth"]["expected_diagnosis"],
            "verdict": "correct" if ok else "WRONG",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # why: cheap probe recovery
    labels, recov, aucs, sens = [], [], [], []
    for dt in _ORDER:
        strong = _strongest(result, dt)
        s = strong["triage"].get("signals", {})
        labels.append(_PRETTY[dt])
        recov.append(100 * float(s.get("cheap_probe_recovery", {}).get("importance_weighted_retrain", 0.0)))
        aucs.append(float(s.get("domain_classifier_auc", 0.5)))
        sens.append(float(s.get("feature_drop_sensitivity", 0.0)))

    fig = go.Figure(go.Bar(x=labels, y=recov, marker_color=[_COLOR[d] for d in _ORDER],
                           text=[f"{v:.0f}%" for v in recov], textposition="outside"))
    fig.add_hline(y=25, line_dash="dot", annotation_text="escalation threshold (25%)")
    fig.update_layout(height=320, yaxis_title="% of the accuracy gap recovered",
                      title="Why triage decides as it does: what the cheapest label-free repair recovers",
                      margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The triage runs importance-weighted retraining (needs no labels) as a probe. It recovers most "
        "of the gap only for covariate shift, so that is the only case where a retrain is escalated. "
        "For prior shift the fix is threshold recalibration; for concept drift and label noise a retrain "
        "on the old labels cannot help, and triage correctly refuses to recommend one."
    )

    with st.expander("The signals behind each verdict"):
        sig_df = pd.DataFrame({
            "injected drift": labels,
            "reference/production separability (AUC)": [round(a, 2) for a in aucs],
            "importance-weighting recovery": [f"{v:.0f}%" for v in recov],
            "feature-drop sensitivity": [round(x, 3) for x in sens],
        })
        st.dataframe(sig_df, width="stretch", hide_index=True)
        for dt in _ORDER:
            strong = _strongest(result, dt)
            st.markdown(f"- **{_PRETTY[dt]}** — {strong['triage']['rationale']}")


# ======================================================================= #
# full table

def _full_table(result: dict) -> None:
    with st.expander("Every injected window (all types x all intensities)"):
        rows = []
        for r in result["runs"]:
            gt = r["ground_truth"]
            v = r["verdict"]
            e = r["grading"]["estimator"]
            rows.append({
                "drift": _PRETTY[r["drift_type"]],
                "intensity": r["intensity_label"],
                "true drop": round(gt["true_accuracy_drop"], 3),
                "label-free est. error": round(e["signed_error"], 3),
                "audit drop": r["labelled_audit"]["measured_drop"],
                "rel. AUROC": r["pipeline"]["reliability"].get("auroc_risk_vs_error"),
                "verdict": _PRETTY[v["predicted_type"]],
                "resolved by": v["stage"],
                "correct": "yes" if v["correct"] else "NO",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ======================================================================= #
# helpers

def _pct(counts: List[float]) -> List[float]:
    total = float(sum(counts)) or 1.0
    return [100.0 * c / total for c in counts]


def _none0(v: Any) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0
