"""Data Drift Autopsy dashboard.

Four tabs, each told in pipeline order. The three real drift studies (ACS,
CLEAR-10, gas sensor) follow the architecture (setup and results; proxy
performance, detection and localization; root cause and drift type;
reliability; remediation triage). The benchmark follows the injection
procedure and ends with the comparison against baselines. Selectors at the top
of a tab choose the model family (and the ACS task or benchmark
configuration); they never hide windows. The dashboard only reads the study
outputs; every number and every "What the data shows" sentence is computed
from them. Run with:  streamlit run examples/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts as C  # noqa: E402
import loaders as L  # noqa: E402
import narrative as N  # noqa: E402

CAM_DIR = Path("outputs/figures/cam")

st.set_page_config(page_title="Data Drift Autopsy", layout="wide")
st.markdown("""
<style>
.step-q {opacity: 0.8; margin-top: -0.6rem; margin-bottom: 0.6rem;}
.chip {display:inline-block; padding: 0.05rem 0.5rem; border-radius: 0.6rem; font-size: 0.85rem;
       border: 1px solid rgba(128,128,128,0.45);}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# small layout helpers

_TAB = {"prefix": ""}          # chart keys must be unique across tabs


def plot(fig, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=f"{_TAB['prefix']}_{key}", config={"displaylogo": False})


def step(number: str, title: str, question: str) -> None:
    st.markdown(f"### {number} {title}")
    st.markdown(f"<div class='step-q'>{question}</div>", unsafe_allow_html=True)


def sub(title: str, how: str) -> None:
    st.markdown(f"#### {title}")
    st.caption(how)


def shows(text: str) -> None:
    with st.container(border=True):
        st.markdown(f"**What the data shows.** {text}")


@st.cache_data(show_spinner=False)
def thumbnail(path: str, size: int = 360):
    """Centre-cropped square copy so a row of images lines up (display only)."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return ImageOps.fit(im.convert("RGB"), (size, size))


def missing(name: str, script: str) -> None:
    st.info(f"{name} not found. Run `python {script}` first.")


@st.cache_data(show_spinner="Loading results...")
def load(path: str):
    return L.load(Path(path))


def model_selector(key: str, available) -> str:
    options = [m for m in ("lr", "hgb") if m in available]
    if len(options) < 2:
        return options[0] if options else "lr"
    return st.radio("Monitored model", options, format_func=L.MODEL_LABEL.get, horizontal=True, key=key,
                    help="Both model families are monitored by exactly the same pipeline.")


def pipeline_dot(stages, note: str = "") -> str:
    nodes = "\n".join(f'  s{i} [label="{i + 1}. {s}"];' for i, s in enumerate(stages))
    edges = " -> ".join(f"s{i}" for i in range(len(stages)))
    return f"""digraph {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#cde2fb", color="#2a78d6", fontcolor="#0d366b",
        fontname="Helvetica", fontsize=10, margin="0.12,0.06"];
  edge [color="#8c8b86", arrowsize=0.7];
{nodes}
  {edges};
}}"""


ARCH_STAGES = ["Setup and results", "Proxy estimate\\n+ detection\\n+ localization", "Root cause\\n+ drift type",
               "Reliability", "Remediation triage"]

GLOSSARY = {
    "AUROC": "area under the ROC curve: the chance a random wrong prediction scores higher than a random right one (0.5 = chance).",
    "BBSE": "black-box shift estimation (Lipton et al., 2018): estimates the new class prior from the predicted labels.",
    "CAM": "class activation map: which image regions push the model toward its predicted class.",
    "CBPE": "confidence-based performance estimation: accuracy estimated from model confidence, without labels.",
    "CI": "confidence interval (95% unless stated).",
    "CLIP": "a vision-language model; used here only to score images against plain-language concepts.",
    "Holm / Bonferroni": "corrections that keep the chance of any false alarm at α across many tests.",
    "KS": "Kolmogorov-Smirnov test: largest gap between two cumulative distributions (D).",
    "MMD": "maximum mean discrepancy: a kernel distance between two multivariate samples, tested by permutation.",
    "OOD": "out of distribution: how unlike the reference an input is.",
    "PCA": "principal component analysis.",
    "PSI": "population stability index; 0.2 is the usual convention for a material shift.",
    "SHAP": "Shapley additive explanations: each feature's contribution to the predicted probability.",
    "pp": "percentage points.",
}


def glossary() -> None:
    with st.expander("Glossary"):
        for k, v in GLOSSARY.items():
            st.markdown(f"**{k}**: {v}")


# --------------------------------------------------------------------------- #
# shared steps (ACS and CLEAR-10)

def overview(study, summary, unit: str, intro: str) -> None:
    st.markdown(intro)
    control = summary[summary["type"] == "control"]
    det = L.detector_table(study)
    ctrl_alarms = det[det["key"].isin(control["key"]) & det["drift"]]
    drift = summary[summary["type"] != "control"]
    c = st.columns(4)
    c[0].metric("Reference accuracy (held out)", f"{study['reference']['test_accuracy']:.3f}")
    c[1].metric("Production windows", len(drift))
    c[2].metric("Significant accuracy drops", f"{int(drift['drop significant'].sum())} of {len(drift)}")
    c[3].metric("Alarms on the no-drift control", "none ✓" if ctrl_alarms.empty else f"{len(ctrl_alarms)} ✗")
    st.graphviz_chart(pipeline_dot(ARCH_STAGES), width="content")


def focus_selector(summary, key: str) -> str:
    default = L.focus_window(summary)
    keys = list(summary["key"])
    labels = dict(zip(summary["key"], summary["window"]))
    col, note = st.columns([1, 3])
    chosen = col.selectbox("Focus window for the detail charts", keys, index=keys.index(default),
                           format_func=lambda k: labels[k], key=key)
    note.caption(f"Default: {labels[default]}, the window with the largest significant accuracy drop. "
                 "Overview charts always show every window; detail charts follow this choice.")
    return chosen


def setup_step(study, summary, unit: str, facts: dict) -> None:
    alpha = study["config"]["alpha"]
    step("1.", "Initial setup and results",
         "How was the model built, what is compared with what, and how much did accuracy really change?")
    left, right = st.columns([3, 2])
    with left:
        sub("How the reference data is split",
            "The reference period is split once, stratified on the label. Hover a part to see its role.")
        plot(C.split_bar(L.split_sizes(study), facts["reference"]), "split")
        plot(C.window_sizes(summary, unit), "sizes")
    with right:
        sub("Study facts", "Read from the run configuration.")
        st.markdown("\n".join(f"- **{k}**: {v}" for k, v in facts.items()))
    sub("Measured accuracy (evaluation only, needs labels)",
        "Production labels are used here only to evaluate the pipeline, never by it. Bars are 95% Wilson "
        f"intervals; a filled marker is a significant drop (one-sided two-proportion test, α = {alpha}).")
    plot(C.accuracy_with_ci(summary, study["reference"]["test_accuracy"]), "acc")
    shows(N.accuracy(summary, alpha))


def proxy_step(study, summary, focus, focus_label) -> None:
    step("2.", "Proxy performance estimation, drift detection and drift localization",
         "Without labels: is accuracy likely falling, did the data change, and where?")
    sub("2a. Label-free accuracy estimate (CBPE)",
        "CBPE maps each prediction's confidence to the accuracy that confidence had on the calibration split and "
        "averages over the window. It assumes that confidence-to-accuracy relation does not change.")
    a, b = st.columns([3, 2])
    with a:
        plot(C.estimate_vs_measured(summary, study["reference"]["test_accuracy"]), "est")
    with b:
        plot(C.estimate_error(summary), "est_err")
    shows(N.proxy(summary))
    a, b = st.columns(2)
    with a:
        plot(C.confidence_hist(L.confidence_histogram(study, focus), focus_label), "conf")
        shows(N.confidence(L.cbpe_test(study, focus), focus_label))
    if "proxy_classwise" in study["windows"][focus]:
        with b:
            plot(C.classwise_f1(L.clear_classwise_f1(study, focus), focus_label), "f1")


def detection_step(study, perms: int) -> pd.DataFrame:
    alpha = study["config"]["alpha"]
    det = L.detector_table(study)
    sub("2b. Drift detection: how each test behaves",
        f"Every detector compares the window with the reference window and decides by a test at α = {alpha}: "
        "KS on each numeric input (Bonferroni across inputs), PSI against the 0.2 convention, MMD by a permutation "
        f"test ({perms} permutations), CBPE by a chi-square test on the confidence histogram"
        + (", PCA reconstruction and Fréchet distance by permutation tests" if "bucket" in next(iter(study["windows"].values())) else "")
        + ". Two baselines from Rabanser et al. (2019) run alongside: BBSD (KS on each class probability) and a "
        "domain classifier (a classifier two-sample test). Each panel has its own scale because the statistics "
        "measure different things. Filled marker = alarm.")
    plot(C.detector_small_multiples(det, L.DETECTOR_STATISTIC, study["config"]["psi_threshold"]), "det_sm")
    note = N.sample_size_note(L.window_summary(study), det)
    if note:
        st.caption(note)
    plot(C.alarm_matrix(det), "det_matrix")
    shows(N.detection(det))
    return det


def diagnosis_step(study, summary, audit_labels, title: str = "3b. Drift type") -> None:
    sub(title,
        "Label free first: input tests, a prediction test and MMD under Holm. BBSE then reweights the reference by "
        "the estimated class prior; shift that survives the reweighting is covariate shift, shift it removes is prior "
        f"shift. Only when nothing shifts does a delayed-label audit ({audit_labels} labels) decide: an accuracy drop whose errors depend "
        "on a feature is concept drift, one explained by confidence alone is label noise. Hover a verdict for its reason.")
    plot(C.decision_matrix(L.decision_path(study)), "decision")
    shows(N.diagnosis(summary))


def reliability_step(study, focus, focus_label, n: str = "4.") -> None:
    step(n, "Per-prediction reliability", "Which individual predictions should not be trusted?")
    fusion = study["reliability_model"]["fusion"]
    st.caption("Each prediction gets three signals: low confidence, an out of distribution score and instability under "
               "small input noise. A logistic regression fitted on the calibration split fuses them into an estimated "
               f"probability of error. LOW means below the reference error rate ({fusion['label_thresholds']['LOW_below']:.3f}); "
               f"HIGH means {fusion['label_thresholds']['HIGH_from']} or more.")
    rel = L.reliability_summary(study)
    coefs = L.fusion_coefficients(study)
    plot(C.auroc_dots(rel), "auroc")
    st.caption("Where the fused marker sits on the confidence marker, fusion adds nothing beyond confidence.")
    a, b = st.columns(2)
    with a:
        plot(C.risk_shares(rel), "risk_share")
    with b:
        plot(C.risk_error_rates(rel), "risk_err")
    a, b = st.columns([3, 2])
    with a:
        plot(C.risk_hist(L.reliability_records(study, focus), fusion["label_thresholds"], focus_label), "risk_hist")
        st.caption(f"Based on the {len(L.reliability_records(study, focus)):,} predictions stored for this window.")
    with b:
        if coefs.empty:
            st.info(f"No fusion weights learned: the calibration split holds only {fusion.get('n_calibration_errors')} "
                    f"wrong predictions, fewer than the {fusion.get('min_events')} a three-signal logistic model needs "
                    "(ten per signal). The risk score is the confidence risk (1 - confidence) alone.")
        else:
            plot(C.fusion_bars(coefs), "fusion")
    shows(N.reliability(rel, coefs))


def remediation_step(rem, groups, n: str = "5.") -> None:
    step(n, "Remediation triage", "Will a cheap retrain help, and which repair actually works?")
    if rem is None:
        missing("Remediation results", "examples/quickstart/remediation_demo.py")
        return
    st.caption("Six repair strategies are graded on a labelled half of each window with a paired bootstrap "
               "(★ = significant gain). The triage decides without production labels: it validates two cheap probes "
               "(importance-weighted retrain on reweighted reference data; retrain on the latest delayed-label window "
               "on its held-out part) and escalates only if a predicted gain is significant (Holm across probes).")
    table = L.remediation_table(rem, groups)
    card = L.triage_scorecard(rem, groups)
    if table.empty:
        missing("Remediation results for this study", "examples/quickstart/remediation_demo.py")
        return
    sub("Accuracy recovered by each strategy (percentage points)",
        "Blue = gain, red = loss, gray = no change; n/a = the strategy needs data this setting does not have (hover).")
    plot(C.remediation_heatmap(table), "rem_heat")
    sub("Did the triage call it right?",
        "Truth = a probe strategy significantly raises accuracy on the labelled holdout.")
    plot(C.triage_scorecard(card), "triage_card")
    plot(C.probe_scatter(L.triage_probes(rem, groups)), "probe")
    shows(N.remediation(table, card))


# --------------------------------------------------------------------------- #
# tabs

TASK_QUESTION = {"income": "earns more than $50,000", "employment": "aged 16 to 90 is employed"}
DENSE_MODEL = {"lr": "L2 logistic regression (C=1) on standardised features",
               "hgb": "histogram gradient boosting (scikit-learn defaults, no early stopping)"}


def acs_tab() -> None:
    _TAB["prefix"] = "acs"
    c1, c2 = st.columns(2)
    with c1:
        task = st.radio("Folktables task", ["income", "employment"], format_func=L.TASK_LABEL.get, horizontal=True,
                        key="acs_task")
    available = [m for m in ("lr", "hgb") if L.study_path("acs", m, task).exists()]
    if not available:
        missing(f"{L.TASK_LABEL[task]} results", f"examples/quickstart/folktables_demo.py --task {task}")
        return
    with c2:
        model = model_selector("acs_model", available)
    acs = load(str(L.study_path("acs", model, task)))
    rem = load(str(L.study_path("remediation", model)))
    _TAB["prefix"] = f"acs_{task}_{model}"
    summary = L.window_summary(acs)
    cfg = acs["config"]
    temporal = summary.loc[summary["type"] == "temporal", "window"].tolist()
    geo = summary.loc[summary["type"] == "geographic", "window"].tolist()
    gaps = summary.loc[summary["gap_before"], "window"].tolist()
    overview(acs, summary, "rows",
             f"**{cfg.get('task_title', 'ACS Income')}** (US census survey): predict whether a person "
             f"{TASK_QUESTION[task]}. Reference: **{cfg['reference']}**. "
             f"Temporal drift: {N.join(temporal)}. Geographic drift: {N.join(geo)}. The held-out part of the reference "
             "is a no-drift control."
             + (f" There is no standard 2020 release, so lines are not joined before {N.join(gaps)}." if gaps else ""))
    focus = focus_selector(summary, f"acs_focus_{task}_{model}")
    focus_label = summary.set_index("key").loc[focus, "window"]
    feats = cfg["features"]
    setup_step(acs, summary, "rows", {
        "reference": cfg["reference"],
        "model": cfg["model"],
        "numeric features": ", ".join(L.feature_label(f) for f in feats["numeric"]),
        "categorical features": ", ".join(L.feature_label(f) for f in feats["categorical"]),
        "significance level α": cfg["alpha"],
        "delayed labels in the audit": cfg["audit_labels"],
    })
    st.divider()
    proxy_step(acs, summary, focus, focus_label)
    detection_step(acs, cfg["mmd"]["n_permutations"])
    n_in, n_num = len(feats["numeric"]) + len(feats["categorical"]), len(feats["numeric"])
    sub("2c. Drift localization: which inputs moved",
        "Each input is tested on its own (KS for numeric, chi-square homogeneity for categorical) with Holm correction. "
        "Two maps because the effect sizes differ: KS D for numeric inputs, Cramér's V for categorical ones. * = significant. "
        f"Holm here runs over all {n_in} inputs, so it is stricter than the KS detector above, which corrects over the "
        f"{n_num} numeric inputs only; a window can therefore raise a KS alarm without a starred numeric input.")
    loc = L.acs_localization(acs)
    plot(C.feature_heatmaps(loc), "loc")
    fig = C.unseen_categories(loc)
    if fig is not None:
        plot(fig, "unseen")
    shows(N.acs_localization(loc, focus) + " " + N.unseen_overall(loc))
    st.divider()

    step("3.", "Root cause analysis and drift type", "Why did the model's behaviour change, and what kind of drift is it?")
    shap_all = L.acs_shap_all(acs)
    if shap_all.empty:
        sub("3a. Change in what the model relies on (SHAP)", "")
        st.info("SHAP is computed for the logistic model only: the exact explainer evaluates every coalition of the "
                "inputs for each explained row, which the boosted model's slower predictions turn into hours per window.")
    else:
        meta = acs["windows"][focus]["pipelines"]["KS Test"]["rca"]["explanations"]
        sub("3a. Change in what the model relies on (SHAP)",
            f"Mean |SHAP| on the predicted probability, {meta['n_reference_samples']} reference rows vs "
            f"{meta['n_test_samples']} window rows. A change is significant when its Bonferroni-adjusted bootstrap "
            "interval excludes zero.")
        plot(C.shap_heatmap(shap_all), "shap_heat")
        shap_focus = L.acs_shap(acs, focus)
        plot(C.shap_dumbbell(shap_focus, focus_label), "shap_db")
        shows(N.shap(shap_focus, focus_label))
        recs = L.acs_recommendations(acs, focus)
        if recs:
            with st.expander(f"Generated notes for {focus_label}"):
                for r in recs:
                    st.markdown(f"- {r}")
    diagnosis_step(acs, summary, cfg["audit_labels"])
    st.caption("BBSE estimates the positive rate from the predicted labels. It is valid only when the class prior is "
               "the only change (prior shift); under covariate shift it is not, which is why the diagnosis tests that "
               "assumption before naming a prior shift.")
    plot(C.prior_lines(L.prior_rates(acs)), "prior")
    shows(N.prior(L.prior_rates(acs), focus))
    st.divider()
    reliability_step(acs, focus, focus_label)
    st.divider()
    group = "acs" if task == "income" else "acs_employment"
    remediation_step(rem, (f"{group}_temporal", f"{group}_geographic"))
    glossary()


def cam_section(cam) -> None:
    sub("3a. Where the model looks (class activation maps)",
        "For the linear head on frozen ResNet-18 features the class activation map is exact: its average equals the "
        "class score. Each map is scaled within its own image; red = most evidence for the predicted class, blue = least.")
    if cam is None:
        missing("CAM images", "examples/paper/build_artifacts.py")
        return
    st.caption(f"Images: {cam['selection_rule']}. Why these: the most unusual inputs are where a shift should show.")
    cols = st.columns(2 * len(cam["images"]), gap="small")
    for j, im in enumerate(cam["images"]):
        ok = "✓" if im["predicted"] == im["true"] else "✗"
        cols[2 * j].image(str(CAM_DIR / im["image_png"]), width="stretch")
        cols[2 * j + 1].image(str(CAM_DIR / im["overlay_png"]), width="stretch")
        cols[2 * j].caption(f"**predicted {im['predicted']}** {ok}  \ntrue {im['true']}")
        cols[2 * j + 1].caption(f"OOD score {im['ood_unclipped']:.1f}")
    try:
        from matplotlib import colormaps
        cmap = colormaps[cam["colormap"]]
        stops = ", ".join(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
                          for r, g, b, _ in (cmap(i / 10) for i in range(11)))
        st.markdown(f"<div style='display:flex;align-items:center;gap:0.5rem;max-width:420px'>low"
                    f"<div style='flex:1;height:12px;border-radius:6px;background:linear-gradient(to right,{stops})'></div>"
                    f"high</div>", unsafe_allow_html=True)
    except Exception:
        pass


def concept_section(concepts, focus_bucket: str, alpha: float) -> None:
    sub("3b. Concept-level root cause (CLIP concept probe)",
        f"CLIP scores each image against {len(concepts['config']['vocabulary'])} plain-language concepts. The share of "
        "each concept is compared with bucket 1 (Bonferroni bootstrap intervals). As a check, matched CLEAR image tags "
        "are compared by Spearman correlation with a permutation p-value.")
    val = L.concept_validation(concepts)
    a, b = st.columns(2)
    with a:
        plot(C.concept_count_line(val), "c_count")
    with b:
        plot(C.concept_rho_line(val, alpha), "c_rho")
    shows(N.concepts(val, alpha))
    buckets = list(concepts["buckets"])
    b_sel = focus_bucket if focus_bucket in buckets else str(concepts["config"]["figure_bucket"])
    df = L.concept_shifts(concepts, b_sel)
    df = pd.concat([df.head(6), df.tail(8)]).drop_duplicates("concept")
    a, b = st.columns(2)
    with a:
        plot(C.interval_bars(df, "concept", "delta", "ci_low", "ci_high", "significant",
                             f"Largest concept shifts, bucket {b_sel} vs bucket 1 (filled = significant)",
                             "change in concept share"), "c_bars")
    with b:
        v = concepts["buckets"][b_sel]["metadata_validation"]
        plot(C.concept_tag_scatter(L.concept_tag_pairs(concepts, b_sel), v["spearman_rho"], v["p_value"], b_sel), "c_tags")
    fb = str(concepts["config"]["figure_bucket"])
    entry = concepts["buckets"][fb]
    ex = entry.get("exemplars")
    if ex:
        focus = entry["focus_class"]["class"]
        sub(f"Visual check: does '{ex['concept']}' grow inside class {focus}?",
            f"Bucket {fb} is shown because it has the largest Fréchet distance to the reference; {focus} is the class "
            f"with the most shifted dimensions there; '{ex['concept']}' is its top significant gaining concept. The bars "
            "give the share of that class's images whose best-matching concept it is. The images are the "
            f"{len(ex['reference']['images'])} strongest matches on each side, picked by the same rule, so a person can "
            "see whether the shift is real.")
        plot(C.share_bars(ex["reference"]["share_top_concept"], ex["production"]["share_top_concept"],
                          "bucket 1", f"bucket {fb}", ex["concept"]), "c_share")
        for side, label in (("reference", "bucket 1"), ("production", f"bucket {fb}")):
            st.markdown(f"**{label}**")
            cols = st.columns(len(ex[side]["images"]))
            for col, path, p in zip(cols, ex[side]["images"], ex[side]["probabilities"]):
                if Path(path).exists():
                    col.image(thumbnail(path), caption=f"CLIP probability {p:.2f}", width="stretch")
        shows(N.exemplars(ex, focus, fb))


def clear_tab(concepts, cam) -> None:
    available = [m for m in ("lr", "hgb") if L.study_path("clear", m).exists()]
    if not available:
        missing("CLEAR-10 results", "examples/quickstart/clear10_drift_study.py")
        return
    model = model_selector("clear_model", available)
    clear = load(str(L.study_path("clear", model)))
    rem = load(str(L.study_path("remediation", model)))
    _TAB["prefix"] = f"clear_{model}"
    summary = L.window_summary(clear)
    cfg = clear["config"]
    n_buckets = int((summary["type"] != "control").sum())
    overview(clear, summary, "images",
             f"**CLEAR-10**: photos of {len(clear['class_names']) - 1} object classes plus background, in "
             f"{n_buckets + 1} buckets ordered by time. Reference: **bucket {cfg['reference_bucket']}**. Production: buckets 2 to "
             f"{n_buckets + 1}. The held-out part of bucket 1 is a no-drift control.")
    focus = focus_selector(summary, "clear_focus")
    focus_label = summary.set_index("key").loc[focus, "window"]
    focus_bucket = str(clear["windows"][focus]["bucket"])
    setup_step(clear, summary, "images", {
        "reference": f"bucket {cfg['reference_bucket']}",
        "representation": cfg["representation"],
        "model": cfg["model"],
        "classes": len(clear["class_names"]),
        "significance level α": cfg["alpha"],
        "delayed labels in the audit": cfg["audit_labels"],
    })
    st.divider()
    proxy_step(clear, summary, focus, focus_label)
    detection_step(clear, cfg["n_permutations"])
    sub("2c. Drift localization: which embedding dimensions moved",
        "Each of the 512 dimensions is tested with KS and Holm correction, over all images and within each predicted "
        "class (label free).")
    loc, slices = L.clear_localization(clear), L.clear_class_slices(clear)
    a, b = st.columns([2, 3])
    with a:
        plot(C.clear_dims(loc), "dims")
    with b:
        plot(C.class_slice_heatmap(slices), "slices")
    shows(N.clear_localization(loc, slices, focus))
    proj = L.clear_projection(clear)
    if proj is not None:
        st.markdown("**The buckets in principal component space**")
        st.caption(f"PCA fitted on the {proj['fitted_on']}; {proj['n_per_window']} random images per bucket. Drag to rotate.")
        a, b = st.columns(2)
        with a:
            plot(C.pca_points(proj, int(focus_bucket)), "pca_pts")
        with b:
            plot(C.pca_centroids(proj, int(focus_bucket)), "pca_cent")
        shows(N.pca(proj))
    st.divider()

    step("3.", "Root cause analysis and drift type", "What changed in the images, and what kind of drift is it?")
    if model != "lr":
        st.info("Class activation maps and the concept probe explain the logistic head (a CAM needs a linear head), "
                "so they are shown for logistic regression. The drift type below is for the selected model.")
    else:
        cam_section(cam)
        if concepts is None:
            missing("Concept probe results", "examples/quickstart/clip_concept_probe_demo.py")
        else:
            concept_section(concepts, focus_bucket, cfg["alpha"])
    diagnosis_step(clear, summary, cfg["audit_labels"])
    st.divider()
    reliability_step(clear, focus, focus_label)
    st.divider()
    remediation_step(rem, ("clear10",))
    glossary()


def gas_tab() -> None:
    available = [m for m in ("lr", "hgb") if L.study_path("gas", m).exists()]
    if not available:
        missing("Gas sensor results", "examples/quickstart/gas_drift_study.py")
        return
    model = model_selector("gas_model", available)
    gas = load(str(L.study_path("gas", model)))
    rem = load(str(L.study_path("remediation", model)))
    _TAB["prefix"] = f"gas_{model}"
    summary = L.window_summary(gas)
    cfg = gas["config"]
    overview(gas, summary, "measurements",
             f"**Gas sensor array drift** (UCI): {cfg['n_sensors']} metal-oxide sensors measure six gases over 36 months, "
             f"in ten batches. The sensors age, so their response to the same gas changes while the labels stay exact: a "
             f"known change in P(X | Y). Reference: **batches {N.join([str(b) for b in cfg['reference_batches']])}**. "
             "Production: batches 3 to 10. The held-out part of the reference is a no-drift control.")
    focus = focus_selector(summary, f"gas_focus_{model}")
    focus_label = summary.set_index("key").loc[focus, "window"]
    setup_step(gas, summary, "measurements", {
        "reference": f"batches {N.join([str(b) for b in cfg['reference_batches']])}",
        "model": DENSE_MODEL[model],
        "inputs": f"{cfg['n_sensors']} sensors × {cfg['features_per_sensor']} features",
        "classes": N.join(list(cfg["gases"].values())),
        "significance level α": cfg["alpha"],
        "delayed labels in the audit": cfg["audit_rule"],
    })
    sub("Gas mix per window", "The batches differ in which gases were measured, so the class prior moves as well.")
    plot(C.composition_bars(L.gas_composition(gas)), "mix")
    st.divider()
    proxy_step(gas, summary, focus, focus_label)
    detection_step(gas, cfg["mmd"]["n_permutations"])
    sub("2c. Drift localization: which sensors moved",
        "Each of the 128 features is tested with KS and Holm correction; the map counts the shifted features of "
        "each sensor.")
    plot(C.sensor_heatmap(L.gas_sensor_localization(gas)), "sensors")
    st.divider()
    step("3.", "Drift type", "What kind of drift is it? (No SHAP or CAM here: the inputs are sensor features.)")
    diagnosis_step(gas, summary, cfg["audit_rule"], title="3a. Drift type")
    st.divider()
    reliability_step(gas, focus, focus_label)
    st.divider()
    remediation_step(rem, ("gas",))
    glossary()


def benchmark_tab(summary) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        dataset = st.radio("Reference data", list(L.DATASET_LABEL), format_func=L.DATASET_LABEL.get, horizontal=True,
                           key="bench_ds")
    with c2:
        design = st.radio("Design", ["A", "B"], horizontal=True, key="bench_design",
                          format_func=lambda d: {"A": "A: matched mechanisms", "B": "B: held-out mechanisms"}[d],
                          help="Design A partly matches what the tests look for. Design B uses mechanisms the tests "
                               "were not built around; its parameters were fixed before it was first run.")
    with c3:
        model = st.radio("Monitored model", ["lr", "hgb"], format_func=L.MODEL_LABEL.get, horizontal=True,
                         key="bench_model")
    path = L.benchmark_path(dataset, design, model)
    if not path.exists():
        missing("This benchmark configuration", f"examples/quickstart/drift_injection_benchmark.py --dataset {dataset} "
                f"--design {design} --model {model}")
        return
    bench = load(str(path))
    _TAB["prefix"] = f"bench_{dataset}_{design}_{model}"
    s, cfg = bench["summary"], bench["config"]
    runs = L.benchmark_runs(bench)
    types = [t for t in L.TYPE_ORDER if t in set(runs["injected"])]
    st.markdown(f"**Drift injection benchmark** on {L.DATASET_LABEL[dataset]} {cfg['year']}: real rows with a known, "
                "injected drift, so every verdict can be graded against the truth.")
    c = st.columns(4)
    c[0].metric("Windows", s["n_runs"])
    c[1].metric("Replicates (state × seed)", f"{len(cfg['states'])} × {len(cfg['seeds'])}")
    ti = s["type_identification"]["overall"]
    c[2].metric("Drift type named correctly", f"{ti['rate']:.1%}", help=f"95% CI {ti['ci'][0]:.1%} to {ti['ci'][1]:.1%}")
    c[3].metric("Triage recall", f"{s['triage']['recall']['rate']:.0%}" if s["triage"]["recall"]["rate"] is not None else "n/a",
                help="share of windows where retraining truly helps that the triage escalates")

    step("1.", "Benchmark design", "What is injected, and how is each window graded?")
    st.graphviz_chart(pipeline_dot([
        f"{cfg['year']} data of one state\\n(train / calibration / test)", f"train the monitored model\\n({L.MODEL_LABEL[model]})",
        f"draw a {cfg['production_size']:,}-row\\nwindow from the test split", "inject one drift" + ("\\n(or none)" if design == "A" else ""),
        "run the full pipeline", f"grade on {cfg['holdout_size']:,}\\nseparate labelled rows"]), width="content")
    st.caption(f"Repeated for states {N.join(cfg['states'])} and seeds {cfg['seeds'][0]} to {cfg['seeds'][-1]}.")
    recipes = L.injection_recipes(bench)
    st.dataframe(recipes, hide_index=True, width="stretch", height=38 + 35 * len(recipes))
    shows(N.bench_design(bench))

    step("2.", "Ground truth", "How much does each injection really hurt the model?")
    plot(C.bench_true_drop(runs, types), "b_drop")
    shows(N.bench_ground_truth(runs))

    step("3.", "Drift detection", "Which detector notices which drift" + (", and how often does it cry wolf?" if design == "A" else "?"))
    fa = L.benchmark_false_alarms(bench)
    if design == "A" and not fa.empty:
        a, b = st.columns([3, 2])
        with a:
            plot(C.bench_alarm_heatmap(L.benchmark_alarm_rates(runs), types), "b_alarm")
        with b:
            plot(C.false_alarms(fa, cfg["alpha"]), "b_fa")
            shows(N.bench_detection(fa, cfg["alpha"]))
    else:
        plot(C.bench_alarm_heatmap(L.benchmark_alarm_rates(runs), types), "b_alarm")
    st.caption("Concept drift and label noise leave the inputs unchanged, so input detectors are not expected to fire on them.")

    step("4.", "Proxy performance estimation", "Does the label-free estimate track the true accuracy?")
    plot(C.bench_estimate_scatter(runs, types), "b_est")
    bt = L.benchmark_types(bench)
    shows(N.bench_estimator(bt))
    est = L.benchmark_estimators(bench)
    if not est.empty:
        plot(C.estimator_mae(est), "b_estmae")
        shows(N.bench_estimators(est))

    if design == "A":
        step("5.", "Drift localization", "When features are blamed, are they the injected ones?")
        cols = st.columns(len(s["localization"]))
        for col, (t, v) in zip(cols, s["localization"].items()):
            e = v["exact_match"]
            col.metric(f"{L.TYPE_LABEL[t]}: exact match", f"{e['k']} of {e['n']}",
                       help=f"95% CI {e['ci'][0]:.1%} to {e['ci'][1]:.1%}")
        shows(N.bench_localization(bench))
        st.caption(f"Covariate: injected into {N.join([L.feature_label(f) for f in cfg['covariate_features']])}. "
                   f"Concept: the audit should point at {L.feature_label(cfg['concept_region_feature'])}.")
    else:
        step("5.", "Drift localization", "Not graded in design B.")
        st.caption("Selection bias moves every feature correlated with the selection score, and the oblique region, the "
                   "income threshold and class-conditional noise are not defined by one feature, so there is no single "
                   "set of 'true' features to compare with.")

    step("6.", "Drift type", "Is the right kind of drift named, and at which stage?")
    a, b = st.columns([3, 2])
    with a:
        sub("Injected vs diagnosed (counts; shade = share of the row)", "")
        plot(C.confusion_heatmap(L.benchmark_confusion(bench)), "b_conf")
    with b:
        plot(C.stage_share(L.benchmark_stage_share(bench), types), "b_stage")
        plot(C.audit_budget(L.benchmark_audit_budget(bench)), "b_budget")
    if design == "B" and summary is not None:
        mech = L.benchmark_mechanisms(summary, dataset, model)
        if not mech.empty:
            plot(C.mechanism_lines(mech), "b_mech")
    else:
        sub("Correct diagnosis by intensity (95% CI)", "")
        plot(C.bench_per_intensity(L.benchmark_per_intensity(bench), types), "b_pi")
    shows(N.bench_type_id(bench) + " " + N.bench_audit(L.benchmark_audit_budget(bench)))

    step("7.", "Baselines and ablations", "Is the drift type named better than by simpler or learned rules?")
    st.caption("All methods see the same windows, the same test results and the same 600 audit labels. Input "
               "monitoring reads any input alarm as covariate shift. BBSE first reads a shift of the predicted labels as "
               "prior shift. Error-rate monitoring adds an audit accuracy drop read as concept drift. The learned "
               "meta-classifier is gradient boosting on the window's test statistics"
               + (", validated leave one state out." if design == "A" else
                  ", trained on design A with the predicted state held out, so it never sees a held-out mechanism.")
               + " Ablations remove one part of our procedure. The paired test compares the share of correct windows "
                 "per replicate (sign flips, Holm).")
    meth = L.benchmark_methods(summary, f"{dataset}_{design}_{model}") if summary is not None else pd.DataFrame()
    if meth.empty:
        missing("The baseline summary", "examples/quickstart/benchmark_summary.py")
    else:
        plot(C.method_dots(meth), "b_methods")
        shows(N.bench_methods(meth))
        tests = meth[meth["key"] != "ours"][["method", "ours minus method", "p-value", "significant (Holm)"]]
        st.dataframe(tests.sort_values("p-value"), hide_index=True, width="stretch",
                     column_config={"ours minus method": st.column_config.NumberColumn(format="%+.3f"),
                                    "p-value": st.column_config.NumberColumn(format="%.4f")})

    step("8.", "Per-prediction reliability", "Does fusing signals beat confidence alone?")
    plot(C.bench_reliability_gain(bt), "b_rel")
    shows(N.bench_reliability(bt))

    step("9.", "Remediation triage", "Does the label-free triage escalate when retraining truly helps?")
    a, b = st.columns([3, 2])
    with a:
        plot(C.bench_triage_strip(runs, types), "b_triage")
    with b:
        sub("Triage vs truth (windows)", "")
        plot(C.triage_confusion(s["triage"]["confusion"]), "b_tconf")
    shows(N.bench_triage(bench))
    glossary()


def main() -> None:
    st.title("Data Drift Autopsy")
    st.caption("Detect drift without labels, find its cause and type, judge each prediction and decide whether to retrain.")
    concepts = load(str(L.DEFAULT_PATHS["concepts"]))
    cam = load(str(L.DEFAULT_PATHS["cam"]))
    summary = load(str(L.DEFAULT_PATHS["benchmark_summary"]))
    tabs = st.tabs(["ACS (census)", "CLEAR-10 (images)", "Gas sensor array", "Injection benchmark"])
    with tabs[0]:
        acs_tab()
    with tabs[1]:
        clear_tab(concepts, cam)
    with tabs[2]:
        gas_tab()
    with tabs[3]:
        benchmark_tab(summary)


main()
