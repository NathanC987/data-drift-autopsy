"""Turn the study outputs (outputs/*.json) into tidy frames, one per chart.

Pure functions only (no Streamlit), so they are unit tested on their own.
Nothing here decides a result: every value is read from the outputs or is a
direct transformation of them (a share, a Wilson interval, a label).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from drift_autopsy.experiments.acs import ACS_FEATURE_DESCRIPTIONS
from drift_autopsy.stats import wilson_interval

TYPE_LABEL = {"none": "no drift", "covariate": "covariate", "prior": "prior", "concept": "concept",
              "label_noise": "label noise"}
TYPE_ORDER = ["no drift", "covariate", "prior", "concept", "label noise"]

STRATEGY_LABEL = {
    "importance_weighted_retrain": "importance-weighted retrain",
    "retrain_on_recent": "retrain on recent window",
    "full_retrain": "retrain on all delayed windows",
    "feature_drop_retrain": "drop drifted features",
    "label_retrain": "retrain with fresh labels",
    "recalibrate": "recalibrate with fresh labels",
}
LABEL_FREE_STRATEGIES = {"importance_weighted_retrain", "feature_drop_retrain"}

DEFAULT_PATHS = {
    "acs": Path("outputs/folktables_drift_results.json"),
    "clear": Path("outputs/clear10_drift_results.json"),
    "gas": Path("outputs/gas_drift_results_lr.json"),
    "concepts": Path("outputs/clip_concepts/concept_rca_results.json"),
    "remediation": Path("outputs/remediation_results.json"),
    "benchmark": Path("outputs/benchmark/acs_income_A_lr.json"),
    "benchmark_summary": Path("outputs/benchmark/summary.json"),
    "cam": Path("outputs/figures/cam/cam_manifest.json"),
}
MODEL_LABEL = {"lr": "logistic regression", "hgb": "gradient boosting"}
TASK_LABEL = {"income": "ACS Income", "employment": "ACS Employment"}
DATASET_LABEL = {"acs_income": "ACS Income", "acs_employment": "ACS Employment"}


def study_path(study: str, model: str = "lr", task: str = "income") -> Path:
    """Output file of a study for a model family (and ACS task)."""
    if study == "acs":
        if task == "income" and model == "lr":
            return Path("outputs/folktables_drift_results.json")
        return Path(f"outputs/folktables_{task}_drift_results_{model}.json")
    if study == "clear":
        return Path("outputs/clear10_drift_results.json" if model == "lr" else f"outputs/clear10_drift_results_{model}.json")
    if study == "gas":
        return Path(f"outputs/gas_drift_results_{model}.json")
    if study == "remediation":
        return Path("outputs/remediation_results.json" if model == "lr" else f"outputs/remediation_results_{model}.json")
    raise ValueError(study)


def benchmark_path(dataset: str, design: str, model: str) -> Path:
    return Path(f"outputs/benchmark/{dataset}_{design}_{model}.json")


def load(path: Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def feature_label(code: str) -> str:
    """'OCCP' -> 'OCCP (occupation)'; embedding dimensions pass through."""
    desc = ACS_FEATURE_DESCRIPTIONS.get(code)
    return f"{code} ({desc})" if desc else code


def is_acs(study: Dict[str, Any]) -> bool:
    return "pipelines" in next(iter(study["windows"].values()))


# --------------------------------------------------------------------------- #
# windows shared by ACS and CLEAR-10

def window_label(key: str, w: Dict[str, Any]) -> str:
    if "bucket" in w:
        return "bucket 1 (held out)" if w["analysis_type"] == "control" else f"bucket {w['bucket']}"
    if "batch" in w:
        return "batches 1-2 (held out)" if w["analysis_type"] == "control" else f"batch {w['batch']}"
    return key


def window_group(w: Dict[str, Any]) -> str:
    """temporal sequence (control first) or geographic comparison."""
    return "geographic" if w["analysis_type"] == "geographic" else "temporal"


def gap_before(study: Dict[str, Any]) -> Dict[str, bool]:
    """True for a temporal window that follows a missing period (ACS has no standard 2020 release),
    so line charts do not join across the gap."""
    out, prev = {}, None
    for key, w in study["windows"].items():
        year = w.get("year")
        temporal = w["analysis_type"] in ("temporal", "control")
        out[key] = bool(temporal and year is not None and prev is not None and year - prev > 1)
        if temporal and year is not None:
            prev = year
    return out


def _detections(w: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "pipelines" in w:                                   # ACS
        return {n: p["detection"] for n, p in w["pipelines"].items()}
    return w["detection"]                                  # CLEAR-10


def _n_rows(w: Dict[str, Any]) -> int:
    return int(w["n_rows"] if "n_rows" in w else w["n_images"])


def _cbpe_estimate(w: Dict[str, Any]) -> float:
    return w["cbpe_estimated_accuracy"] if "cbpe_estimated_accuracy" in w else w["proxy"]["estimated"]["accuracy"]


def window_summary(study: Dict[str, Any]) -> pd.DataFrame:
    alpha = study["config"]["alpha"]
    gaps = gap_before(study)
    rows = []
    for key, w in study["windows"].items():
        n = _n_rows(w)
        acc = w["actual_accuracy"]
        lo, hi = wilson_interval(int(round(acc * n)), n)
        est = _cbpe_estimate(w)
        diag = w["diagnosis"]["final"]
        p_drop = w["accuracy_drop_test"]["p_value"]
        rows.append({
            "key": key,
            "window": window_label(key, w),
            "group": window_group(w),
            "type": w["analysis_type"],
            "n": n,
            "accuracy": acc, "acc_low": lo, "acc_high": hi,
            "accuracy change": w["accuracy_change"],
            "drop p-value": p_drop,
            "drop significant": bool(p_drop < alpha and w["accuracy_change"] < 0),
            "CBPE estimate": est,
            "CBPE error": est - acc,
            "diagnosis": TYPE_LABEL[diag["drift_type"]],
            "stage": diag["stage"],
            "rationale": diag["rationale"],
            "gap_before": gaps[key],
        })
    return pd.DataFrame(rows)


def focus_window(summary: pd.DataFrame) -> str:
    """Default focus: the largest significant accuracy drop (else the largest drop)."""
    sig = summary[summary["drop significant"]]
    pool = sig if not sig.empty else summary[summary["type"] != "control"]
    return str(pool.sort_values("accuracy change").iloc[0]["key"])


def split_sizes(study: Dict[str, Any]) -> pd.DataFrame:
    cfg = study["config"]
    role = {"train": "fits the monitored model",
            "calibration": "reference window for every detector; fits CBPE and the reliability fusion",
            "test": "honest reference accuracy; no-drift control window"}
    return pd.DataFrame([{"part": k, "rows": int(v), "share": cfg["split_fractions"][k], "role": role[k]}
                         for k, v in cfg["split_sizes"].items()])


def detector_table(study: Dict[str, Any]) -> pd.DataFrame:
    order = list(study["windows"])
    gaps = gap_before(study)
    rows = []
    for key, w in study["windows"].items():
        for name, det in _detections(w).items():
            rows.append({"key": key, "window": window_label(key, w), "group": window_group(w),
                         "gap_before": gaps[key],
                         "order": order.index(key), "detector": name, "drift": bool(det["drift_detected"]),
                         "score": det["score"], "statistic": det.get("statistic"), "p-value": det.get("p_value"),
                         "threshold": det.get("threshold")})
    return pd.DataFrame(rows)


DETECTOR_STATISTIC = {
    "BBSD": ("largest KS statistic on a class probability", "decision: Bonferroni-corrected p < α"),
    "Domain classifier": ("held-out accuracy telling the two windows apart", "decision: binomial p < α"),
    "KS Test": ("largest KS statistic D", "decision: Bonferroni-corrected p < α"),
    "PSI": ("largest feature PSI", "decision: PSI ≥ 0.2 (convention)"),
    "MMD": ("MMD (kernel two-sample distance)", "decision: permutation p < α"),
    "CBPE": ("chi-square of the confidence histogram", "decision: p < α"),
    "PCA reconstruction": ("increase in reconstruction error", "decision: permutation p < α"),
    "Frechet distance": ("Fréchet distance", "decision: permutation p < α"),
}


def confidence_histogram(study: Dict[str, Any], key: str) -> pd.DataFrame:
    meta = _detections(study["windows"][key])["CBPE"]["metadata"]
    ref = np.asarray(meta["reference_bin_counts"], dtype=float)
    win = np.asarray(meta["test_bin_counts"], dtype=float)
    n_bins = len(ref)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lab = f"{edges[i]:.1f}–{edges[i + 1]:.1f}"
        rows.append({"bin": lab, "lower": edges[i], "population": "reference", "share": ref[i] / ref.sum()})
        rows.append({"bin": lab, "lower": edges[i], "population": "window", "share": win[i] / win.sum()})
    df = pd.DataFrame(rows)
    keep = df.groupby("bin")["share"].transform("sum") > 0          # drop bins empty on both sides
    return df[keep].reset_index(drop=True)


def cbpe_test(study: Dict[str, Any], key: str) -> Dict[str, Any]:
    det = _detections(study["windows"][key])["CBPE"]
    meta = det["metadata"]
    return {"p_value": det["p_value"], "drift": det["drift_detected"],
            "ref_mean_confidence": meta.get("ref_mean_confidence"),
            "test_mean_confidence": meta.get("test_mean_confidence")}


def reliability_summary(study: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, w in study["windows"].items():
        s = w["reliability"]["summary"]
        rows.append({"key": key, "window": window_label(key, w), "group": window_group(w), **{k: s.get(k) for k in (
            "auroc_risk", "auroc_confidence_only", "auroc_ood", "auroc_instability",
            "error_rate_low", "error_rate_medium", "error_rate_high", "share_low", "share_medium", "share_high")}})
    return pd.DataFrame(rows)


def reliability_records(study: Dict[str, Any], key: str) -> pd.DataFrame:
    df = pd.DataFrame(study["windows"][key]["reliability"]["records"])
    df["outcome"] = np.where(df["correct"] == 1, "correct", "wrong")
    return df


def fusion_coefficients(study: Dict[str, Any]) -> pd.DataFrame:
    coefs = study["reliability_model"]["fusion"]["coefficients"]
    label = {"confidence_risk": "low confidence", "ood": "out of distribution", "instability": "instability"}
    return pd.DataFrame([{"signal": label.get(k, k), "coefficient": v} for k, v in coefs.items()],
                        columns=["signal", "coefficient"])


# --------------------------------------------------------------------------- #
# drift type decision path (both studies)

DECISION_QUESTIONS = [
    ("features", "feature tests reject"),
    ("multivariate", "MMD rejects"),
    ("predictions", "predictions shifted"),
    ("residual", "shift left after prior correction"),
    ("audit_drop", "audit: accuracy fell"),
    ("audit_structure", "audit: errors depend on a feature"),
]


def decision_path(study: Dict[str, Any]) -> pd.DataFrame:
    """One row per window and question: answer 'yes', 'no' or 'not reached'.

    Mirrors drift_autopsy.diagnosis: the label-free verdict uses the first
    four questions; the audit questions decide only when the label-free stage
    finds nothing.
    """
    alpha = study["config"]["alpha"]
    rows = []
    for key, w in study["windows"].items():
        d = w["diagnosis"]
        final = d["final"]
        verdict = d["label_free_verdict"]
        if "input_shift_features" in d:                                     # ACS
            features = bool(d["input_shift_features"])
            audit_drop = bool(d["audit"]["accuracy_drop_significant"])
            structure = bool(d["audit"]["structured_features"])
            residual = bool(d["residual_shift_features"])
        elif "n_input_shift_features" in d:                                 # gas sensor
            features = d["n_input_shift_features"] > 0
            audit_drop = bool(d["audit"]["accuracy_drop_significant"])
            structure = bool(d["audit"]["structured_features"])
            residual = d["n_residual_shift_features"] > 0
        else:                                                               # CLEAR-10
            features = d["n_input_shift_dimensions"] > 0
            audit_drop = bool(d["audit_accuracy_drop_p_value"] < alpha)
            structure = bool(d["audit_structured_dimensions"])
            residual = bool(final["localized_features"]) if verdict == "covariate" else False
        if "comparison" in w:                       # newer records store the residual tests directly
            residual = bool(w["comparison"]["tests"]["residual_rejected"])
        answers = {
            "features": features,
            "multivariate": bool(d["multivariate_shift"]),
            "predictions": bool(d["prediction_shift"]),
            "residual": None if verdict == "none" else residual,
            "audit_drop": audit_drop if verdict == "none" else None,
            "audit_structure": (structure if audit_drop else None) if verdict == "none" else None,
        }
        for q, text in DECISION_QUESTIONS:
            a = answers[q]
            rows.append({"key": key, "window": window_label(key, w), "question": text,
                         "answer": "not reached" if a is None else ("yes" if a else "no"),
                         "verdict": TYPE_LABEL[final["drift_type"]], "stage": final["stage"],
                         "rationale": final["rationale"]})
    return pd.DataFrame(rows)


def prior_rates(study: Dict[str, Any]) -> pd.DataFrame:
    """ACS only: actual, predicted and BBSE-estimated positive rate per window."""
    rows = []
    gaps = gap_before(study)
    for key, w in study["windows"].items():
        for what, v in (("actual (needs labels)", w["positive_rate"]),
                        ("predicted by the model", w["predicted_positive_rate"]),
                        ("BBSE estimate (label free)", w["diagnosis"]["bbse"]["estimated_production_prior"][-1])):
            rows.append({"key": key, "window": window_label(key, w), "group": window_group(w),
                         "gap_before": gaps[key], "series": what, "positive rate": v})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# ACS specifics

def acs_localization(study: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, w in study["windows"].items():
        for fd in w["pipelines"]["KS Test"]["localization"]["feature_drifts"]:
            ds = fd.get("distribution_shift") or {}
            rows.append({"key": key, "window": window_label(key, w), "feature": fd["feature_name"],
                         "label": feature_label(fd["feature_name"]), "effect size": fd["score"],
                         "p-value": fd["p_value"], "drifted": bool(fd["drift_detected"]),
                         "test": ds.get("test", "numeric"),
                         "unseen category share": ds.get("unseen_category_share")})
    return pd.DataFrame(rows)


def acs_shap(study: Dict[str, Any], key: str) -> pd.DataFrame:
    rca = study["windows"][key]["pipelines"]["KS Test"]["rca"] or {}
    rows = [{"feature": f, "label": feature_label(f), **v} for f, v in (rca.get("distribution_changes") or {}).items()]
    return pd.DataFrame(rows).sort_values("ref_importance") if rows else pd.DataFrame()


def acs_shap_all(study: Dict[str, Any]) -> pd.DataFrame:
    frames = []
    for key, w in study["windows"].items():
        df = acs_shap(study, key)
        if not df.empty:
            frames.append(df.assign(key=key, window=window_label(key, w)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def acs_recommendations(study: Dict[str, Any], key: str) -> List[str]:
    rca = study["windows"][key]["pipelines"]["KS Test"]["rca"] or {}
    return list(rca.get("recommendations") or [])


# --------------------------------------------------------------------------- #
# CLEAR-10 specifics

def clear_localization(study: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, w in study["windows"].items():
        rows.append({"key": key, "window": window_label(key, w),
                     "drifted dimensions": w["localization"]["n_drifted_dimensions"],
                     "of": w["localization"]["n_dimensions"]})
    return pd.DataFrame(rows)


def clear_class_slices(study: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, w in study["windows"].items():
        for cls, v in w["predicted_class_slices"].items():
            rows.append({"key": key, "window": window_label(key, w), "class": cls,
                         "drifted dimensions": v["n_drifted_dimensions"], "n_window": v["n_window"]})
    return pd.DataFrame(rows)


def clear_classwise_f1(study: Dict[str, Any], key: str) -> pd.DataFrame:
    w = study["windows"][key]
    names = study["class_names"]
    rows = []
    for ck, est in w["proxy_classwise"]["estimated"].items():
        act = w["proxy_classwise"]["actual"][ck]
        rows.append({"class": names[ck.split("_")[1]], "estimated F1": est["f1"], "actual F1": act["f1"]})
    return pd.DataFrame(rows).sort_values("actual F1")


def clear_projection(study: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    proj = study.get("embedding_projection")
    if not proj:
        return None
    points = []
    for key, v in proj["windows"].items():
        for (x, y, z), cls in zip(v["points"], v["class_name"]):
            points.append({"bucket": int(key), "PC1": x, "PC2": y, "PC3": z, "class": cls})
    centroids = pd.DataFrame([{"bucket": int(k), "PC1": v["centroid"][0], "PC2": v["centroid"][1],
                               "PC3": v["centroid"][2], "n_images": v["n_images"]}
                              for k, v in proj["windows"].items()]).sort_values("bucket")
    return {"points": pd.DataFrame(points), "centroids": centroids,
            "explained": proj["explained_variance_ratio"], "fitted_on": proj["fitted_on"],
            "n_per_window": proj["n_per_window"]}


# --------------------------------------------------------------------------- #
# gas sensor specifics

def gas_sensor_localization(study: Dict[str, Any]) -> pd.DataFrame:
    """Drifted features per sensor and window (Holm over all features)."""
    rows = []
    for key, w in study["windows"].items():
        for sensor, v in w["localization"]["by_sensor"].items():
            rows.append({"key": key, "window": window_label(key, w), "sensor": f"sensor {int(sensor):02d}",
                         "drifted features": v["n_drifted"], "of": v["n_features"],
                         "mean KS": v["mean_statistic"]})
    return pd.DataFrame(rows)


def gas_composition(study: Dict[str, Any]) -> pd.DataFrame:
    """Share of each gas in the reference and in every window (labels: evaluation only)."""
    rows = [{"window": "reference (batches 1-2)", "gas": g, "share": n / sum(study["reference"]["gas_counts"].values())}
            for g, n in study["reference"]["gas_counts"].items()]
    for key, w in study["windows"].items():
        total = sum(w["gas_counts"].values())
        rows += [{"window": window_label(key, w), "gas": g, "share": n / total} for g, n in w["gas_counts"].items()]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# concept probe

def concept_validation(concepts: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for b, e in concepts["buckets"].items():
        v, fv = e["metadata_validation"], e["focus_class"]["metadata_validation"]
        rows.append({"bucket": int(b), "significant concepts": e["n_significant_concepts"],
                     "top gaining concept": e["top_significant_gaining_concept"],
                     "rho (all)": v["spearman_rho"], "p (all)": v["p_value"], "tags": v["n_tags"],
                     "focus class": e["focus_class"]["class"], "rho (focus)": fv["spearman_rho"],
                     "p (focus)": fv["p_value"]})
    return pd.DataFrame(rows).sort_values("bucket")


def concept_shifts(concepts: Dict[str, Any], bucket: str, focus: bool = False) -> pd.DataFrame:
    e = concepts["buckets"][bucket]
    iv = e["focus_class"]["intervals"] if focus else e["intervals"]
    return pd.DataFrame([{"concept": c, **v} for c, v in iv.items()]).sort_values("delta")


def concept_tag_pairs(concepts: Dict[str, Any], bucket: str) -> pd.DataFrame:
    m = concepts["buckets"][bucket]["metadata_validation"]["matched"]
    return pd.DataFrame([{"tag": r["tag"], "concepts": ", ".join(r["concepts"]),
                          "concept shift": r["concept_delta"], "tag shift": r["tag_shift"]} for r in m])


# --------------------------------------------------------------------------- #
# remediation

def remediation_table(rem: Dict[str, Any], groups=None) -> pd.DataFrame:
    rows = []
    for s in rem["settings"]:
        if groups is not None and s["group"] not in groups:
            continue
        base = {"setting": _setting_label(s["shift_name"]), "group": s["group"], "gap": s["gap"],
                "triage": "escalate" if s["triage"]["will_retraining_help"] else "hold",
                "truly helps": s["probe_truth_helps"], "triage correct": s["triage_correct"]}
        done = {r["strategy"]: r for r in s["results"]}
        for strat in rem["config"]["strategies"]:
            r = done.get(strat)
            row = {**base, "strategy": STRATEGY_LABEL[strat], "strategy_key": strat,
                   "needs labels": strat not in LABEL_FREE_STRATEGIES}
            if r is None:
                row.update({"recovered": np.nan, "ci low": np.nan, "ci high": np.nan, "significant": False,
                            "applicable": False, "reason": s["skipped"].get(strat, "not run")})
            else:
                row.update({"recovered": r["accuracy_recovered"], "ci low": r["recovery_test"]["ci_low"],
                            "ci high": r["recovery_test"]["ci_high"], "significant": bool(r["recovery_significant"]),
                            "applicable": True, "reason": "", "seconds": r["wall_clock_seconds"],
                            "labels": r["n_production_labels_required"]})
            rows.append(row)
    return pd.DataFrame(rows)


def _setting_label(name: str) -> str:
    return (name.replace("ACS temporal ", "").replace("ACS geographic ", "")
            .replace("CLEAR-10 ", ""))


def triage_probes(rem: Dict[str, Any], groups=None) -> pd.DataFrame:
    """Predicted (label free) vs actual (graded) gain of each probe strategy."""
    rows = []
    for s in rem["settings"]:
        if groups is not None and s["group"] not in groups:
            continue
        done = {r["strategy"]: r for r in s["results"]}
        for p in s["triage"]["probes"]:
            r = done.get(p["probe"])
            if r is None:
                continue
            rows.append({"setting": _setting_label(s["shift_name"]), "probe": STRATEGY_LABEL[p["probe"]],
                         "predicted": p["difference"], "pred low": p["ci_low"], "pred high": p["ci_high"],
                         "predicted significant": bool(p["predicted_gain_significant"]),
                         "actual": r["accuracy_recovered"], "act low": r["recovery_test"]["ci_low"],
                         "act high": r["recovery_test"]["ci_high"], "actual significant": bool(r["recovery_significant"])})
    return pd.DataFrame(rows)


def triage_scorecard(rem: Dict[str, Any], groups=None) -> pd.DataFrame:
    rows = []
    for s in rem["settings"]:
        if groups is not None and s["group"] not in groups:
            continue
        rows.append({"setting": _setting_label(s["shift_name"]), "gap": s["gap"],
                     "triage": "escalate" if s["triage"]["will_retraining_help"] else "hold",
                     "truth": "a probe truly helps" if s["probe_truth_helps"] else "no probe helps",
                     "correct": bool(s["triage_correct"]), "rationale": s["triage"]["rationale"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# benchmark

def benchmark_runs(bench: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for r in bench["runs"]:
        gt = r["ground_truth"]
        row = {
            "state": r["state"], "seed": r["seed"], "injected": TYPE_LABEL[r["drift_type"]],
            "mechanism": MECHANISM_LABEL.get(r.get("mechanism", r["drift_type"]), r.get("mechanism", r["drift_type"])),
            "intensity": r["intensity_label"], "intensity_order": r["intensity_order"],
            "true accuracy": r["true_accuracy"], "true drop": r["true_accuracy_drop"],
            "reference accuracy": gt["reference_accuracy"],
            "estimated accuracy": r["estimator"]["estimated_accuracy"], "estimate error": r["estimator"]["error"],
            "diagnosed": TYPE_LABEL[r["diagnosis"]["drift_type"]], "stage": r["diagnosis"]["stage"],
            "correct": bool(r["diagnosis_correct"]),
            "triage": "escalate" if r["triage"]["will_retraining_help"] else "hold",
            "actual gain": r["triage_truth"]["actual_gain"], "retraining helps": bool(r["triage_truth"]["retraining_helps"]),
            "auroc fused": r["reliability"]["auroc_risk"], "auroc confidence": r["reliability"]["auroc_confidence_only"],
        }
        for name, det in r["detection"].items():
            row[f"alarm {name}"] = bool(det["drift_detected"])
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_alarm_rates(runs: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in runs.columns if c.startswith("alarm ")]
    long = runs.melt(id_vars=["injected", "intensity", "intensity_order"], value_vars=cols,
                     var_name="detector", value_name="alarm")
    long["detector"] = long["detector"].str.replace("alarm ", "", regex=False)
    g = long.groupby(["injected", "intensity", "intensity_order", "detector"], as_index=False)["alarm"].agg(["sum", "count"])
    g["rate"] = g["sum"] / g["count"]
    return g.reset_index(drop=True)


def benchmark_types(bench: Dict[str, Any]) -> pd.DataFrame:
    s = bench["summary"]
    ti = s["type_identification"]
    rows = []
    for t in ti["labels"]:
        if not ti["per_type"][t]["n"]:
            continue                                  # type absent from this design
        st = ti["stage_counts"][t]
        rows.append({
            "injected": TYPE_LABEL[t],
            "correct": ti["per_type"][t]["rate"],
            "ci low": ti["per_type"][t]["ci"][0], "ci high": ti["per_type"][t]["ci"][1],
            "label-free share": st.get("label-free", 0) / max(sum(st.values()), 1),
            "CBPE error": s["estimator"][t]["error"]["mean"],
            "fused AUROC": s["reliability"][t]["auroc_risk"]["mean"],
            "fused minus confidence": s["reliability"][t]["auroc_risk_minus_confidence"]["mean"],
            "fused minus confidence low": s["reliability"][t]["auroc_risk_minus_confidence"]["ci"][0],
            "fused minus confidence high": s["reliability"][t]["auroc_risk_minus_confidence"]["ci"][1],
            "triage escalates": s["triage"]["per_type"][t]["escalated"]["rate"],
            "retraining truly helps": s["triage"]["per_type"][t]["truly_helps"]["rate"],
        })
    return pd.DataFrame(rows)


def benchmark_per_intensity(bench: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k, v in bench["summary"]["type_identification"]["per_intensity"].items():
        t, lab = k.split("/")
        rows.append({"injected": TYPE_LABEL[t], "intensity": lab, "correct": v["rate"],
                     "ci low": v["ci"][0], "ci high": v["ci"][1], "k": v["k"], "n": v["n"]})
    return pd.DataFrame(rows)


def benchmark_stage_share(bench: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for t, counts in bench["summary"]["type_identification"]["stage_counts"].items():
        total = sum(counts.values())
        if not total:
            continue
        for stage in ("label-free", "audit"):
            rows.append({"injected": TYPE_LABEL[t], "stage": stage, "share": counts.get(stage, 0) / total,
                         "count": counts.get(stage, 0)})
    return pd.DataFrame(rows)


def benchmark_false_alarms(bench: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"detector": k, "rate": v["rate"], "ci low": v["ci"][0], "ci high": v["ci"][1],
                          "k": v["k"], "n": v["n"]} for k, v in bench["summary"]["detector_false_alarms"].items()])


def benchmark_confusion(bench: Dict[str, Any]) -> pd.DataFrame:
    """Injected (rows, only types present in the design) against diagnosed (all five types)."""
    ti = bench["summary"]["type_identification"]
    labels = ti["labels"]
    rows = [a for a in labels if sum(ti["confusion_matrix"][a].values())]
    return pd.DataFrame([[ti["confusion_matrix"][a][b] for b in labels] for a in rows],
                        index=[TYPE_LABEL[t] for t in rows], columns=[TYPE_LABEL[t] for t in labels])


def benchmark_audit_budget(bench: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for t, by_n in bench["summary"]["audit_budget"].items():
        for n, v in by_n.items():
            rows.append({"type": TYPE_LABEL[t], "labels": int(n), "correct": v["rate"],
                         "ci low": v["ci"][0], "ci high": v["ci"][1]})
    return pd.DataFrame(rows)


MECHANISM_LABEL = {
    "none": "no drift", "covariate": "covariate (scale and offset)", "prior": "prior increase",
    "concept": "concept (one-feature region)", "label_noise": "uniform label noise",
    "selection": "covariate (selection bias)", "prior_decrease": "prior decrease",
    "concept_oblique": "concept (oblique region)", "concept_threshold": "concept (income threshold)",
    "class_noise": "class-conditional label noise",
}


def injection_recipes(bench: Dict[str, Any]) -> pd.DataFrame:
    cfg = bench["config"]
    cov = ", ".join(feature_label(f) for f in cfg["covariate_features"])
    region = feature_label(cfg["concept_region_feature"])
    describe = {
        "covariate": lambda p: (f"{cov}: x → μ + {p['scale']}·(x − μ) + {p['offset']}·σ "
                                f"for a random {p['fraction']:.0%} of rows; labels untouched"),
        "prior": lambda p: f"rows resampled so the positive rate rises by {p['positive_rate_increase']:.2f}; P(X|Y) kept",
        "concept": lambda p: f"labels inverted for the top {p['region_fraction']:.0%} of rows by {region}; X untouched",
        "label_noise": lambda p: f"a random {p['rate']:.0%} of labels flipped; X untouched",
        "selection": lambda p: (f"each row kept with probability sigmoid({p['beta']}·s(x)), s a random score over all "
                                "inputs; real rows, P(Y|X) unchanged"),
        "prior_decrease": lambda p: (f"rows resampled so the positive rate falls by {p['positive_rate_decrease']:.2f}; "
                                     "P(X|Y) kept"),
        "concept_oblique": lambda p: (f"labels inverted for the top {p['region_fraction']:.0%} of rows by a second "
                                      "random score; X untouched"),
        "concept_threshold": lambda p: f"label = income above ${p['threshold']:,.0f} instead of $50,000; X untouched",
        "class_noise": lambda p: f"each positive label flipped to negative with probability {p['rate']:.0%}; X untouched",
    }
    types = cfg.get("mechanism_type", {t: t for t in cfg["intensity_ladder"]})
    rows = []
    if cfg.get("design", "A") == "A":
        rows.append({"mechanism": "no drift", "type": "no drift", "intensity": "control",
                     "what changes": "nothing (held-out rows as they are)"})
    for m, ladder in cfg["intensity_ladder"].items():
        for p in ladder:
            rows.append({"mechanism": MECHANISM_LABEL[m], "type": TYPE_LABEL[types.get(m, m)], "intensity": p["label"],
                         "what changes": describe[m](p)})
    return pd.DataFrame(rows)


METHOD_KIND = {"ours": "ours", "input_monitoring": "baseline", "bbse_first": "baseline",
               "error_rate_monitoring": "baseline", "meta_classifier": "learned", "no_mmd": "ablation",
               "no_confidence_null": "ablation", "class_confidence_null": "ablation"}
METHOD_NAME = {
    "ours": "Ours (hierarchical tests)", "input_monitoring": "Input monitoring", "bbse_first": "BBSE first",
    "error_rate_monitoring": "Error-rate monitoring", "meta_classifier": "Learned meta-classifier",
    "no_mmd": "Ours without the MMD block", "no_confidence_null": "Ours, audit null without confidence",
    "class_confidence_null": "Ours, audit null with class × confidence",
}


def benchmark_methods(summary: Dict[str, Any], key: str) -> pd.DataFrame:
    """Accuracy (Wilson CI), macro F1 and the paired test against ours for each method of one configuration."""
    m = summary["methods"].get(key)
    if not m:
        return pd.DataFrame()
    rows = []
    for name, v in m["scores"].items():
        test = m["paired_tests"].get(name, {})
        rows.append({"method": METHOD_NAME.get(name, name), "key": name, "kind": METHOD_KIND.get(name, "other"),
                     "accuracy": v["accuracy"]["rate"], "ci low": v["accuracy"]["ci"][0],
                     "ci high": v["accuracy"]["ci"][1], "macro F1": v["macro_f1"],
                     "ours minus method": test.get("mean_difference"), "p-value": test.get("p_value"),
                     "significant (Holm)": test.get("holm_significant")})
    return pd.DataFrame(rows)


def benchmark_mechanisms(summary: Dict[str, Any], dataset: str, model: str) -> pd.DataFrame:
    mech = summary["mechanisms"].get(f"{dataset}_{model}", {})
    rows = []
    for m, v in mech.items():
        for lab, rate in v["per_intensity"].items():
            rows.append({"mechanism": MECHANISM_LABEL[m], "type": TYPE_LABEL[v["drift_type"]], "intensity": lab,
                         "correct": rate, "true drop": v["mean_true_drop_per_intensity"][lab]})
    return pd.DataFrame(rows)


ESTIMATOR_NAME = {"cbpe": "CBPE (ours)", "ac": "average confidence", "doc": "DoC", "atc": "ATC"}


def benchmark_estimators(bench: Dict[str, Any]) -> pd.DataFrame:
    e = bench["summary"].get("estimators") or {}
    rows = []
    for name, by_type in (e.get("mae") or {}).items():
        for t, v in by_type.items():
            if v["mean"] is None:
                continue
            rows.append({"estimator": ESTIMATOR_NAME.get(name, name), "key": name,
                         "type": "all windows" if t == "overall" else TYPE_LABEL[t],
                         "MAE": v["mean"], "ci low": v["ci"][0], "ci high": v["ci"][1]})
    return pd.DataFrame(rows)
