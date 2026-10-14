"""Turn the study outputs (outputs/*.json) into tidy tables for the dashboard.

Pure functions only (no Streamlit), so they are unit tested on their own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

TYPE_LABEL = {"none": "no drift", "covariate": "covariate", "prior": "prior", "concept": "concept",
              "label_noise": "label noise"}

DEFAULT_PATHS = {
    "acs": Path("outputs/folktables_drift_results.json"),
    "clear": Path("outputs/clear10_drift_results.json"),
    "concepts": Path("outputs/clip_concepts/concept_rca_results.json"),
    "remediation": Path("outputs/remediation_results.json"),
    "benchmark": Path("outputs/drift_injection_benchmark.json"),
}


def load(path: Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- #
# shared window tables (ACS and CLEAR-10 use the same shape)

def _detections(w: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "pipelines" in w:                                   # ACS
        return {n: p["detection"] for n, p in w["pipelines"].items()}
    return w["detection"]                                  # CLEAR-10


def window_summary(study: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, w in study["windows"].items():
        est = w["cbpe_estimated_accuracy"] if "cbpe_estimated_accuracy" in w else w["proxy"]["estimated"]["accuracy"]
        diag = w["diagnosis"]["final"]
        rows.append({
            "window": key,
            "type": w["analysis_type"],
            "accuracy": w["actual_accuracy"],
            "accuracy change": w["accuracy_change"],
            "drop p-value": w["accuracy_drop_test"]["p_value"],
            "CBPE estimate": est,
            "CBPE error": est - w["actual_accuracy"],
            "diagnosis": TYPE_LABEL[diag["drift_type"]],
            "stage": diag["stage"],
            "rationale": diag["rationale"],
        })
    return pd.DataFrame(rows)


def detector_table(study: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, w in study["windows"].items():
        for name, det in _detections(w).items():
            rows.append({"window": key, "detector": name, "drift": bool(det["drift_detected"]),
                         "score": det["score"], "p-value": det.get("p_value")})
    return pd.DataFrame(rows)


def reliability_summary(study: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, w in study["windows"].items():
        s = w["reliability"]["summary"]
        rows.append({"window": key, **{k: s.get(k) for k in (
            "auroc_risk", "auroc_confidence_only", "auroc_ood", "auroc_instability",
            "error_rate_low", "error_rate_medium", "error_rate_high", "share_low", "share_medium", "share_high")}})
    return pd.DataFrame(rows)


def reliability_records(study: Dict[str, Any], window: str) -> pd.DataFrame:
    return pd.DataFrame(study["windows"][window]["reliability"]["records"])


# --------------------------------------------------------------------------- #
# ACS specifics

def acs_localization(study: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, w in study["windows"].items():
        for fd in w["pipelines"]["KS Test"]["localization"]["feature_drifts"]:
            ds = fd.get("distribution_shift") or {}
            rows.append({"window": key, "feature": fd["feature_name"], "effect size": fd["score"],
                         "p-value": fd["p_value"], "drifted": fd["drift_detected"], "test": ds.get("test"),
                         "unseen category share": ds.get("unseen_category_share")})
    return pd.DataFrame(rows)


def acs_shap(study: Dict[str, Any], window: str) -> pd.DataFrame:
    rca = study["windows"][window]["pipelines"]["KS Test"]["rca"] or {}
    rows = [{"feature": f, **v} for f, v in (rca.get("distribution_changes") or {}).items()]
    return pd.DataFrame(rows).sort_values("change") if rows else pd.DataFrame()


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
    return pd.DataFrame(rows)


def concept_shifts(concepts: Dict[str, Any], bucket: str, focus: bool = False) -> pd.DataFrame:
    e = concepts["buckets"][bucket]
    iv = e["focus_class"]["intervals"] if focus else e["intervals"]
    return pd.DataFrame([{"concept": c, **v} for c, v in iv.items()]).sort_values("delta")


# --------------------------------------------------------------------------- #
# remediation and benchmark

def remediation_table(rem: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for s in rem["settings"]:
        base = {"setting": s["shift_name"], "group": s["group"], "gap": s["gap"],
                "triage": "escalate" if s["triage"]["will_retraining_help"] else "hold",
                "truly helps": s["probe_truth_helps"], "triage correct": s["triage_correct"]}
        for r in s["results"]:
            rows.append({**base, "strategy": r["strategy"], "recovered": r["accuracy_recovered"],
                         "ci low": r["recovery_test"]["ci_low"], "ci high": r["recovery_test"]["ci_high"],
                         "significant": r["recovery_significant"], "seconds": r["wall_clock_seconds"],
                         "labels": r["n_production_labels_required"]})
    return pd.DataFrame(rows)


def benchmark_types(bench: Dict[str, Any]) -> pd.DataFrame:
    s = bench["summary"]
    ti = s["type_identification"]
    rows = []
    for t in ti["labels"]:
        st = ti["stage_counts"][t]
        rows.append({
            "injected": TYPE_LABEL[t],
            "correct": ti["per_type"][t]["rate"],
            "ci low": ti["per_type"][t]["ci"][0], "ci high": ti["per_type"][t]["ci"][1],
            "label-free share": st.get("label-free", 0) / max(sum(st.values()), 1),
            "CBPE error": s["estimator"][t]["error"]["mean"],
            "fused AUROC": s["reliability"][t]["auroc_risk"]["mean"],
            "fused minus confidence": s["reliability"][t]["auroc_risk_minus_confidence"]["mean"],
            "triage escalates": s["triage"]["per_type"][t]["escalated"]["rate"],
            "retraining truly helps": s["triage"]["per_type"][t]["truly_helps"]["rate"],
        })
    return pd.DataFrame(rows)


def benchmark_confusion(bench: Dict[str, Any]) -> pd.DataFrame:
    ti = bench["summary"]["type_identification"]
    labels = ti["labels"]
    return pd.DataFrame([[ti["confusion_matrix"][a][b] for b in labels] for a in labels],
                        index=[TYPE_LABEL[t] for t in labels], columns=[TYPE_LABEL[t] for t in labels])


def benchmark_audit_budget(bench: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for t, by_n in bench["summary"]["audit_budget"].items():
        for n, v in by_n.items():
            rows.append({"type": TYPE_LABEL[t], "labels": int(n), "correct": v["rate"],
                         "ci low": v["ci"][0], "ci high": v["ci"][1]})
    return pd.DataFrame(rows)
