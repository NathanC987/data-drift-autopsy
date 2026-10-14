"""Aggregate benchmark runs into rates with confidence intervals.

Proportions carry 95% Wilson intervals; means carry 95% percentile intervals
from a bootstrap over replicates (state, seed), which respects the fact that
windows within a replicate share a model and a reference.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.stats import spearmanr

from drift_autopsy.benchmark.spec import DRIFT_TYPES
from drift_autopsy.stats import wilson_interval

EXPECTED_TYPE = {t: t for t in DRIFT_TYPES}


def _rho(x: Sequence[float], y: Sequence[float]):
    """Spearman correlation, or None when undefined (fewer than 3 points or a constant input)."""
    if len(x) < 3 or len(set(np.round(x, 12))) < 2 or len(set(np.round(y, 12))) < 2:
        return None
    r = spearmanr(x, y).correlation
    return float(r) if np.isfinite(r) else None


def _prop(k: int, n: int) -> Dict[str, Any]:
    lo, hi = wilson_interval(k, n)
    return {"k": int(k), "n": int(n), "rate": (k / n) if n else None, "ci": [lo, hi]}


def _mean_ci(values_by_replicate: Dict[Any, List[float]], n_boot: int = 2000, seed: int = 0) -> Dict[str, Any]:
    reps = [v for v in values_by_replicate.values() if v]
    if not reps:
        return {"mean": None, "ci": [None, None], "n": 0}
    flat = np.concatenate([np.asarray(v, dtype=float) for v in reps])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(reps), len(reps))
        boots.append(np.mean(np.concatenate([np.asarray(reps[i], dtype=float) for i in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(flat.mean()), "ci": [float(lo), float(hi)], "n": int(flat.size)}


def _rep(r: Dict[str, Any]):
    return (r["state"], r["seed"])


def type_identification(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = list(DRIFT_TYPES)
    matrix = {a: {b: 0 for b in labels} for a in labels}
    stage = {a: defaultdict(int) for a in labels}
    for r in runs:
        matrix[r["drift_type"]][r["diagnosis"]["drift_type"]] += 1
        stage[r["drift_type"]][r["diagnosis"]["stage"]] += 1
    per_type = {t: _prop(matrix[t][EXPECTED_TYPE[t]], sum(matrix[t].values())) for t in labels}
    per_intensity: Dict[str, Dict[str, Any]] = {}
    for t in labels:
        for lab in sorted({r["intensity_label"] for r in runs if r["drift_type"] == t}):
            sub = [r for r in runs if r["drift_type"] == t and r["intensity_label"] == lab]
            per_intensity[f"{t}/{lab}"] = _prop(sum(r["diagnosis_correct"] for r in sub), len(sub))
    total = sum(r["diagnosis_correct"] for r in runs)
    return {
        "overall": _prop(total, len(runs)),
        "per_type": per_type,
        "per_intensity": per_intensity,
        "confusion_matrix": matrix,
        "stage_counts": {t: dict(v) for t, v in stage.items()},
        "labels": labels,
    }


def audit_budget(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Type-identification rate by audit size, for the types the audit decides."""
    out: Dict[str, Any] = {}
    sizes = sorted({int(k) for r in runs for k in r["diagnosis_by_audit_size"]})
    for t in ("none", "concept", "label_noise"):
        sub = [r for r in runs if r["drift_type"] == t]
        out[t] = {str(n): _prop(sum(r["diagnosis_by_audit_size"][str(n)] == EXPECTED_TYPE[t] for r in sub), len(sub))
                  for n in sizes}
    return out


def estimator(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for t in DRIFT_TYPES:
        sub = [r for r in runs if r["drift_type"] == t]
        by_rep: Dict[Any, List[float]] = defaultdict(list)
        for r in sub:
            by_rep[_rep(r)].append(r["estimator"]["error"])
        out[t] = {
            "error": _mean_ci(by_rep),
            "optimistic": _prop(sum(r["estimator"]["error"] > 0 for r in sub), len(sub)),
        }
    return out


def reliability(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for t in DRIFT_TYPES:
        sub = [r for r in runs if r["drift_type"] == t and r["reliability"].get("auroc_risk") is not None]
        res = {}
        for key in ("auroc_risk", "auroc_confidence_only", "auroc_ood", "auroc_instability"):
            by_rep: Dict[Any, List[float]] = defaultdict(list)
            for r in sub:
                by_rep[_rep(r)].append(r["reliability"][key])
            res[key] = _mean_ci(by_rep)
        diff: Dict[Any, List[float]] = defaultdict(list)
        for r in sub:
            diff[_rep(r)].append(r["reliability"]["auroc_risk"] - r["reliability"]["auroc_confidence_only"])
        res["auroc_risk_minus_confidence"] = _mean_ci(diff)
        out[t] = res
    return out


def triage(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(r["triage"]["will_retraining_help"] and r["triage_truth"]["retraining_helps"] for r in runs)
    tn = sum((not r["triage"]["will_retraining_help"]) and (not r["triage_truth"]["retraining_helps"]) for r in runs)
    fp = sum(r["triage"]["will_retraining_help"] and not r["triage_truth"]["retraining_helps"] for r in runs)
    fn = sum((not r["triage"]["will_retraining_help"]) and r["triage_truth"]["retraining_helps"] for r in runs)
    predicted = [next(p["difference"] for p in r["triage"]["probes"] if p["probe"] == "importance_weighted_retrain")
                 for r in runs]
    actual = [r["triage_truth"]["actual_gain"] for r in runs]
    rho = _rho(predicted, actual)
    per_type = {}
    for t in DRIFT_TYPES:
        sub = [r for r in runs if r["drift_type"] == t]
        per_type[t] = {
            "escalated": _prop(sum(r["triage"]["will_retraining_help"] for r in sub), len(sub)),
            "truly_helps": _prop(sum(r["triage_truth"]["retraining_helps"] for r in sub), len(sub)),
            "agreement": _prop(sum(r["triage"]["will_retraining_help"] == r["triage_truth"]["retraining_helps"]
                                   for r in sub), len(sub)),
        }
    return {
        "agreement": _prop(tp + tn, len(runs)),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision": _prop(tp, tp + fp),
        "recall": _prop(tp, tp + fn),
        "spearman_predicted_vs_actual_gain": rho,
        "per_type": per_type,
    }


def localization(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for t in ("covariate", "concept"):
        sub = [r for r in runs if r["drift_type"] == t and r["diagnosis_correct"] and r["localization"]]
        exact = sum(set(r["localization"]["localized"]) == set(r["localization"]["injected"]) for r in sub)
        prec: Dict[Any, List[float]] = defaultdict(list)
        rec: Dict[Any, List[float]] = defaultdict(list)
        for r in sub:
            prec[_rep(r)].append(r["localization"]["precision"])
            rec[_rep(r)].append(r["localization"]["recall"])
        out[t] = {"exact_match": _prop(exact, len(sub)), "precision": _mean_ci(prec), "recall": _mean_ci(rec)}
    return out


def detector_false_alarms(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-detector alarm rate on no-drift control windows (nominal: alpha for tests)."""
    ctrl = [r for r in runs if r["drift_type"] == "none"]
    names = sorted({n for r in ctrl for n in r["detection"]})
    return {n: _prop(sum(r["detection"][n]["drift_detected"] for r in ctrl), len(ctrl)) for n in names}


def intensity_response(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Spearman correlation of intensity order with the true drop, the label-free
    evidence (-log10 of the smallest shift-test p-value) and the audited drop."""
    out: Dict[str, Any] = {}
    for t in DRIFT_TYPES:
        if t == "none":
            continue
        sub = [r for r in runs if r["drift_type"] == t]
        order = [r["intensity_order"] for r in sub]
        evidence = [-np.log10(max(r["label_free"]["min_input_p_value"], 1e-300)) for r in sub]
        drop = [r["true_accuracy_drop"] for r in sub]
        audit_drop = [r["audit"]["audit_accuracy"] for r in sub]
        out[t] = {
            "true_drop_rho": _rho(order, drop),
            "label_free_evidence_rho": _rho(order, evidence),
            "audit_accuracy_rho": _rho(order, audit_drop),
            "mean_true_drop_by_intensity": {
                str(o): float(np.mean([d for oo, d in zip(order, drop) if oo == o])) for o in sorted(set(order))
            },
        }
    return out


def aggregate(runs: List[Dict[str, Any]], cfg=None) -> Dict[str, Any]:
    return {
        "n_runs": len(runs),
        "n_replicates": len({_rep(r) for r in runs}),
        "type_identification": type_identification(runs),
        "audit_budget": audit_budget(runs),
        "detector_false_alarms": detector_false_alarms(runs),
        "estimator": estimator(runs),
        "reliability": reliability(runs),
        "triage": triage(runs),
        "localization": localization(runs),
        "intensity_response": intensity_response(runs),
    }
