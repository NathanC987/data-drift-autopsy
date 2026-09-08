"""Grade one benchmark run and aggregate a sweep.

Every function here compares a pipeline output against an
:class:`~drift_autopsy.benchmark.spec.InjectedDriftSpec` answer key. Nothing is
label-free -- this is the marking scheme, not part of the monitored system.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from drift_autopsy.benchmark.spec import DRIFT_TYPES


def localisation_prf(
    localised: Sequence[str],
    injected: Sequence[str],
    audit_feature: Optional[str] = None,
) -> Dict[str, Any]:
    """Precision/recall/F1 of the localised feature set against the injected one.

    ``localised`` is the label-free (KS) localisation; ``audit_feature`` is the
    single feature the delayed-label audit flagged as carrying the loss. When no
    feature was injected (prior, label noise) there is no target, so the score is
    the true-negative check: did localisation correctly stay empty?
    """
    localised = set(localised)
    injected = set(injected)
    audit_hit = audit_feature in injected if (audit_feature and injected) else None

    if not injected:
        return {
            "applicable": False,
            "precision": None,
            "recall": None,
            "f1": None,
            "false_positive_features": sorted(localised),
            "clean": len(localised) == 0,
            "audit_feature": audit_feature,
            "audit_match": None,
        }

    tp = len(localised & injected)
    precision = tp / len(localised) if localised else 0.0
    recall = tp / len(injected)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "applicable": True,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched": sorted(localised & injected),
        "missed": sorted(injected - localised),
        "false_positive_features": sorted(localised - injected),
        "audit_feature": audit_feature,
        "audit_match": audit_hit,
    }


def estimator_error(estimated_accuracy: float, spec) -> Dict[str, Any]:
    """How the label-free accuracy estimate compares to the true production accuracy."""
    implied_drop = spec.reference_accuracy - estimated_accuracy
    true_drop = spec.true_accuracy_drop
    signed_error = float(estimated_accuracy - spec.production_accuracy)
    return {
        "estimated_accuracy": round(float(estimated_accuracy), 4),
        "true_accuracy": round(float(spec.production_accuracy), 4),
        "signed_error": round(signed_error, 4),
        "optimistic": bool(signed_error > 0),
        "implied_drop": round(float(implied_drop), 4),
        "true_drop": round(float(true_drop), 4),
        "drop_captured_fraction": round(
            float(implied_drop / true_drop) if abs(true_drop) > 1e-6 else 0.0, 4
        ),
    }


def type_identification_summary(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Confusion matrix + accuracy of predicted vs injected drift type."""
    labels = list(DRIFT_TYPES)
    matrix = {a: {b: 0 for b in labels} for a in labels}
    correct = 0
    for r in runs:
        true_t = r["ground_truth"]["drift_type"]
        pred_t = r["verdict"]["predicted_type"]
        if pred_t not in matrix[true_t]:
            matrix[true_t][pred_t] = 0
        matrix[true_t][pred_t] += 1
        correct += int(true_t == pred_t)
    return {
        "accuracy": round(correct / len(runs), 4) if runs else 0.0,
        "n_runs": len(runs),
        "confusion_matrix": matrix,
        "labels": labels,
    }


def _max_detector_score(detection: Dict[str, Any]) -> float:
    """A single continuous 'how loud is detection' number: the largest KS/PSI/MMD
    statistic plus a bounded CBPE contribution."""
    ks = detection.get("KS Test", {})
    psi = detection.get("PSI", {})
    mmd = detection.get("MMD", {})
    cbpe = detection.get("CBPE", {})
    parts = []
    for b in (ks, psi, mmd):
        if isinstance(b, dict) and "error" not in b:
            parts.append(float(b.get("score", 0.0)))
    cbpe_term = 0.0
    if isinstance(cbpe, dict) and "error" not in cbpe:
        cbpe_term = float(np.tanh(cbpe.get("score", 0.0) / 200.0))
    return float(max(parts, default=0.0) + cbpe_term)


def intensity_response(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per drift type: does each stage's output rise monotonically with the
    injected intensity? Spearman rho over the intensity ladder for the label-free
    detector loudness, the label-free estimator's implied drop, the delayed-label
    audit's measured drop, and the batch high-risk rate."""
    from scipy.stats import spearmanr

    def _rho(pairs: List[tuple]) -> Optional[float]:
        if len(pairs) < 3:
            return None
        xs, ys = zip(*sorted(pairs))
        if len(set(round(v, 6) for v in ys)) < 2:
            return 0.0
        rho, _ = spearmanr(xs, ys)
        return round(float(rho), 4) if np.isfinite(rho) else None

    by_type: Dict[str, Dict[str, List[tuple]]] = {}
    for r in runs:
        t = r["ground_truth"]["drift_type"]
        if t == "none":
            continue
        o = r["intensity_order"]
        acc = by_type.setdefault(t, {"detector": [], "estimator": [], "audit": [], "risk": []})
        acc["detector"].append((o, _max_detector_score(r["pipeline"]["detection"])))
        acc["estimator"].append((o, abs(r["grading"]["estimator"]["implied_drop"])))
        acc["audit"].append((o, r["labelled_audit"]["measured_drop"]))
        acc["risk"].append((o, r["pipeline"]["reliability"].get("high_risk_pct", 0.0) or 0.0))

    return {
        t: {
            "label_free_detector_rho": _rho(v["detector"]),
            "label_free_estimator_rho": _rho(v["estimator"]),
            "labelled_audit_drop_rho": _rho(v["audit"]),
            "reliability_high_risk_rho": _rho(v["risk"]),
        }
        for t, v in by_type.items()
    }


def reliability_auroc_by_type(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Strongest-intensity reliability AUROC (risk score vs prediction error) per type."""
    out: Dict[str, Any] = {}
    for t in DRIFT_TYPES:
        if t == "none":
            continue
        cands = [r for r in runs if r["ground_truth"]["drift_type"] == t]
        if not cands:
            continue
        strongest = max(cands, key=lambda r: r["intensity_order"])
        out[t] = strongest["pipeline"]["reliability"].get("auroc_risk_vs_error")
    return out


def aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    loc_f1 = {}
    for t in ("covariate", "concept"):
        vals = [
            r["grading"]["localisation"]["f1"]
            for r in runs
            if r["ground_truth"]["drift_type"] == t
            and r["grading"]["localisation"].get("f1") is not None
        ]
        if vals:
            loc_f1[t] = round(float(np.mean(vals)), 4)

    optimistic = [
        r["grading"]["estimator"]["optimistic"]
        for r in runs
        if r["ground_truth"]["drift_type"] != "none"
    ]

    clean_when_unlocalised = [
        r["grading"]["localisation"]["clean"]
        for r in runs
        if not r["grading"]["localisation"]["applicable"]
        and r["ground_truth"]["drift_type"] != "none"
    ]

    audit_localised = [
        r["grading"]["localisation"]["audit_match"]
        for r in runs
        if r["grading"]["localisation"].get("audit_match") is not None
    ]

    return {
        "type_identification": type_identification_summary(runs),
        "intensity_response": intensity_response(runs),
        "mean_localisation_f1": loc_f1,
        "audit_localisation_hit_rate": (
            round(float(np.mean(audit_localised)), 4) if audit_localised else None
        ),
        "estimator_optimism_rate": round(float(np.mean(optimistic)), 4) if optimistic else None,
        "localisation_clean_rate_when_no_feature_target": (
            round(float(np.mean(clean_when_unlocalised)), 4) if clean_when_unlocalised else None
        ),
        "reliability_auroc_by_type": reliability_auroc_by_type(runs),
    }
