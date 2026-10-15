"""Aggregate benchmark runs into rates with confidence intervals.

Proportions carry 95% Wilson intervals; means carry 95% percentile intervals
from a bootstrap over replicates (state, seed), which respects the fact that
windows within a replicate share a model and a reference.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

from drift_autopsy.benchmark.spec import DRIFT_TYPES
from drift_autopsy.diagnosis_baselines import META_FEATURES, meta_features
from drift_autopsy.stats import holm_reject, macro_f1, sign_flip_test, wilson_interval

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
        if not sub:
            continue
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
    alpha = getattr(cfg, "alpha", 0.05)
    return {
        "methods": methods(runs, alpha) if runs and "verdicts" in runs[0] else {},
        "estimators": estimators(runs),
        "detection_power": detection_power(runs),
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


# --------------------------------------------------------------------------- #
# baselines, ablations and the learned meta-classifier

def present_types(runs: List[Dict[str, Any]]) -> List[str]:
    return [t for t in DRIFT_TYPES if any(r["drift_type"] == t for r in runs)]


def method_scores(runs: List[Dict[str, Any]], predictions: Dict[str, List[str]]) -> Dict[str, Any]:
    """Accuracy, macro F1, per-type rates and confusion for each method's predictions."""
    truth = [r["drift_type"] for r in runs]
    types = present_types(runs)
    out: Dict[str, Any] = {}
    for name, pred in predictions.items():
        correct = [p == t for p, t in zip(pred, truth)]
        confusion = {a: {b: 0 for b in DRIFT_TYPES} for a in types}
        for p, t in zip(pred, truth):
            confusion[t][p] += 1
        out[name] = {
            "accuracy": _prop(sum(correct), len(correct)),
            "macro_f1": macro_f1(truth, pred, types),
            "per_type": {t: _prop(sum(c for c, tt in zip(correct, truth) if tt == t), truth.count(t)) for t in types},
            "confusion": confusion,
        }
    return out


def paired_method_tests(runs: List[Dict[str, Any]], predictions: Dict[str, List[str]],
                        reference: str = "ours", alpha: float = 0.05) -> Dict[str, Any]:
    """Ours minus each other method in the share of correctly typed windows, per
    replicate, with a sign-flip test and Holm correction across the comparisons."""
    truth = [r["drift_type"] for r in runs]
    reps = sorted({_rep(r) for r in runs})
    rep_of = [_rep(r) for r in runs]

    def share(pred):
        return np.array([np.mean([p == t for p, t, rr in zip(pred, truth, rep_of) if rr == rep]) for rep in reps])

    base = share(predictions[reference])
    others = [m for m in predictions if m != reference]
    tests = {m: sign_flip_test(base - share(predictions[m])) for m in others}
    rejected = holm_reject([tests[m]["p_value"] for m in others], alpha) if others else []
    for m, rej in zip(others, rejected):
        tests[m]["holm_significant"] = bool(rej)
    return tests


def rule_predictions(runs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    names = list(runs[0]["verdicts"]) if runs else []
    return {n: [r["verdicts"][n] for r in runs] for n in names}


def meta_classifier_predictions(train_runs: List[Dict[str, Any]], test_runs: Optional[List[Dict[str, Any]]] = None,
                                seed: int = 0) -> List[str]:
    """Learned diagnosis: boosted trees (scikit-learn defaults, no early stopping)
    on the window statistics of :func:`meta_features`.

    Without ``test_runs`` the predictions are leave-one-state-out on
    ``train_runs`` (the model never sees the state it predicts); with them, the
    model is fitted on all of ``train_runs`` and applied to ``test_runs``."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    def xy(rs):
        return np.array([meta_features(r["tests"]) for r in rs]), np.array([r["drift_type"] for r in rs])

    def fit(X, y):
        return HistGradientBoostingClassifier(early_stopping=False, random_state=seed).fit(X, y)

    X, y = xy(train_runs)
    if test_runs is not None:
        return fit(X, y).predict(xy(test_runs)[0]).tolist()
    states = np.array([r["state"] for r in train_runs])
    pred = np.empty(len(train_runs), dtype=object)
    for s in np.unique(states):
        held = states == s
        pred[held] = fit(X[~held], y[~held]).predict(X[held])
    return pred.tolist()


def methods(runs: List[Dict[str, Any]], alpha: float = 0.05) -> Dict[str, Any]:
    """Scores and paired tests for ours, the rule baselines, the ablations and the
    leave-one-state-out meta-classifier on one benchmark configuration."""
    if not runs:
        return {}
    preds = rule_predictions(runs)
    if len({r["state"] for r in runs}) >= 2:      # leave-one-state-out needs two states
        preds["meta_classifier"] = meta_classifier_predictions(runs)
    return {"scores": method_scores(runs, preds), "paired_tests": paired_method_tests(runs, preds, alpha=alpha),
            "meta_features": META_FEATURES, "meta_validation": "leave one state out"}


def estimators(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean absolute error of each label-free accuracy estimate, overall and by type,
    and CBPE minus each baseline in per-replicate mean absolute error."""
    if not runs or "estimates" not in runs[0]["estimator"]:
        return {}
    names = list(runs[0]["estimator"]["errors"])
    out: Dict[str, Any] = {"mae": {}, "bias": {}, "paired_vs_cbpe": {}}
    for n in names:
        by_rep: Dict[Any, List[float]] = defaultdict(list)
        signed: Dict[Any, List[float]] = defaultdict(list)
        for r in runs:
            by_rep[_rep(r)].append(abs(r["estimator"]["errors"][n]))
            signed[_rep(r)].append(r["estimator"]["errors"][n])
        out["mae"][n] = {"overall": _mean_ci(by_rep)}
        out["bias"][n] = {"overall": _mean_ci(signed)}
        for t in present_types(runs):
            sub: Dict[Any, List[float]] = defaultdict(list)
            for r in runs:
                if r["drift_type"] == t:
                    sub[_rep(r)].append(abs(r["estimator"]["errors"][n]))
            out["mae"][n][t] = _mean_ci(sub)
    reps = sorted({_rep(r) for r in runs})

    def rep_mae(n):
        return np.array([np.mean([abs(r["estimator"]["errors"][n]) for r in runs if _rep(r) == rep]) for rep in reps])

    others = [n for n in names if n != "cbpe"]
    for n in others:
        out["paired_vs_cbpe"][n] = sign_flip_test(rep_mae("cbpe") - rep_mae(n))
    rejected = holm_reject([out["paired_vs_cbpe"][n]["p_value"] for n in others], 0.05)
    for n, rej in zip(others, rejected):
        out["paired_vs_cbpe"][n]["holm_significant"] = bool(rej)
    return out


def detection_power(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Alarm rate of each detector by drift type and by type x intensity."""
    names = sorted({n for r in runs for n in r["detection"]})
    out: Dict[str, Any] = {}
    for n in names:
        res: Dict[str, Any] = {}
        for t in present_types(runs):
            sub = [r for r in runs if r["drift_type"] == t]
            res[t] = _prop(sum(r["detection"][n]["drift_detected"] for r in sub), len(sub))
            for lab in sorted({r["intensity_label"] for r in sub}):
                ss = [r for r in sub if r["intensity_label"] == lab]
                res[f"{t}/{lab}"] = _prop(sum(r["detection"][n]["drift_detected"] for r in ss), len(ss))
        out[n] = res
    return out
