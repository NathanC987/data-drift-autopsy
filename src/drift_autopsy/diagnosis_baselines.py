"""Baselines and ablations for the drift-type diagnosis.

Every method here names one of the five types (none, covariate, prior,
concept, label_noise) from the same test results the method in
:mod:`drift_autopsy.diagnosis` computes on a window, at the same significance
level and with the same audit labels. They are pure functions of a compact
record (:func:`window_tests`), so they are graded on exactly the same windows.

Baselines
* ``input_monitoring`` - the usual monitoring practice: an input drift alarm
  (feature tests or MMD, weighted Holm over the two blocks) is read as
  covariate shift; otherwise no drift. It never names the other types.
* ``bbse_first`` - label shift detection first (Lipton et al., 2018;
  Rabanser et al., 2019): a shift in the predicted-label distribution is read
  as prior shift; otherwise an input alarm is covariate shift; otherwise none.
  The same weighted Holm over the same three blocks as the method, so it is
  exactly the method without the label-shift consistency test.
* ``error_rate_monitoring`` - ``bbse_first`` plus performance monitoring on
  the audit labels (as in DDM, Gama et al., 2004): when no label-free test
  fires, a significant audit accuracy drop is read as concept drift. It cannot
  name label noise.
* ``meta_classifier`` - a learned diagnosis: gradient boosted trees on the
  window statistics of :func:`meta_features`, trained on windows with known
  types (see :mod:`drift_autopsy.benchmark.metrics`).

Ablations of the method
* ``no_mmd`` - the method without the multivariate block (weighted Holm over two blocks).
* ``no_confidence_null`` / ``class_confidence_null`` - the audit test with the
  alternative nulls of :func:`drift_autopsy.diagnosis.audit_diagnosis`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from drift_autopsy.diagnosis import label_free_decisions

TYPES = ("none", "covariate", "prior", "concept", "label_noise")
BASELINES = ("input_monitoring", "bbse_first", "error_rate_monitoring")
ABLATIONS = ("no_mmd", "no_confidence_null", "class_confidence_null")
METHOD_LABELS = {
    "ours": "Ours (hierarchical tests)",
    "input_monitoring": "Input monitoring",
    "bbse_first": "BBSE first",
    "error_rate_monitoring": "Error-rate monitoring",
    "meta_classifier": "Learned meta-classifier",
    "no_mmd": "Ours without MMD block",
    "no_confidence_null": "Ours, audit null without confidence",
    "class_confidence_null": "Ours, audit null with class x confidence",
}


def _audit_record(au: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "drop_p": float(au["accuracy_drop_test"]["p_value"]),
        "drop_significant": bool(au["accuracy_drop_significant"]),
        "accuracy_change": float(au["audit_accuracy"] - au["reference_accuracy"]),
        "error_p": {r["feature"]: float(r["p_value"]) for r in au["error_structure_tests"]},
        "structured_features": list(au["structured_features"]),
    }


def window_tests(lf: Dict[str, Any], audit: Dict[str, Any], cbpe_change: float,
                 audit_variants: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Compact record of every test result the methods need."""
    q = np.asarray(lf["bbse"]["estimated_production_prior"], dtype=float)
    p = np.asarray(lf["bbse"]["reference_prior"], dtype=float)
    return {
        "feature_p": {t["feature"]: float(t["p_value"]) for t in lf["input_tests"]},
        "prediction_p": float(lf["prediction_shift_test"]["p_value"]),
        "mmd_p": None if lf["multivariate_test"] is None else float(lf["multivariate_test"]["p_value"]),
        "residual_rejected": list(lf["residual_shift_features"]),
        "n_features": len(lf["input_tests"]),
        "bbse_prior_tv": float(0.5 * np.abs(q - p).sum()),
        "cbpe_change": float(cbpe_change),
        "audit": _audit_record(audit),
        "audit_variants": {k: _audit_record(v) for k, v in (audit_variants or {}).items()},
    }


def _blocks(t: Dict[str, Any], alpha: float, use_mmd: bool = True, use_prediction: bool = True) -> Dict[str, bool]:
    """Which blocks have a rejected test, under the method's weighted Holm procedure."""
    d = label_free_decisions(list(t["feature_p"].values()), t["prediction_p"] if use_prediction else None,
                             t["mmd_p"] if use_mmd else None, alpha)
    out = {"univariate": d["univariate"]}
    if use_prediction:
        out["prediction"] = d["prediction"]
    if use_mmd and t["mmd_p"] is not None:
        out["mmd"] = d["multivariate"]
    return out


def _audit_type(audit: Dict[str, Any]) -> str:
    if not audit["drop_significant"]:
        return "none"
    return "concept" if audit["structured_features"] else "label_noise"


def ours(t: Dict[str, Any], alpha: float, use_mmd: bool = True, audit_key: Optional[str] = None) -> str:
    """The method of :mod:`drift_autopsy.diagnosis`, recomputed from the record.

    Equal to the stored verdict by construction (checked in the tests); the
    arguments give the ablations."""
    rej = _blocks(t, alpha, use_mmd=use_mmd)
    if any(rej.values()):
        if t["residual_rejected"]:
            return "covariate"
        return "prior" if rej["prediction"] else "covariate"
    audit = t["audit"] if audit_key is None else t["audit_variants"][audit_key]
    return _audit_type(audit)


def input_monitoring(t: Dict[str, Any], alpha: float) -> str:
    rej = _blocks(t, alpha, use_prediction=False)
    return "covariate" if any(rej.values()) else "none"


def bbse_first(t: Dict[str, Any], alpha: float) -> str:
    rej = _blocks(t, alpha)
    if rej["prediction"]:
        return "prior"
    return "covariate" if any(rej.values()) else "none"


def error_rate_monitoring(t: Dict[str, Any], alpha: float) -> str:
    v = bbse_first(t, alpha)
    if v == "none" and t["audit"]["drop_significant"]:
        return "concept"
    return v


def all_rule_verdicts(t: Dict[str, Any], alpha: float) -> Dict[str, str]:
    """Verdicts of the method, the rule baselines and the ablations for one window."""
    out = {
        "ours": ours(t, alpha),
        "input_monitoring": input_monitoring(t, alpha),
        "bbse_first": bbse_first(t, alpha),
        "error_rate_monitoring": error_rate_monitoring(t, alpha),
        "no_mmd": ours(t, alpha, use_mmd=False),
    }
    for key, name in (("no_confidence", "no_confidence_null"), ("class_confidence", "class_confidence_null")):
        if key in t["audit_variants"]:
            out[name] = ours(t, alpha, audit_key=key)
    return out


META_FEATURES: List[str] = [
    "univariate_evidence", "prediction_evidence", "mmd_evidence", "residual_share",
    "bbse_prior_tv", "cbpe_change", "audit_drop_evidence", "audit_accuracy_change", "error_structure_evidence",
]


def _evidence(p: Optional[float]) -> float:
    return 0.0 if p is None else float(-np.log10(max(p, 1e-300)))


def meta_features(t: Dict[str, Any]) -> List[float]:
    """Window statistics for the learned meta-classifier (no tuning: raw evidence values)."""
    fp = np.array(list(t["feature_p"].values()), dtype=float)
    err = list(t["audit"]["error_p"].values())
    return [
        _evidence(min(1.0, len(fp) * float(fp.min())) if fp.size else 1.0),
        _evidence(t["prediction_p"]),
        _evidence(t["mmd_p"]),
        len(t["residual_rejected"]) / max(t["n_features"], 1),
        t["bbse_prior_tv"],
        t["cbpe_change"],
        _evidence(t["audit"]["drop_p"]),
        t["audit"]["accuracy_change"],
        _evidence(min(1.0, len(err) * min(err)) if err else 1.0),
    ]
