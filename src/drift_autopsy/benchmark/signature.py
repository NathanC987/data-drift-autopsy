"""Turn a pipeline probe into a drift-type fingerprint and a verdict.

The signature is a handful of scalars, every one of them label-free, read
straight off the probe and the remediation triage. ``classify_drift_type``
applies a small transparent rule to the signature (and, when available, a tiny
delayed-label audit) and names the drift type. The thresholds are module
constants so they can be inspected and, if the reference distribution changes,
retuned.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---- decision thresholds (label-free signature) ------------------------- #
T_CONTROL_ESTIMATOR = 0.010     # |implied drop| below this and quiet detectors => "none"
T_CONTROL_PRIOR = 0.030
T_COVARIATE_KS = 0.150          # a localised KS statistic this large means a real marginal move
T_COVARIATE_AUC = 0.750        # ref-vs-prod domain separability for a covariate call
T_COVARIATE_KS_WEAK = 0.100    # weaker marginal move, needs audit confirmation
T_COVARIATE_AUC_WEAK = 0.640
T_PRIOR_PRED_SHIFT = 0.120     # shift in the model's own positive rate (label-free prior call)
T_PRIOR_MODEL_CORROBORATION = 0.030  # the model's rate must move too, to call an audited prior shift
T_SILENT_DETECTION = 0         # max detector severity rank at/below this => label-free stack silent
# ---- delayed-label audit thresholds ----------------------------------- #
T_AUDIT_DROP = 0.030           # measured accuracy drop that counts as real
T_AUDIT_PRIOR = 0.090          # audited class-prior move that counts as a prior shift
T_AUDIT_STRUCTURE = 0.150      # excess per-feature accuracy gap that counts as "localised"


def _detector_severity_rank(detection: Dict[str, Any], name: str) -> int:
    block = detection.get(name, {})
    return int(block.get("severity_rank", 0)) if "error" not in block else 0


def _union_localised(localisation: Dict[str, Any], detectors=("KS Test", "PSI", "MMD")) -> List[str]:
    out: List[str] = []
    for d in detectors:
        for f in localisation.get(d, {}).get("drifted_features", []):
            if f not in out:
                out.append(f)
    return out


def _max_localised_ks(localisation: Dict[str, Any]) -> float:
    ks = localisation.get("KS Test", {})
    drifted = set(ks.get("drifted_features", []))
    scores = ks.get("drift_scores", {})
    vals = [v for k, v in scores.items() if k in drifted]
    return float(max(vals)) if vals else 0.0


def risk_region_concentration(
    reliability: Dict[str, Any],
    production_X: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Largest |Spearman| between per-sample fused risk and a feature's rank.

    A structured (feature-localised) reliability response pushes this up; a flat
    one leaves it near zero. Purely label-free -- it only uses the risk scores
    the layer already produced and the feature values of the sampled rows.
    """
    records = reliability.get("records", [])
    idx = reliability.get("sample_indices", [])
    if not records or not idx or len(records) != len(idx):
        return {"score": 0.0, "feature": None}

    risk = np.array([r.get("risk_score") if r.get("risk_score") is not None else np.nan
                     for r in records], dtype=float)
    if np.all(~np.isfinite(risk)) or np.nanstd(risk) < 1e-9:
        return {"score": 0.0, "feature": None}

    sample = production_X.iloc[idx].reset_index(drop=True)
    best_score, best_feat = 0.0, None
    from scipy.stats import spearmanr

    ok = np.isfinite(risk)
    for f in feature_names:
        col = sample[f].to_numpy(dtype=float)[ok]
        if np.std(col) < 1e-9:
            continue
        rho, _ = spearmanr(col, risk[ok])
        if np.isfinite(rho) and abs(rho) > abs(best_score):
            best_score, best_feat = float(rho), f
    return {"score": abs(best_score), "signed": best_score, "feature": best_feat}


def drift_signature(
    probe: Dict[str, Any],
    triage: Dict[str, Any],
    reference_accuracy: float,
    production_X: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, Any]:
    """Assemble the label-free fingerprint."""
    detection = probe.get("detection", {})
    localisation = probe.get("localisation", {})
    reliability = probe.get("reliability", {})
    signals = triage.get("signals", {})

    max_input_sev = max(
        _detector_severity_rank(detection, n) for n in ("KS Test", "PSI", "MMD")
    )
    concentration = risk_region_concentration(reliability, production_X, feature_names)

    est_acc = float(probe.get("label_free_accuracy_estimate", reference_accuracy))
    implied_drop = float(reference_accuracy - est_acc)

    pred_shift = abs(
        float(probe.get("predicted_positive_rate_production", 0.0))
        - float(probe.get("predicted_positive_rate_reference", 0.0))
    )

    return {
        "input_detector_max_severity": int(max_input_sev),
        "input_drift_detected": bool(max_input_sev > 0),
        "n_features_localised": len(_union_localised(localisation)),
        "features_localised": _union_localised(localisation),
        "max_localised_ks_statistic": round(_max_localised_ks(localisation), 4),
        "cbpe_detected": bool(_detector_severity_rank(detection, "CBPE") > 0),
        "cbpe_severity_rank": _detector_severity_rank(detection, "CBPE"),
        "cbpe_confidence_shift": round(
            float(detection.get("CBPE", {}).get("confidence_shift", 0.0)), 4
        ),
        "domain_classifier_auc": round(float(signals.get("domain_classifier_auc", 0.5)), 4),
        "predicted_prior_shift": round(pred_shift, 4),
        "estimator_implied_drop": round(implied_drop, 4),
        "reliability_mean_ood": reliability.get("mean_ood"),
        "reliability_mean_calibration_risk": reliability.get("mean_calibration_risk"),
        "reliability_suspicious_pct": reliability.get("suspicious_pct"),
        "reliability_high_risk_pct": reliability.get("high_risk_pct"),
        "iw_probe_recovery": round(
            float(signals.get("cheap_probe_recovery", {}).get("importance_weighted_retrain", 0.0)), 4
        ),
        "feature_drop_sensitivity": round(float(signals.get("feature_drop_sensitivity", 0.0)), 4),
        "risk_region_concentration": round(float(concentration["score"]), 4),
        "risk_region_feature": concentration["feature"],
        "label_free_stack_silent": bool(
            max_input_sev <= T_SILENT_DETECTION
            and _detector_severity_rank(detection, "CBPE") <= T_SILENT_DETECTION
            and abs(implied_drop) < T_AUDIT_DROP
        ),
    }


def classify_drift_type(
    signature: Dict[str, Any],
    audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Name the drift type from the signature plus an optional labelled audit.

    The label-free signature resolves ``none``, ``covariate`` and (large moves
    of) ``prior``. When the label-free stack is silent but accuracy is falling
    -- the concept / label-noise regime -- the delayed-label audit is what
    separates a feature-localised target-rule change from unstructured label
    corruption or a base-rate move.

    Returns ``{predicted_type, confidence, rationale, stage}``.
    """
    s = signature
    audit = audit or {}
    audit_drop = float(audit.get("measured_drop", 0.0))
    audit_prior = abs(float(audit.get("prior_shift", 0.0)))
    audit_structure = float(audit.get("structure_score", 0.0))
    audit_feat = audit.get("structured_feature")
    real_drop = audit_drop >= T_AUDIT_DROP

    covariate_strong = (
        s["max_localised_ks_statistic"] >= T_COVARIATE_KS
        and s["domain_classifier_auc"] >= T_COVARIATE_AUC
    )
    covariate_weak = (
        s["max_localised_ks_statistic"] >= T_COVARIATE_KS_WEAK
        and s["domain_classifier_auc"] >= T_COVARIATE_AUC_WEAK
        and s["n_features_localised"] <= 4
    )

    # --- control -------------------------------------------------------
    if (
        not s["input_drift_detected"]
        and abs(s["estimator_implied_drop"]) < T_CONTROL_ESTIMATOR
        and s["predicted_prior_shift"] < T_CONTROL_PRIOR
        and not real_drop
        and audit_prior < T_AUDIT_PRIOR
    ):
        return _verdict("none", 0.9,
                        "every detector is quiet, the label-free estimate did not move, and a "
                        "labelled audit shows no accuracy loss", stage="label-free")

    # --- covariate: a marginal move on a small feature set, ref/prod separable
    if covariate_strong:
        r = (f"KS statistic {s['max_localised_ks_statistic']:.2f} on "
             f"{', '.join(s['features_localised'][:3]) or 'input features'}, domain AUC "
             f"{s['domain_classifier_auc']:.2f}")
        if s["iw_probe_recovery"] >= 0.2:
            r += f"; importance weighting recovers {s['iw_probe_recovery']:.0%} of the gap"
            return _verdict("covariate", 0.9, r, stage="label-free")
        return _verdict("covariate", 0.8, r, stage="label-free")

    # --- prior: the audited class prior moved a lot (or the model's own rate did,
    #     with the inputs staying broad-and-shallow rather than sharp)
    prior_label_free = (
        s["predicted_prior_shift"] >= T_PRIOR_PRED_SHIFT
        and not covariate_strong
        and s["max_localised_ks_statistic"] < T_COVARIATE_KS
    )
    # A genuine prior move shifts the model's OWN positive rate too; symmetric
    # label noise on an unbalanced base rate nudges the audited prior without
    # the model's predictions moving at all, so require both to agree.
    if (
        audit_prior >= T_AUDIT_PRIOR
        and s["predicted_prior_shift"] >= T_PRIOR_MODEL_CORROBORATION
        and not covariate_strong
    ):
        return _verdict("prior", 0.85,
                        f"a {audit.get('n_labels', 0)}-label audit shows the class base rate moved "
                        f"{audit.get('prior_shift', 0):+.2f} and the model's own positive rate moved "
                        f"with it ({s['predicted_prior_shift']:.2f}), with the loss spread across features",
                        stage="labelled-audit")
    if prior_label_free and not real_drop:
        return _verdict("prior", 0.7,
                        f"the model's positive rate shifted {s['predicted_prior_shift']:.2f} with "
                        f"only broad, shallow input drift (domain AUC {s['domain_classifier_auc']:.2f})",
                        stage="label-free")

    # --- covariate confirmed by the audit (a weaker marginal move) ----
    if covariate_weak and real_drop:
        return _verdict("covariate", 0.7,
                        f"a marginal move on {', '.join(s['features_localised'][:3])} (KS "
                        f"{s['max_localised_ks_statistic']:.2f}, AUC {s['domain_classifier_auc']:.2f}) "
                        f"confirmed by a {audit.get('n_labels', 0)}-label audit ({audit_drop:.1%} drop)",
                        stage="labelled-audit")

    # --- label-free stack is silent -----------------------------------
    if not real_drop:
        if s["label_free_stack_silent"]:
            return _verdict("none", 0.55,
                            "label-free monitors are all quiet and the audit shows no clear loss; "
                            "if accuracy is falling it is below the audit's resolution",
                            stage="labelled-audit" if audit else "label-free")
        return _verdict("concept", 0.4,
                        "detectors sit near their thresholds with the inputs unchanged - treat as a "
                        "possible target-rule change until more labels arrive", stage="label-free")

    # --- real drop, inputs unchanged: concept vs label noise ----------
    if audit_structure >= T_AUDIT_STRUCTURE:
        return _verdict("concept", 0.8,
                        f"a {audit.get('n_labels', 0)}-label audit shows accuracy down {audit_drop:.1%}, "
                        f"concentrated in the high-{audit_feat} range (excess gap "
                        f"{audit_structure:.2f}) while every input distribution is unchanged - the "
                        f"target rule for that region moved", stage="labelled-audit")

    return _verdict("label_noise", 0.75,
                    f"a {audit.get('n_labels', 0)}-label audit shows accuracy down {audit_drop:.1%}, "
                    f"spread evenly across features, with the inputs, the class prior and the model's "
                    f"own predictions all unchanged - the labels are corrupted, not the model",
                    stage="labelled-audit")


def _verdict(predicted: str, confidence: float, rationale: str, stage: str) -> Dict[str, Any]:
    return {
        "predicted_type": predicted,
        "confidence": round(float(confidence), 2),
        "rationale": rationale,
        "stage": stage,
    }
