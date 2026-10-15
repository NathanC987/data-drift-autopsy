"""Evaluate the pipeline on one injected production window and grade it.

Everything the monitor computes is label free on the production side, except
the delayed-label audit (a small labelled sample of the production window).
The holdout labels and the injection answer key are used only for grading.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.detectors import BBSD, CBPE, DomainClassifier, KSTest, MMD, PSI
from drift_autopsy.diagnosis import audit_diagnosis, combine, label_free_diagnosis
from drift_autopsy.diagnosis_baselines import all_rule_verdicts, window_tests
from drift_autopsy.estimation import label_free_estimates
from drift_autopsy.remediation import RemediationContext, remediation_triage
from drift_autopsy.remediation import strategies as S
from drift_autopsy.reliability import ReliabilityAnalyzer
from drift_autopsy.benchmark.spec import DRIFT_TYPES

# design B mechanism names (fixed order: it seeds the audit sample)
MECHANISMS_B = ("selection", "prior_decrease", "concept_oblique", "concept_threshold", "class_noise")


@dataclass
class ReplicateContext:
    """Everything fixed within one (state, seed) replicate."""

    state: str
    seed: int
    dataset: str
    model_kind: str
    model: Any
    model_factory: Any
    feature_names: List[str]
    categorical_features: List[str]
    train_X: pd.DataFrame
    train_y: np.ndarray
    reference: Dataset                 # calibration split with target and predictions
    reference_accuracy: float
    reliability: ReliabilityAnalyzer
    alpha: float
    psi_threshold: float
    mmd_permutations: int
    mmd_max_samples: int
    audit_sizes: Sequence[int]
    primary_audit_size: int


def _dataset(rc: ReplicateContext, X: pd.DataFrame) -> Dataset:
    X = X[rc.feature_names].reset_index(drop=True)
    proba = rc.model.predict_proba(X.to_numpy(dtype=float))
    return Dataset(
        data=X, feature_names=list(rc.feature_names),
        predictions=rc.model.classes_[proba.argmax(axis=1)], prediction_probabilities=proba,
        categorical_features=list(rc.categorical_features),
    )


def _detectors(rc: ReplicateContext, prod: Dataset) -> Dict[str, Any]:
    detectors = {
        "KS Test": KSTest(threshold=rc.alpha, correction="bonferroni"),
        "PSI": PSI(threshold=rc.psi_threshold),
        "MMD": MMD(threshold=rc.alpha, n_permutations=rc.mmd_permutations, max_samples=rc.mmd_max_samples,
                   standardize=True, random_state=rc.seed),
        "CBPE": CBPE(threshold=rc.alpha),
        "BBSD": BBSD(threshold=rc.alpha),
        "Domain classifier": DomainClassifier(threshold=rc.alpha, max_samples=rc.mmd_max_samples,
                                              random_state=rc.seed),
    }
    out = {}
    for name, det in detectors.items():
        res = det.fit_detect(rc.reference, prod)
        out[name] = {"drift_detected": bool(res.drift_detected), "score": float(res.score),
                     "p_value": None if res.p_value is None else float(res.p_value)}
        if name == "CBPE":
            out[name]["estimated_accuracy"] = float(res.metadata["estimated_accuracy"])
            out[name]["confidence_shift"] = float(res.metadata["confidence_shift_signed"])
    return out


def _compact_label_free(lf: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "verdict": lf["verdict"],
        "input_shift_features": lf["input_shift_features"],
        "residual_shift_features": lf["residual_shift_features"],
        "prediction_shift": lf["prediction_shift"],
        "estimated_production_prior": lf["bbse"]["estimated_production_prior"],
        "reference_prior": lf["bbse"]["reference_prior"],
        "multivariate_shift": lf["multivariate_shift"],
        "min_input_p_value": min([t["p_value"] for t in lf["input_tests"]] + [lf["prediction_shift_test"]["p_value"]]
                                 + ([lf["multivariate_test"]["p_value"]] if lf["multivariate_test"] else [])),
    }


def _compact_audit(au: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_labels": au["n_labels"],
        "audit_accuracy": au["audit_accuracy"],
        "accuracy_drop_p_value": au["accuracy_drop_test"]["p_value"],
        "accuracy_drop_significant": au["accuracy_drop_significant"],
        "audit_positive_rate": au["audit_positive_rate"],
        "structured_features": au["structured_features"],
        "error_structure_p_values": {r["feature"]: r["p_value"] for r in au["error_structure_tests"]},
    }


def evaluate_window(
    rc: ReplicateContext,
    design: str,
    mechanism: str,
    drift_type: str,
    intensity_label: str,
    intensity_order: int,
    params: Dict[str, Any],
    prod_X: pd.DataFrame, prod_y: np.ndarray,
    hold_X: pd.DataFrame, hold_y: np.ndarray,
    spec: Dict[str, Any],
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    prod = _dataset(rc, prod_X)
    detection = _detectors(rc, prod)
    timings["detectors"] = time.perf_counter() - t0

    # ---- diagnosis -------------------------------------------------------
    t0 = time.perf_counter()
    # the MMD detector above ran with the same settings and seed on the same pair: reuse it
    mmd = {"statistic": detection["MMD"]["score"], "p_value": detection["MMD"]["p_value"]}
    lf = label_free_diagnosis(rc.reference, prod, alpha=rc.alpha, mmd_permutations=rc.mmd_permutations,
                              mmd_max_samples=rc.mmd_max_samples, random_state=rc.seed, precomputed_mmd=mmd)
    # design A keeps the original audit seeding; design B adds the mechanism
    seed_key = [rc.seed, DRIFT_TYPES.index(drift_type), intensity_order]
    if design == "B":
        seed_key.append(1 + MECHANISMS_B.index(mechanism))
    rng = np.random.default_rng(seed_key)
    order = rng.permutation(len(prod_X))
    audits = {}
    for n in rc.audit_sizes:
        idx = order[: min(n, len(order))]       # nested label samples: 200 subset of 600 subset of 1000
        au = audit_diagnosis(rc.model, rc.reference, prod_X.iloc[idx], np.asarray(prod_y)[idx], alpha=rc.alpha)
        audits[n] = (au, combine(lf, au))
    primary_audit, primary = audits[rc.primary_audit_size]
    idx = order[: min(rc.primary_audit_size, len(order))]
    variants = {k: audit_diagnosis(rc.model, rc.reference, prod_X.iloc[idx], np.asarray(prod_y)[idx],
                                   alpha=rc.alpha, null=k)
                for k in ("none", "class_confidence")}
    timings["diagnosis"] = time.perf_counter() - t0

    # ---- label-free accuracy estimate --------------------------------------
    true_acc = float(np.mean(prod.predictions == np.asarray(prod_y)))
    est_acc = detection["CBPE"]["estimated_accuracy"]
    cal_correct = (rc.reference.predictions == np.asarray(rc.reference.target)).astype(float)
    estimates = label_free_estimates(rc.reference.prediction_probabilities, cal_correct,
                                     prod.prediction_probabilities)
    tests = window_tests(lf, primary_audit, est_acc - rc.reference_accuracy,
                         audit_variants={"no_confidence": variants["none"],
                                         "class_confidence": variants["class_confidence"]})
    verdicts = all_rule_verdicts(tests, rc.alpha)
    if verdicts["ours"] != primary.drift_type:
        raise AssertionError(f"recomputed verdict {verdicts['ours']} != {primary.drift_type}")

    # ---- reliability --------------------------------------------------------
    t0 = time.perf_counter()
    frame = rc.reliability.analyze(prod_X[rc.feature_names].to_numpy(dtype=float))
    reliability = ReliabilityAnalyzer.summarize(frame, prod_y)
    timings["reliability"] = time.perf_counter() - t0

    # ---- remediation triage and its ground truth ----------------------------
    t0 = time.perf_counter()
    ctx = RemediationContext(
        reference_X=rc.train_X.to_numpy(dtype=float), reference_y=rc.train_y,
        production_X=prod_X[rc.feature_names].to_numpy(dtype=float),
        holdout_X=hold_X[rc.feature_names].to_numpy(dtype=float), holdout_y=np.asarray(hold_y),
        feature_names=rc.feature_names, drifted_features=primary.localized_features,
        shift_name=f"{drift_type}/{intensity_label}", base_model=rc.model,
        model_factory=rc.model_factory, reference_accuracy=rc.reference_accuracy, seed=rc.seed,
    )
    triage = remediation_triage(ctx, alpha=rc.alpha)
    actual = S.importance_weighted_retrain(ctx, timing_repeats=1)
    timings["triage"] = time.perf_counter() - t0

    # ---- grading ------------------------------------------------------------
    expected = drift_type
    injected = set(spec.get("affected_features", []))
    localized = set(primary.localized_features)
    loc = None
    if injected:
        tp = len(localized & injected)
        loc = {"precision": tp / len(localized) if localized else 0.0, "recall": tp / len(injected),
               "localized": sorted(localized), "injected": sorted(injected)}

    return {
        "dataset": rc.dataset,
        "model": rc.model_kind,
        "design": design,
        "mechanism": mechanism,
        "state": rc.state,
        "seed": rc.seed,
        "drift_type": drift_type,
        "intensity_label": intensity_label,
        "intensity_order": intensity_order,
        "params": params,
        "n_production_rows": int(len(prod_X)),
        "ground_truth": spec,
        "true_accuracy": true_acc,
        "true_accuracy_drop": rc.reference_accuracy - true_acc,
        "detection": detection,
        "label_free": _compact_label_free(lf),
        "audit": _compact_audit(primary_audit),
        "diagnosis": primary.to_dict(),
        "diagnosis_correct": primary.drift_type == expected,
        "diagnosis_by_audit_size": {str(n): d.drift_type for n, (_, d) in audits.items()},
        "localization": loc,
        "estimator": {"estimated_accuracy": est_acc, "true_accuracy": true_acc, "error": est_acc - true_acc,
                      "estimates": estimates, "errors": {k: v - true_acc for k, v in estimates.items()}},
        "tests": tests,
        "verdicts": verdicts,
        "reliability": reliability,
        "triage": {
            "will_retraining_help": triage["will_retraining_help"],
            "probes": [{k: p[k] for k in ("probe", "difference", "ci_low", "ci_high", "p_value_greater",
                                          "predicted_gain_significant")} for p in triage["probes"]],
            "domain_classifier_auc": triage["signals"]["domain_classifier_auc"],
        },
        "triage_truth": {
            "actual_gain": actual.accuracy_recovered,
            "actual_gain_ci": [actual.recovery_test["ci_low"], actual.recovery_test["ci_high"]],
            "retraining_helps": actual.recovery_significant,
        },
        "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
    }
