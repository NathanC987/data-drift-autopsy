"""Run the full drift-autopsy pipeline on one reference/production pair.

This is the same sequence the demos run -- detection with four detectors,
univariate localisation, SHAP root cause, a label-free accuracy estimate, and
the per-prediction reliability layer -- packaged so the benchmark can call it
once per injected window and grade the output. Nothing here knows which drift
was injected.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from drift_autopsy import Dataset, DriftPipeline
from drift_autopsy.detectors import CBPE, MMD, PSI, KSTest
from drift_autopsy.rca import SHAPAnalyzer
from drift_autopsy.reliability import ReliabilityAnalyzer

logger = logging.getLogger(__name__)

DETECTOR_THRESHOLDS = {"KS Test": 0.05, "PSI": 0.2, "MMD": 0.1, "CBPE": 0.05}
LOCALIZATION_THRESHOLD = 0.05
_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


def label_free_accuracy_estimate(
    ref_confidence: np.ndarray,
    ref_correct: np.ndarray,
    prod_confidence: np.ndarray,
    n_bins: int = 10,
) -> float:
    """CBPE-style estimate: calibrate per-confidence-bin accuracy on the
    reference, then average it over the production confidence histogram."""
    ref_confidence = np.asarray(ref_confidence, dtype=float)
    ref_correct = np.asarray(ref_correct, dtype=float)
    prod_confidence = np.asarray(prod_confidence, dtype=float)
    if ref_confidence.size == 0 or prod_confidence.size == 0:
        return 0.5

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ref_bin = np.clip(np.digitize(ref_confidence, edges[1:-1]), 0, n_bins - 1)
    prod_bin = np.clip(np.digitize(prod_confidence, edges[1:-1]), 0, n_bins - 1)

    global_acc = float(ref_correct.mean())
    bin_acc = np.full(n_bins, global_acc, dtype=float)
    for b in range(n_bins):
        mask = ref_bin == b
        if np.any(mask):
            bin_acc[b] = float(ref_correct[mask].mean())
    return float(np.clip(np.mean(bin_acc[prod_bin]), 0.0, 1.0))


def _make_pipelines(model_for_shap: Any) -> Dict[str, DriftPipeline]:
    return {
        "KS Test": DriftPipeline(
            detector=KSTest(threshold=0.05, correction="bonferroni"),
            localizer="univariate", rca="shap", model=model_for_shap,
            enable_localization=True, enable_rca=True,
        ),
        "PSI": DriftPipeline(
            detector=PSI(threshold=0.2, n_bins=10),
            localizer="univariate", enable_localization=True, enable_rca=False,
        ),
        "MMD": DriftPipeline(
            detector=MMD(threshold=0.1, kernel="rbf", n_permutations=20, max_samples=3000),
            localizer="univariate", enable_localization=True, enable_rca=False,
        ),
        "CBPE": DriftPipeline(
            detector=CBPE(threshold=0.05, n_bins=10),
            localizer="univariate", enable_localization=True, enable_rca=False,
        ),
    }


def _detection_block(result) -> Dict[str, Any]:
    det = result.detection
    return {
        "drift_detected": bool(det.drift_detected),
        "severity": det.severity.value,
        "severity_rank": _SEVERITY_RANK.get(det.severity.value, 0),
        "score": float(det.score),
        "p_value": None if det.p_value is None else float(det.p_value),
        "confidence_shift": float(det.metadata.get("confidence_shift", 0.0)),
    }


def run_pipeline_probe(
    reference_X: pd.DataFrame,
    production_X: pd.DataFrame,
    model: Any,
    feature_names: Sequence[str],
    reference_predictions_cache: Dict[str, np.ndarray],
    reliability_reference: pd.DataFrame,
    reliability_sample: int = 150,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run detection + localisation + RCA + estimate + reliability.

    Everything here is label-free, so the output depends only on the inputs and
    can be reused across windows that share an X. ``reference_predictions_cache``
    holds the model's confidence/correctness on the reference window;
    ``reliability_reference`` is the frame the reliability layer fits its OOD and
    confidence baselines on. Per-window reliability grading (which needs labels)
    is done afterwards by :func:`grade_reliability`.
    """
    feature_names = list(feature_names)
    reference_X = reference_X.reset_index(drop=True)[feature_names]
    production_X = production_X.reset_index(drop=True)[feature_names]
    rng = np.random.default_rng(seed)
    started = time.time()

    ref_np = reference_X.to_numpy(dtype=float)
    prod_np = production_X.to_numpy(dtype=float)

    ref_proba = model.predict_proba(ref_np)
    prod_proba = model.predict_proba(prod_np)
    prod_pred = model.predict(prod_np)
    ref_pred = model.predict(ref_np)

    ref_ds = Dataset(data=reference_X.copy(), feature_names=feature_names)
    prod_ds = Dataset(data=production_X.copy(), feature_names=feature_names)
    ref_ds_pred = Dataset(
        data=reference_X.copy(), feature_names=feature_names,
        predictions=ref_pred, prediction_probabilities=ref_proba,
    )
    prod_ds_pred = Dataset(
        data=production_X.copy(), feature_names=feature_names,
        predictions=prod_pred, prediction_probabilities=prod_proba,
    )

    pipelines = _make_pipelines(model)

    detection: Dict[str, Any] = {}
    localisation: Dict[str, Any] = {}
    rca: Dict[str, Any] = {}
    for name, pipe in pipelines.items():
        try:
            if name == "CBPE":
                res = pipe.run(ref_ds_pred, prod_ds_pred,
                               detection_threshold=DETECTOR_THRESHOLDS[name],
                               localization_threshold=LOCALIZATION_THRESHOLD)
            else:
                res = pipe.run(ref_ds, prod_ds,
                               detection_threshold=DETECTOR_THRESHOLDS[name],
                               localization_threshold=LOCALIZATION_THRESHOLD)
        except Exception as exc:  # a detector failing should not sink the probe
            logger.warning("probe: detector %s failed: %s", name, exc)
            detection[name] = {"error": str(exc)}
            continue

        detection[name] = _detection_block(res)
        if res.localization is not None:
            localisation[name] = {
                "drifted_features": list(res.localization.drifted_features),
                "drift_scores": {k: float(v) for k, v in res.localization.drift_scores.items()},
            }
        if name == "KS Test" and res.rca is not None:
            expl = res.rca.explanations or {}
            rca = {
                "top_importance_changes": [
                    [f, float(c)] for f, c in expl.get("top_importance_changes", [])
                ],
                "feature_importances": {
                    k: float(v) for k, v in (res.rca.feature_importances or {}).items()
                },
                "recommendations": list(res.rca.recommendations),
            }

    # ---- label-free accuracy estimate -----------------------------------
    ref_conf = reference_predictions_cache["confidence"]
    ref_correct = reference_predictions_cache["correct"]
    prod_conf = prod_proba.max(axis=1)
    est_acc = label_free_accuracy_estimate(ref_conf, ref_correct, prod_conf)

    # ---- predicted-label distribution shift ----------------------------
    ref_pred_rate = float(np.mean(ref_pred))
    prod_pred_rate = float(np.mean(prod_pred))

    # ---- reliability layer (label-free; per-window grading added later) --
    reliability = _run_reliability(
        model=model,
        reliability_reference=reliability_reference[feature_names],
        production_X=production_X,
        estimated_accuracy=est_acc,
        n_sample=reliability_sample,
        rng=rng,
    )

    return {
        "detection": detection,
        "localisation": localisation,
        "rca": rca,
        "label_free_accuracy_estimate": float(est_acc),
        "predicted_positive_rate_reference": ref_pred_rate,
        "predicted_positive_rate_production": prod_pred_rate,
        "reliability": reliability,
        "probe_seconds": round(time.time() - started, 2),
    }


def _run_reliability(
    model: Any,
    reliability_reference: pd.DataFrame,
    production_X: pd.DataFrame,
    estimated_accuracy: float,
    n_sample: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    n = min(n_sample, len(production_X))
    idx = np.sort(rng.choice(len(production_X), n, replace=False))
    sample_X = production_X.iloc[idx].reset_index(drop=True)

    try:
        analyzer = ReliabilityAnalyzer(
            model=model,
            data_type="tabular",
            reference_data=reliability_reference.reset_index(drop=True),
            task_type="classification",
            cbpe_reference_score=float(estimated_accuracy),
        )
        records = analyzer.analyze_batch(sample_X)
    except Exception as exc:
        logger.warning("probe: reliability layer failed: %s", exc)
        return {"error": str(exc), "n": 0, "records": [], "sample_indices": []}

    per_record: List[Dict[str, Any]] = []
    for rec in records:
        per_record.append({
            "confidence": _f(rec.get("confidence")),
            "ood": _f(rec.get("ood")),
            "stability": _f(rec.get("stability")),
            "calibration": rec.get("calibration"),
            "calibration_risk": _f(rec.get("calibration_risk")),
            "explanation": _f(rec.get("explanation")),
            "risk_score": _f(rec.get("risk_score")),
            "risk_label": rec.get("risk_label"),
        })

    susp = sum(1 for r in per_record if r["calibration"] == "suspicious")
    high = sum(1 for r in per_record if r["risk_label"] == "HIGH")

    return {
        "n": len(per_record),
        "mean_confidence": _nan_mean([r["confidence"] for r in per_record]),
        "mean_ood": _nan_mean([r["ood"] for r in per_record]),
        "mean_stability": _nan_mean([r["stability"] for r in per_record]),
        "mean_calibration_risk": _nan_mean([r["calibration_risk"] for r in per_record]),
        "mean_risk": _nan_mean([r["risk_score"] for r in per_record]),
        "suspicious_pct": round(100.0 * susp / max(len(per_record), 1), 2),
        "high_risk_pct": round(100.0 * high / max(len(per_record), 1), 2),
        "sample_indices": idx.tolist(),
        "records": per_record,
    }


def grade_reliability(
    reliability: Dict[str, Any],
    model: Any,
    production_X: pd.DataFrame,
    production_y: np.ndarray,
    feature_names: Sequence[str],
) -> Dict[str, Any]:
    """Attach per-window correctness to reliability records and score the risk
    layer as a label-free error detector (AUROC of risk vs prediction error)."""
    records = reliability.get("records", [])
    idx = reliability.get("sample_indices", [])
    if not records or not idx:
        return reliability

    feature_names = list(feature_names)
    production_X = production_X.reset_index(drop=True)[feature_names]
    production_y = np.asarray(production_y).astype(int)
    sample_X = production_X.iloc[idx]
    sample_y = production_y[idx]
    preds = model.predict(sample_X.to_numpy(dtype=float))
    correct = (preds == sample_y).astype(int)

    graded_records = [dict(r) for r in records]
    for rec, c in zip(graded_records, correct):
        rec["correct"] = int(c)

    risk = np.array(
        [r["risk_score"] for r in graded_records if r["risk_score"] is not None], dtype=float
    )
    errors = np.array(
        [1 - r["correct"] for r in graded_records if r["risk_score"] is not None], dtype=int
    )
    graded = dict(reliability)
    graded["records"] = graded_records
    graded["sample_accuracy"] = round(float(np.mean(correct)), 4)
    graded["auroc_risk_vs_error"] = _safe_auroc(errors, risk)
    return graded


def _f(value: Any) -> Optional[float]:
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _nan_mean(values: Sequence[Optional[float]]) -> Optional[float]:
    arr = np.array([v for v in values if v is not None], dtype=float)
    return round(float(arr.mean()), 4) if arr.size else None


def _safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if len(y_true) < 8 or len(np.unique(y_true)) < 2 or scores.size != y_true.size:
        return None
    from sklearn.metrics import roc_auc_score

    try:
        return round(float(roc_auc_score(y_true, scores)), 4)
    except ValueError:
        return None
