"""
ACS Income drift study (temporal and geographic).

Monitored model: the shared ACS model (drift_autopsy.experiments.acs), trained
on 60% of California 2014. The other 40% is split into a calibration set (the
reference window for every detector and the CBPE accuracy map) and a test set
(the reference accuracy). Production windows:

* control    - the California 2014 test split (same distribution; nothing
               should fire, which checks the false alarm behaviour);
* temporal   - California 2015, 2016, 2017 and 2018 (full years);
* geographic - Texas, New York, Florida and Washington 2014 (full years).

For each window the script records the detector decisions, the feature
localisation, the SHAP attribution changes with bootstrap intervals, the CBPE
accuracy estimate, the drift-type diagnosis (label-free stage plus a
delayed-label audit of AUDIT_LABELS rows), per-prediction reliability over the
whole window and, for evaluation only, the true accuracy. Output: outputs/folktables_drift_results.json.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from drift_autopsy import DriftPipeline
from drift_autopsy.detectors import CBPE, KSTest, MMD, PSI
from drift_autopsy.diagnosis import audit_diagnosis, combine, label_free_diagnosis
from drift_autopsy.experiments.acs import (
    ACS_CATEGORICAL,
    ACS_FEATURES,
    ACS_NUMERIC,
    SPLIT_FRACTIONS,
    load_acs,
    make_acs_model,
    split_reference,
    to_dataset,
)
from drift_autopsy.localizers import UnivariateLocalizer
from drift_autopsy.rca import SHAPAnalyzer
from drift_autopsy.reliability import ReliabilityAnalyzer
from drift_autopsy.stats import two_proportion_test
from drift_autopsy.utils import setup_logging

SEED = 42
ALPHA = 0.05                 # significance level of every test
PSI_THRESHOLD = 0.2          # conventional "significant shift" PSI level (Siddiqi, 2006)
MMD_PERMUTATIONS = 200
MMD_MAX_SAMPLES = 3000
SHAP_BACKGROUND = 100
SHAP_EXPLAINED = 1000
RELIABILITY_RECORDS_KEPT = 1000   # per-prediction rows stored for the dashboard
AUDIT_LABELS = 600                # delayed labels for the audit stage of the diagnosis

BASE_STATE, BASE_YEAR = "CA", 2014
TEMPORAL_YEARS = [2015, 2016, 2017, 2018]
GEO_STATES = ["TX", "NY", "FL", "WA"]
OUTPUT = Path("outputs/folktables_drift_results.json")


def make_pipelines(model) -> Dict[str, DriftPipeline]:
    """KS carries localisation and SHAP root cause analysis; the others detect only."""
    return {
        "KS Test": DriftPipeline(
            detector=KSTest(threshold=ALPHA, correction="bonferroni"),
            localizer=UnivariateLocalizer(threshold=ALPHA, correction="holm"),
            rca=SHAPAnalyzer(
                n_background_samples=SHAP_BACKGROUND, n_test_samples=SHAP_EXPLAINED,
                random_state=SEED, alpha=ALPHA,
            ),
            model=model,
            enable_localization=True,
            enable_rca=True,
        ),
        "PSI": DriftPipeline(detector=PSI(threshold=PSI_THRESHOLD, n_bins=10)),
        "MMD": DriftPipeline(
            detector=MMD(
                threshold=ALPHA, n_permutations=MMD_PERMUTATIONS, max_samples=MMD_MAX_SAMPLES,
                standardize=True, random_state=SEED,
            )
        ),
        "CBPE": DriftPipeline(detector=CBPE(threshold=ALPHA, n_bins=10)),
    }


def explanation_shift(rca: Dict[str, Any]) -> Dict[str, float]:
    """Cosine distance and relative L1 change between reference and test |SHAP| profiles."""
    changes = rca.get("distribution_changes") or {}
    if not changes:
        return {}
    ref = np.array([v["ref_importance"] for v in changes.values()])
    test = np.array([v["test_importance"] for v in changes.values()])
    cosine = 1.0 - float(ref @ test / (np.linalg.norm(ref) * np.linalg.norm(test)))
    return {"cosine_distance": cosine, "relative_l1_change": float(np.abs(test - ref).sum() / ref.sum())}


def analyse_window(
    key: str, analysis_type: str, X, y, *, model, pipelines, reference_ds, reliability,
    reference_accuracy: float, reference_test_correct: np.ndarray, rng: np.random.Generator,
) -> Dict[str, Any]:
    print(f"\n=== {key} ({analysis_type}, {len(X)} rows)")
    window_ds = to_dataset(X, y, model=model)
    timings: Dict[str, float] = {}

    pipelines_out = {}
    for name, pipe in pipelines.items():
        start = time.perf_counter()
        res = pipe.run(reference_ds, window_ds)
        timings[name] = round(time.perf_counter() - start, 3)
        pipelines_out[name] = res.to_dict()
        det = res.detection
        p = det.p_value if det.p_value is not None else float("nan")
        print(f"  {name:8s} drift={det.drift_detected!s:5s} severity={det.severity.value:8s} "
              f"score={det.score:.4g} p={p:.3g}")

    correct = (window_ds.predictions == np.asarray(y)).astype(int)
    accuracy = float(correct.mean())
    drop_test = two_proportion_test(
        int(reference_test_correct.sum()), len(reference_test_correct), int(correct.sum()), len(correct),
        alternative="greater",
    )
    estimated = float(pipelines_out["CBPE"]["detection"]["metadata"]["estimated_accuracy"])

    start = time.perf_counter()
    frame = reliability.analyze(X[ACS_FEATURES].to_numpy(dtype=float))
    timings["reliability"] = round(time.perf_counter() - start, 3)
    summary = ReliabilityAnalyzer.summarize(frame, y)
    keep = np.sort(rng.choice(len(frame), min(RELIABILITY_RECORDS_KEPT, len(frame)), replace=False))
    records = frame.iloc[keep].assign(correct=correct[keep], row=keep)
    records["risk_label"] = records["risk_label"].astype(str)

    start = time.perf_counter()
    lf = label_free_diagnosis(reference_ds, window_ds, alpha=ALPHA, mmd_permutations=MMD_PERMUTATIONS,
                              mmd_max_samples=MMD_MAX_SAMPLES, random_state=SEED)
    audit_idx = rng.choice(len(X), min(AUDIT_LABELS, len(X)), replace=False)
    au = audit_diagnosis(model, reference_ds, X.iloc[audit_idx], np.asarray(y)[audit_idx], alpha=ALPHA)
    diagnosis = combine(lf, au)
    timings["diagnosis"] = round(time.perf_counter() - start, 3)
    print(f"  diagnosis: {diagnosis.drift_type} [{diagnosis.stage}] {diagnosis.rationale}")

    print(f"  accuracy {accuracy:.4f} (change {accuracy - reference_accuracy:+.4f}, "
          f"one-sided p={drop_test['p_value']:.3g}); CBPE estimate {estimated:.4f}; "
          f"risk AUROC {summary.get('auroc_risk', float('nan')):.3f} vs confidence-only "
          f"{summary.get('auroc_confidence_only', float('nan')):.3f}")

    return {
        "analysis_type": analysis_type,
        "n_rows": int(len(X)),
        "positive_rate": float(np.mean(y)),
        "predicted_positive_rate": float(np.mean(window_ds.predictions)),
        "actual_accuracy": accuracy,
        "accuracy_change": accuracy - reference_accuracy,
        "accuracy_drop_test": drop_test,
        "cbpe_estimated_accuracy": estimated,
        "cbpe_estimate_error": estimated - accuracy,
        "pipelines": pipelines_out,
        "explanation_shift": explanation_shift(pipelines_out["KS Test"].get("rca") or {}),
        "diagnosis": {
            "final": diagnosis.to_dict(),
            "label_free_verdict": lf["verdict"],
            "input_shift_features": lf["input_shift_features"],
            "residual_shift_features": lf["residual_shift_features"],
            "prediction_shift": lf["prediction_shift"],
            "multivariate_shift": lf["multivariate_shift"],
            "bbse": lf["bbse"],
            "audit": {k: au[k] for k in ("n_labels", "audit_accuracy", "accuracy_drop_significant",
                                         "audit_positive_rate", "structured_features")},
            "audit_accuracy_drop_p_value": au["accuracy_drop_test"]["p_value"],
            "audit_prior_p_value": au["prior_test"]["p_value"],
        },
        "reliability": {"summary": summary, "records": records.to_dict(orient="records")},
        "timings_seconds": timings,
    }


def main() -> None:
    setup_logging(level="WARNING")
    rng = np.random.default_rng(SEED)

    X_base, y_base = load_acs(BASE_YEAR, BASE_STATE)
    split = split_reference(X_base, y_base, seed=SEED)
    model = make_acs_model(random_state=SEED).fit(split.train_X.to_numpy(dtype=float), split.train_y)

    reference_ds = to_dataset(split.calibration_X, split.calibration_y, model=model)
    test_pred = model.predict(split.test_X.to_numpy(dtype=float))
    reference_test_correct = (test_pred == split.test_y).astype(int)
    reference_accuracy = float(reference_test_correct.mean())
    print(f"split {split.sizes()}; reference (held-out test) accuracy {reference_accuracy:.4f}")

    reliability = ReliabilityAnalyzer(
        model,
        reference_X=split.train_X.to_numpy(dtype=float),
        calibration_X=split.calibration_X.to_numpy(dtype=float),
        calibration_y=split.calibration_y,
        categorical_indices=[ACS_FEATURES.index(f) for f in ACS_CATEGORICAL],
        random_state=SEED,
    )
    print(f"reliability fusion: {reliability.describe()['fusion']}")

    pipelines = make_pipelines(model)
    common = dict(
        model=model, pipelines=pipelines, reference_ds=reference_ds, reliability=reliability,
        reference_accuracy=reference_accuracy, reference_test_correct=reference_test_correct, rng=rng,
    )

    windows: Dict[str, Any] = {}
    key = f"{BASE_STATE} {BASE_YEAR} held-out"
    windows[key] = {"state": BASE_STATE, "year": BASE_YEAR,
                    **analyse_window(key, "control", split.test_X, split.test_y, **common)}
    for year in TEMPORAL_YEARS:
        X, y = load_acs(year, BASE_STATE)
        key = f"{BASE_STATE} {year}"
        windows[key] = {"state": BASE_STATE, "year": year, **analyse_window(key, "temporal", X, y, **common)}
    for state in GEO_STATES:
        X, y = load_acs(BASE_YEAR, state)
        key = f"{state} {BASE_YEAR}"
        windows[key] = {"state": state, "year": BASE_YEAR, **analyse_window(key, "geographic", X, y, **common)}

    result = {
        "config": {
            "seed": SEED,
            "alpha": ALPHA,
            "psi_threshold": PSI_THRESHOLD,
            "mmd": {"n_permutations": MMD_PERMUTATIONS, "max_samples": MMD_MAX_SAMPLES, "standardize": True},
            "shap": {"n_background": SHAP_BACKGROUND, "n_explained_per_sample": SHAP_EXPLAINED},
            "audit_labels": AUDIT_LABELS,
            "ks_correction": "bonferroni",
            "localization_correction": "holm",
            "reference": f"{BASE_STATE} {BASE_YEAR}",
            "split_fractions": dict(zip(("train", "calibration", "test"), SPLIT_FRACTIONS)),
            "split_sizes": split.sizes(),
            "features": {"numeric": ACS_NUMERIC, "categorical": ACS_CATEGORICAL},
            "model": "one-hot nominal + standardised numeric, L2 logistic regression (C=1)",
        },
        "reference": {
            "test_accuracy": reference_accuracy,
            "train_positive_rate": float(np.mean(split.train_y)),
            "calibration_accuracy": float(np.mean(reference_ds.predictions == split.calibration_y)),
            "calibration_mean_confidence": float(
                model.predict_proba(split.calibration_X.to_numpy(dtype=float)).max(axis=1).mean()
            ),
        },
        "reliability_model": reliability.describe(),
        "windows": windows,
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=1, allow_nan=False, default=_json_default))
    print(f"\nwrote {OUTPUT}")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


if __name__ == "__main__":
    main()
