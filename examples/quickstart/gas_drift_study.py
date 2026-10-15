"""
Gas sensor drift study (UCI Gas Sensor Array Drift, ten batches over 36 months).

Setup (drift_autopsy.experiments.gas): batches 1 and 2 are split 60 / 20 / 20
into train / calibration / test; the monitored model is trained on the train
split. The calibration split is the reference window for every detector and
fits the CBPE map. Windows: the reference test split (no-drift control) and
batches 3 to 10, each compared with the same fixed reference.

The drift here is known: the sensors age, so their response to the same gas
changes (P(X | Y) moves) while the gas labels stay exact. The gas mix also
differs between batches (toluene is absent from batches 3 to 5).

Per window: every detector (decided by a test at ALPHA), feature localisation
(Holm over the 128 features) aggregated by sensor, the CBPE accuracy estimate
and its baselines, per-prediction reliability, the label-free diagnosis plus a
delayed-label audit of min(600, n / 2) measurements with the baseline and
ablation verdicts, and the true accuracy for evaluation only.

Run:  python examples/quickstart/gas_drift_study.py --model lr
Output: outputs/gas_drift_results_{model}.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from drift_autopsy.detectors import CBPE, KSTest, MMD, PSI
from drift_autopsy.diagnosis import audit_diagnosis, combine, label_free_diagnosis
from drift_autopsy.experiments import gas
from drift_autopsy.experiments.common import baseline_detectors, comparison_block, json_default, window_rng
from drift_autopsy.experiments.models import MODEL_DESCRIPTIONS
from drift_autopsy.localizers import UnivariateLocalizer
from drift_autopsy.reliability import ReliabilityAnalyzer
from drift_autopsy.stats import two_proportion_test
from drift_autopsy.utils import setup_logging

SEED = 42
ALPHA = 0.05
PSI_THRESHOLD = 0.2
MMD_PERMUTATIONS = 200
MMD_MAX_SAMPLES = 3000
RELIABILITY_RECORDS_KEPT = 1000


def detectors() -> Dict[str, Any]:
    return {
        "KS Test": KSTest(threshold=ALPHA, correction="bonferroni"),
        "PSI": PSI(threshold=PSI_THRESHOLD, n_bins=10),
        "MMD": MMD(threshold=ALPHA, n_permutations=MMD_PERMUTATIONS, max_samples=MMD_MAX_SAMPLES,
                   standardize=True, random_state=SEED),
        "CBPE": CBPE(threshold=ALPHA, n_bins=10),
        **baseline_detectors(ALPHA, SEED, max_samples=MMD_MAX_SAMPLES),
    }


def sensor_localization(loc) -> Dict[str, Any]:
    """Drifted features (Holm over all 128) counted per sensor, with the mean KS statistic."""
    rows = {s: {"n_drifted": 0, "n_features": 0, "statistics": []} for s in range(1, gas.N_SENSORS + 1)}
    drifted = set(loc.drifted_features)
    for fd in loc.feature_drifts:
        r = rows[gas.sensor_of(fd.feature_name)]
        r["n_features"] += 1
        r["statistics"].append(float(fd.score))
        r["n_drifted"] += int(fd.feature_name in drifted)
    return {str(s): {"n_drifted": r["n_drifted"], "n_features": r["n_features"],
                     "mean_statistic": float(np.mean(r["statistics"]))} for s, r in rows.items()}


def analyse_window(key: str, analysis_type: str, frame: pd.DataFrame, *, model, reference_ds, reliability,
                   reference_accuracy: float, reference_test_correct: np.ndarray) -> Dict[str, Any]:
    print(f"\n=== {key} ({analysis_type}, {len(frame)} measurements)")
    rng = window_rng(SEED, key)
    window_ds = gas.to_dataset(frame, model=model, with_target=False)
    timings: Dict[str, float] = {}

    detection = {}
    for name, det in detectors().items():
        start = time.perf_counter()
        res = det.fit_detect(reference_ds, window_ds)
        timings[name] = round(time.perf_counter() - start, 3)
        detection[name] = res.to_dict()
        p = res.p_value if res.p_value is not None else float("nan")
        print(f"  {name:18s} drift={res.drift_detected!s:5s} score={res.score:.4g} p={p:.3g}")

    loc = UnivariateLocalizer(threshold=ALPHA, correction="holm").localize(reference_ds, window_ds)

    y = frame["gas"].to_numpy()
    correct = (window_ds.predictions == y).astype(int)
    accuracy = float(correct.mean())
    drop = two_proportion_test(int(reference_test_correct.sum()), len(reference_test_correct),
                               int(correct.sum()), len(correct), alternative="greater")
    estimated = float(detection["CBPE"]["metadata"]["estimated_accuracy"])

    start = time.perf_counter()
    rel_frame = reliability.analyze(gas.features(frame))
    timings["reliability"] = round(time.perf_counter() - start, 3)
    rel_summary = ReliabilityAnalyzer.summarize(rel_frame, y)
    keep = np.sort(rng.choice(len(rel_frame), min(RELIABILITY_RECORDS_KEPT, len(rel_frame)), replace=False))
    records = rel_frame.iloc[keep].assign(correct=correct[keep], row=keep)
    records["risk_label"] = records["risk_label"].astype(str)

    start = time.perf_counter()
    lf = label_free_diagnosis(reference_ds, window_ds, alpha=ALPHA, mmd_permutations=MMD_PERMUTATIONS,
                              mmd_max_samples=MMD_MAX_SAMPLES, random_state=SEED)
    n_audit = gas.audit_size(len(frame))
    audit_idx = rng.choice(len(frame), n_audit, replace=False)
    audit_X = frame.iloc[audit_idx][gas.FEATURES]
    au = audit_diagnosis(model, reference_ds, audit_X, y[audit_idx], alpha=ALPHA)
    diagnosis = combine(lf, au)
    timings["diagnosis"] = round(time.perf_counter() - start, 3)
    comparison = comparison_block(
        lf=lf, audit=au, model=model, reference_ds=reference_ds, audit_X=audit_X, audit_y=y[audit_idx],
        alpha=ALPHA, cbpe_estimate=estimated, reference_accuracy=reference_accuracy,
        production_probabilities=window_ds.prediction_probabilities, actual_accuracy=accuracy,
    )
    if comparison["verdicts"]["ours"] != diagnosis.drift_type:
        raise AssertionError("recomputed verdict differs from the diagnosis")

    print(f"  accuracy {accuracy:.4f} ({accuracy - reference_accuracy:+.4f}, p={drop['p_value']:.3g}); "
          f"CBPE {estimated:.4f}; drifted features {len(loc.drifted_features)}/{len(loc.feature_drifts)}; "
          f"diagnosis {diagnosis.drift_type} [{diagnosis.stage}] {diagnosis.rationale}")

    composition = {gas.GASES[int(k)]: int(v) for k, v in frame["gas"].value_counts().sort_index().items()}
    return {
        "analysis_type": analysis_type,
        "n_rows": int(len(frame)),
        "gas_counts": composition,
        "actual_accuracy": accuracy,
        "accuracy_change": accuracy - reference_accuracy,
        "accuracy_drop_test": drop,
        "cbpe_estimated_accuracy": estimated,
        "cbpe_estimate_error": estimated - accuracy,
        "detection": detection,
        "localization": {
            "n_features": len(loc.feature_drifts),
            "n_drifted_features": len(loc.drifted_features),
            "drifted_features": list(loc.drifted_features),
            "by_sensor": sensor_localization(loc),
        },
        "diagnosis": {
            "final": diagnosis.to_dict(),
            "label_free_verdict": lf["verdict"],
            "n_input_shift_features": len(lf["input_shift_features"]),
            "n_residual_shift_features": len(lf["residual_shift_features"]),
            "prediction_shift": lf["prediction_shift"],
            "multivariate_shift": lf["multivariate_shift"],
            "bbse": lf["bbse"],
            "audit": {k: au[k] for k in ("n_labels", "audit_accuracy", "accuracy_drop_significant",
                                         "structured_features")},
            "audit_accuracy_drop_p_value": au["accuracy_drop_test"]["p_value"],
        },
        "comparison": comparison,
        "reliability": {"summary": rel_summary, "records": records.to_dict(orient="records")},
        "timings_seconds": timings,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODEL_DESCRIPTIONS), default="lr")
    args = ap.parse_args()
    setup_logging(level="WARNING")

    data = gas.load_batches()
    reference = data[data["batch"].isin(gas.REFERENCE_BATCHES)].reset_index(drop=True)
    split = gas.split_reference(reference, seed=SEED)
    with threadpool_limits(limits=1):
        model = gas.make_gas_model(random_state=SEED, kind=args.model).fit(gas.features(split.train),
                                                                         split.train["gas"])
    reference_ds = gas.to_dataset(split.calibration, model=model)
    test_correct = (model.predict(gas.features(split.test)) == split.test["gas"].to_numpy()).astype(int)
    reference_accuracy = float(test_correct.mean())
    reliability = ReliabilityAnalyzer(model, gas.features(split.train), gas.features(split.calibration),
                                      split.calibration["gas"].to_numpy(), random_state=SEED)
    print(f"model {args.model}; split {split.sizes()}; reference (test split) accuracy {reference_accuracy:.4f}")

    common = dict(model=model, reference_ds=reference_ds, reliability=reliability,
                  reference_accuracy=reference_accuracy, reference_test_correct=test_correct)
    windows: Dict[str, Any] = {"reference held-out": {"batch": 0, "months": gas.BATCH_MONTHS[1] + ", " + gas.BATCH_MONTHS[2],
                                                      **analyse_window("reference held-out", "control",
                                                                       split.test, **common)}}
    for b in gas.ANALYSIS_BATCHES:
        frame = data[data["batch"] == b].reset_index(drop=True)
        windows[f"batch {b}"] = {"batch": b, "months": gas.BATCH_MONTHS[b],
                                 **analyse_window(f"batch {b}", "temporal", frame, **common)}

    result = {
        "config": {
            "dataset": "UCI Gas Sensor Array Drift (id 224)", "sha256": gas.SHA256,
            "model_kind": args.model, "model": MODEL_DESCRIPTIONS[args.model],
            "seed": SEED, "alpha": ALPHA, "psi_threshold": PSI_THRESHOLD,
            "mmd": {"n_permutations": MMD_PERMUTATIONS, "max_samples": MMD_MAX_SAMPLES, "standardize": True},
            "audit_rule": f"min({gas.AUDIT_CAP}, floor(n / 2)) labels",
            "reference_batches": list(gas.REFERENCE_BATCHES),
            "split_fractions": dict(zip(("train", "calibration", "test"), gas.SPLIT_FRACTIONS)),
            "split_sizes": split.sizes(),
            "n_sensors": gas.N_SENSORS, "features_per_sensor": gas.FEATURES_PER_SENSOR,
            "gases": gas.GASES, "batch_months": gas.BATCH_MONTHS,
        },
        "reference": {
            "test_accuracy": reference_accuracy,
            "gas_counts": {gas.GASES[int(k)]: int(v) for k, v in reference["gas"].value_counts().sort_index().items()},
        },
        "reliability_model": reliability.describe(),
        "windows": windows,
    }
    out = Path(f"outputs/gas_drift_results_{args.model}.json")
    out.write_text(json.dumps(result, indent=1, allow_nan=False, default=json_default))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
