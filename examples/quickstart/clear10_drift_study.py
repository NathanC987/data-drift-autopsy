"""
CLEAR-10 drift study on frozen ResNet-18 embeddings.

Setup (drift_autopsy.experiments.clear10): bucket 1 is split 60 / 20 / 20 into
train / calibration / test; the monitored linear head is trained on the train
split. The calibration split is the reference window for every detector and
fits the CBPE map. Windows: the bucket 1 test split (no-drift control) and
buckets 2 to 10, each compared with the same fixed reference.

Per window: every detector (decided by a test at ALPHA), per-dimension
localisation (Holm), slices by predicted class, CBPE proxy metrics with their
error against the true metrics, per-prediction reliability, and the label-free
diagnosis plus a delayed-label audit of AUDIT_LABELS images.
Output: outputs/clear10_drift_results.json.

Requires the embeddings from examples/quickstart/clear10_tabularize_demo.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from drift_autopsy import DriftPipeline
from drift_autopsy.data import MulticlassProxyEstimator
from drift_autopsy.detectors import CBPE, KSTest, MMD, PSI, FrechetDistance, PCAReconstructionError
from drift_autopsy.diagnosis import audit_diagnosis, combine, label_free_diagnosis
from drift_autopsy.experiments.clear10 import (
    ANALYSIS_BUCKETS,
    EMBEDDING_COLUMNS,
    REFERENCE_BUCKET,
    SPLIT_FRACTIONS,
    class_names,
    embeddings,
    load_bucket,
    make_clear_model,
    split_reference,
    to_dataset,
)
from drift_autopsy.localizers import UnivariateLocalizer
from drift_autopsy.reliability import ReliabilityAnalyzer
from drift_autopsy.stats import two_proportion_test
from drift_autopsy.utils import setup_logging

SEED = 42
ALPHA = 0.05
PSI_THRESHOLD = 0.2
N_PERMUTATIONS = 200
AUDIT_LABELS = 600
MIN_SLICE_SAMPLES = 30
RELIABILITY_RECORDS_KEPT = 1000
OUTPUT = Path("outputs/clear10_drift_results.json")


def detectors() -> Dict[str, Any]:
    return {
        "KS Test": KSTest(threshold=ALPHA, correction="bonferroni"),
        "PSI": PSI(threshold=PSI_THRESHOLD, n_bins=10),
        "MMD": MMD(threshold=ALPHA, n_permutations=N_PERMUTATIONS, standardize=True, random_state=SEED),
        "PCA reconstruction": PCAReconstructionError(threshold=ALPHA, n_permutations=N_PERMUTATIONS,
                                                     random_state=SEED),
        "Frechet distance": FrechetDistance(threshold=ALPHA, n_permutations=N_PERMUTATIONS, random_state=SEED),
        "CBPE": CBPE(threshold=ALPHA, n_bins=10),
    }


def proxy_frame(frame: pd.DataFrame, model) -> pd.DataFrame:
    proba = model.predict_proba(embeddings(frame))
    out = pd.DataFrame(proba, columns=[f"pred_proba_{k}" for k in range(proba.shape[1])])
    out["y_pred"] = model.classes_[proba.argmax(axis=1)]
    out["y_true"] = frame["y_true"].to_numpy()
    return out


def predicted_class_slices(reference_ds, window_ds, names: Dict[int, str]) -> Dict[str, Any]:
    """Per predicted class: how many embedding dimensions drift (Holm within the class)."""
    out = {}
    ref_pred = np.asarray(reference_ds.metadata["predicted_class"])
    win_pred = np.asarray(window_ds.metadata["predicted_class"])
    localizer = UnivariateLocalizer(threshold=ALPHA, correction="holm")
    for cls in sorted(set(ref_pred.tolist()) & set(win_pred.tolist())):
        r_mask, w_mask = ref_pred == cls, win_pred == cls
        if r_mask.sum() < MIN_SLICE_SAMPLES or w_mask.sum() < MIN_SLICE_SAMPLES:
            continue
        r = DriftPipeline._slice_dataset(reference_ds, r_mask)
        w = DriftPipeline._slice_dataset(window_ds, w_mask)
        loc = localizer.localize(r, w)
        out[names.get(int(cls), str(cls))] = {
            "n_reference": int(r_mask.sum()), "n_window": int(w_mask.sum()),
            "n_drifted_dimensions": len(loc.drifted_features),
        }
    return out


def analyse_window(key: str, analysis_type: str, frame: pd.DataFrame, *, model, reference_ds, reference_frame,
                   proxy, reliability, names, reference_test_correct, reference_accuracy, rng) -> Dict[str, Any]:
    print(f"\n=== {key} ({analysis_type}, {len(frame)} images)")
    window_ds = to_dataset(frame, model=model, with_target=False)
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
    top = sorted(loc.feature_drifts, key=lambda f: f.p_value)[:20]

    y = frame["y_true"].to_numpy()
    correct = (window_ds.predictions == y).astype(int)
    accuracy = float(correct.mean())
    drop = two_proportion_test(int(reference_test_correct.sum()), len(reference_test_correct),
                               int(correct.sum()), len(correct), alternative="greater")
    pm = proxy.estimate(proxy_frame(frame, model))

    start = time.perf_counter()
    rel_frame = reliability.analyze(embeddings(frame))
    timings["reliability"] = round(time.perf_counter() - start, 3)
    rel_summary = ReliabilityAnalyzer.summarize(rel_frame, y)
    keep = np.sort(rng.choice(len(rel_frame), min(RELIABILITY_RECORDS_KEPT, len(rel_frame)), replace=False))
    records = rel_frame.iloc[keep].assign(correct=correct[keep], row=keep)
    records["risk_label"] = records["risk_label"].astype(str)

    start = time.perf_counter()
    lf = label_free_diagnosis(reference_ds, window_ds, alpha=ALPHA, mmd_permutations=N_PERMUTATIONS,
                              random_state=SEED)
    audit_idx = rng.choice(len(frame), min(AUDIT_LABELS, len(frame)), replace=False)
    au = audit_diagnosis(model, reference_ds, frame.iloc[audit_idx][EMBEDDING_COLUMNS], y[audit_idx], alpha=ALPHA)
    diagnosis = combine(lf, au)
    timings["diagnosis"] = round(time.perf_counter() - start, 3)

    print(f"  accuracy {accuracy:.4f} ({accuracy - reference_accuracy:+.4f}, p={drop['p_value']:.3g}); "
          f"CBPE {pm.estimated['accuracy']:.4f}; drifted dims {len(loc.drifted_features)}; "
          f"diagnosis {diagnosis.drift_type} [{diagnosis.stage}]; risk AUROC "
          f"{rel_summary.get('auroc_risk', float('nan')):.3f} vs conf {rel_summary.get('auroc_confidence_only', float('nan')):.3f}")

    return {
        "analysis_type": analysis_type,
        "n_images": int(len(frame)),
        "actual_accuracy": accuracy,
        "accuracy_change": accuracy - reference_accuracy,
        "accuracy_drop_test": drop,
        "proxy": {"estimated": pm.estimated, "actual": pm.actual, "error": {
            k: pm.estimated[k] - pm.actual[k] for k in pm.estimated}},
        "proxy_classwise": {"estimated": pm.class_wise_estimated, "actual": pm.class_wise_actual},
        "detection": detection,
        "localization": {
            "n_dimensions": len(loc.feature_drifts),
            "n_drifted_dimensions": len(loc.drifted_features),
            "top_dimensions": [{"feature": f.feature_name, "ks": f.score, "p_value": f.p_value} for f in top],
        },
        "predicted_class_slices": predicted_class_slices(reference_ds, window_ds, names),
        "diagnosis": {
            "final": diagnosis.to_dict(),
            "label_free_verdict": lf["verdict"],
            "multivariate_shift": lf["multivariate_shift"],
            "n_input_shift_dimensions": len(lf["input_shift_features"]),
            "prediction_shift": lf["prediction_shift"],
            "bbse_condition_number": lf["bbse"]["condition_number"],
            "audit_accuracy": au["audit_accuracy"],
            "audit_accuracy_drop_p_value": au["accuracy_drop_test"]["p_value"],
            "audit_structured_dimensions": au["structured_features"],
        },
        "reliability": {"summary": rel_summary, "records": records.to_dict(orient="records")},
        "timings_seconds": timings,
    }


def main() -> None:
    setup_logging(level="WARNING")
    rng = np.random.default_rng(SEED)
    bucket1 = load_bucket(REFERENCE_BUCKET)
    names = class_names(bucket1)
    split = split_reference(bucket1, seed=SEED)
    with threadpool_limits(limits=1):
        model = make_clear_model(random_state=SEED).fit(embeddings(split.train), split.train["y_true"])

    reference_ds = to_dataset(split.calibration, model=model)
    test_correct = (model.predict(embeddings(split.test)) == split.test["y_true"].to_numpy()).astype(int)
    reference_accuracy = float(test_correct.mean())
    proxy = MulticlassProxyEstimator(n_bins=10)
    proxy.fit(proxy_frame(split.calibration, model))
    reliability = ReliabilityAnalyzer(model, embeddings(split.train), embeddings(split.calibration),
                                      split.calibration["y_true"].to_numpy(), random_state=SEED)
    print(f"split {split.sizes()}; reference (bucket {REFERENCE_BUCKET} test) accuracy {reference_accuracy:.4f}")

    common = dict(model=model, reference_ds=reference_ds, reference_frame=split.calibration, proxy=proxy,
                  reliability=reliability, names=names, reference_test_correct=test_correct,
                  reference_accuracy=reference_accuracy, rng=rng)
    windows: Dict[str, Any] = {"1 held-out": {"bucket": 1, **analyse_window("bucket 1 held-out", "control",
                                                                            split.test, **common)}}
    for b in ANALYSIS_BUCKETS:
        windows[str(b)] = {"bucket": b, **analyse_window(f"bucket {b}", "temporal", load_bucket(b), **common)}

    result = {
        "config": {
            "seed": SEED, "alpha": ALPHA, "psi_threshold": PSI_THRESHOLD, "n_permutations": N_PERMUTATIONS,
            "audit_labels": AUDIT_LABELS, "reference_bucket": REFERENCE_BUCKET,
            "split_fractions": dict(zip(("train", "calibration", "test"), SPLIT_FRACTIONS)),
            "split_sizes": split.sizes(),
            "representation": "frozen ImageNet ResNet-18 global average pool (512-d)",
            "model": "standardised multinomial logistic regression on the embeddings",
        },
        "class_names": names,
        "reference": {"test_accuracy": reference_accuracy},
        "reliability_model": reliability.describe(),
        "windows": windows,
    }
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
