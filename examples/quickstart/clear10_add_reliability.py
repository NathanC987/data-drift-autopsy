"""Attach per-prediction reliability records to outputs/clear10_drift_results.json.

Reuses the cached tabularised buckets (outputs/clear10_tabularized_demo/), so it
does not re-run the ResNet extractor. Runs the model-agnostic
``ReliabilityAnalyzer`` against a logistic-regression head on the frozen
embeddings (the same lightweight model the remediation study monitors).
Explanation-consistency (SHAP) is disabled - it does not scale to 512-d
embeddings; the reliability layer renormalises its weights over the remaining
signals. Idempotent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from drift_autopsy.reliability import (
    ConfidenceExtractor,
    OODDetector,
    RiskScoringEngine,
    StabilityChecker,
    ReliabilityAnalyzer,
)
from drift_autopsy.reliability.calibration import CalibrationChecker

TAB_DIR = Path("outputs/clear10_tabularized_demo")
REPORT = Path("outputs/clear10_drift_results.json")
REFERENCE_BUCKET = 1
MAX_SAMPLES = 80


class _NoExplanation:
    """Stand-in for the explanation-consistency checker (SHAP is too slow at 512-d)."""

    def compute(self, *_, **__):
        return {"explanation_score": None,
                "metadata": {"available": False, "reason": "disabled for embeddings", "method": "disabled"}}


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feature_")]


def main() -> None:
    report = json.loads(REPORT.read_text())
    ref = pd.read_parquet(TAB_DIR / f"bucket_{REFERENCE_BUCKET}.parquet")
    cols = feature_cols(ref)
    X_ref = ref[cols].to_numpy(float)
    y_ref = ref["y_true"].to_numpy(int)

    model = LogisticRegression(max_iter=1000, random_state=42).fit(X_ref, y_ref)

    n_attached = 0
    for key in sorted(report.get("bucket_results", {}), key=lambda k: int(k)):
        path = TAB_DIR / f"bucket_{key}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        sample = frame.sample(n=min(MAX_SAMPLES, len(frame)), random_state=42).reset_index(drop=True)
        X = sample[cols].to_numpy(float)

        acc_est = None
        if {"y_true", "y_pred"}.issubset(sample.columns):
            valid = sample[["y_true", "y_pred"]].dropna()
            if len(valid):
                acc_est = float((valid["y_true"].to_numpy(int) == valid["y_pred"].to_numpy(int)).mean())

        analyzer = ReliabilityAnalyzer(
            model=model,
            data_type="tabular",
            reference_data=X_ref,
            task_type="classification",
            cbpe_reference_score=acc_est,
            confidence_extractor=ConfidenceExtractor(model=model, task_type="classification"),
            ood_detector=OODDetector(data_type="tabular"),
            stability_checker=StabilityChecker(model=model, data_type="tabular"),
            calibration_checker=CalibrationChecker(),
            explanation_checker=_NoExplanation(),
            risk_engine=RiskScoringEngine(),
        )
        records = analyzer.analyze_batch(
            input_batch=X,
            cbpe_score=acc_est,
            prediction_ids=[f"clear10_{key}_{i}" for i in range(len(X))],
            shared_explanation_for_batch=False,
        )
        for r in records:
            r["analysis_key"] = str(key)
            r["detector"] = "embedding_lr"

        report["bucket_results"][key]["reliability"] = records
        n_attached += len(records)
        conf = np.mean([r["confidence"] for r in records if r["confidence"] is not None])
        cal = np.mean([r["calibration_risk"] for r in records if r["calibration_risk"] is not None])
        print(f"  bucket {key}: {len(records)} records, mean confidence {conf:.3f}, "
              f"mean calibration risk {cal:.3f}, acc_est {acc_est}")

    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"attached {n_attached} reliability records to {REPORT}")


if __name__ == "__main__":
    main()
