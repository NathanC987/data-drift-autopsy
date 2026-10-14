"""Tests for the per-prediction reliability layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from drift_autopsy.reliability import OODDetector, ReliabilityAnalyzer, RiskFusion, StabilityChecker


def _data(n, rng, shift=0.0):
    x_num = rng.normal(loc=shift, size=(n, 2))
    x_cat = rng.choice([1.0, 2.0, 3.0], size=(n, 1), p=[0.6, 0.3, 0.1])
    X = np.hstack([x_num, x_cat])
    logits = 1.5 * x_num[:, 0] - x_num[:, 1] + 0.5 * (x_cat[:, 0] == 3.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y


def test_ood_score_is_calibrated_on_reference_and_flags_extremes_and_unseen():
    rng = np.random.default_rng(0)
    X, _ = _data(4000, rng)
    det = OODDetector(categorical_indices=[2]).fit(X)
    ref_scores = det.score(X)
    assert abs(np.mean(ref_scores > 0) - 0.10) < 0.02      # 90th percentile is the zero point
    extreme = X[:50].copy(); extreme[:, 0] = 12.0
    assert np.all(det.score(extreme) == 1.0)
    unseen = X[:50].copy(); unseen[:, 2] = 99.0
    assert det.score(unseen).mean() > ref_scores.mean()


def test_stability_uses_reference_scale_and_never_perturbs_categoricals():
    rng = np.random.default_rng(1)
    X, y = _data(3000, rng)
    model = LogisticRegression().fit(X, y)
    chk = StabilityChecker(model, perturb_indices=[0, 1], strength=0.1).fit(X)
    s = chk.score(X[:200])
    assert np.all(s > 0)                                   # the old row-SD bug gave ~1e-10
    assert chk._scale[2] == 0.0


def test_fusion_learns_weights_and_labels_by_meaningful_bands():
    rng = np.random.default_rng(2)
    n = 4000
    informative = rng.random(n)
    noise = rng.random(n)
    error = (rng.random(n) < 0.4 * informative).astype(int)   # base error rate ~0.2
    frame = pd.DataFrame({"a": informative, "b": noise})
    fusion = RiskFusion(["a", "b"]).fit(frame, error)
    coef = fusion.coefficients()
    assert coef["a"] > 0.5 and coef["a"] > 5 * abs(coef["b"])
    labels = fusion.label(np.array([0.0, fusion.reference_error_rate, 0.5, 0.9]))
    assert labels.tolist() == ["LOW", "MEDIUM", "HIGH", "HIGH"]


def test_analyzer_end_to_end_ranks_errors_above_chance():
    rng = np.random.default_rng(3)
    X_tr, y_tr = _data(6000, rng)
    X_cal, y_cal = _data(3000, rng)
    X_prod, y_prod = _data(3000, rng, shift=0.5)
    model = LogisticRegression().fit(X_tr, y_tr)
    ra = ReliabilityAnalyzer(model, X_tr, X_cal, y_cal, categorical_indices=[2])
    frame = ra.analyze(X_prod)
    assert set(frame["risk_label"]) <= {"LOW", "MEDIUM", "HIGH"}
    summary = ReliabilityAnalyzer.summarize(frame, y_prod)
    assert summary["auroc_risk"] > 0.6
    assert summary["error_rate_high"] is None or summary["error_rate_high"] > summary["error_rate_low"]
    assert ra.describe()["fusion"]["reference_error_rate"] > 0


def test_analyzer_without_calibration_is_marked_uncalibrated():
    rng = np.random.default_rng(4)
    X, y = _data(1000, rng)
    model = LogisticRegression().fit(X, y)
    frame = ReliabilityAnalyzer(model, X, categorical_indices=[2]).analyze(X[:20])
    assert (frame["risk_label"] == "UNCALIBRATED").all()
