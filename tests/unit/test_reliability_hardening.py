"""Unit tests for reliability hardening behavior."""

from __future__ import annotations

import numpy as np

from drift_autopsy.reliability import ReliabilityAnalyzer, RiskScoringEngine


class DummyModel:
    """Simple deterministic model stub with predict/predict_proba."""

    def predict_proba(self, x):
        arr = np.asarray(x)
        n = arr.shape[0] if arr.ndim > 1 else 1
        return np.tile(np.array([[0.8, 0.2]], dtype=float), (n, 1))

    def predict(self, x):
        arr = np.asarray(x)
        n = arr.shape[0] if arr.ndim > 1 else 1
        return np.zeros(n, dtype=int)


def _flatten_embedding(x):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr.reshape(1, -1)
    return arr.reshape(arr.shape[0], -1)


def test_risk_engine_renormalizes_available_signals_only():
    engine = RiskScoringEngine()

    out = engine.combine(
        confidence_score=0.9,
        ood_score=0.4,
        stability_score=0.2,
        calibration_flag="good",
        calibration_risk=None,
        explanation_score=None,
    )

    assert out["risk_score"] is not None
    assert out["risk_label"] in {"LOW", "MEDIUM", "HIGH"}
    used = out["weighting"]["used_signals"]
    assert "calibration_risk" not in used
    assert "explanation_score" not in used
    renorm = out["weighting"]["renormalized_weights"]
    assert abs(sum(renorm.values()) - 1.0) < 1e-8


def test_image_mode_marks_missing_explanation_as_required_gap():
    model = DummyModel()
    reference_images = [np.zeros((2, 2), dtype=float), np.ones((2, 2), dtype=float)]

    analyzer = ReliabilityAnalyzer(
        model=model,
        data_type="image",
        reference_data=reference_images,
        task_type="classification",
        embedding_extractor=_flatten_embedding,
        gradcam_extractor=None,
    )

    result = analyzer.analyze(np.zeros((2, 2), dtype=float))

    assert result["explanation"] is None
    assert "quality" not in result["details"]
    status = result["details"]["signal_status"]["explanation"]
    assert status["available"] is False


def test_text_mode_requires_calibration_and_flags_missing_cbpe():
    model = DummyModel()
    reference_text = ["hello world", "this is a reference sample"]

    analyzer = ReliabilityAnalyzer(
        model=model,
        data_type="text",
        reference_data=reference_text,
        task_type="classification",
        cbpe_reference_score=None,
    )

    result = analyzer.analyze("new prediction sample")

    assert result["calibration_risk"] is None
    assert "quality" not in result["details"]
    assert result["details"]["signal_status"]["calibration"]["available"] is False
