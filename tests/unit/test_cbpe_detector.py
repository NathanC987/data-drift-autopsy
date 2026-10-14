"""Unit tests for CBPE detector numeric stability."""

from __future__ import annotations

import math

import numpy as np

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.detectors import CBPE


def test_cbpe_produces_finite_score_with_zero_reference_bins():
    # Reference is highly confident in one class, so many bins will have zero count.
    ref_proba = np.column_stack([np.full(200, 0.99), np.full(200, 0.01)])

    # Test has a shifted confidence profile.
    test_proba = np.column_stack([np.full(200, 0.55), np.full(200, 0.45)])

    ref = Dataset.from_numpy(np.random.randn(200, 3), prediction_probabilities=ref_proba)
    test = Dataset.from_numpy(np.random.randn(200, 3), prediction_probabilities=test_proba)

    detector = CBPE(threshold=0.05, n_bins=10)
    result = detector.fit_detect(ref, test)

    assert math.isfinite(result.score)
    assert result.drift_detected
    assert result.metadata["dof"] >= 1


def test_cbpe_reference_mean_uses_actual_confidences():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1.0, 500)
    proba = np.column_stack([p, 1 - p])

    ref = Dataset.from_numpy(rng.normal(size=(500, 3)), prediction_probabilities=proba)
    test = Dataset.from_numpy(rng.normal(size=(500, 3)), prediction_probabilities=proba.copy())

    result = CBPE(threshold=0.05, n_bins=10).fit_detect(ref, test)

    assert math.isclose(result.metadata["ref_mean_confidence"], float(p.mean()), rel_tol=1e-9)
    assert math.isclose(result.metadata["test_mean_confidence"], float(p.mean()), rel_tol=1e-9)
    assert abs(result.metadata["confidence_shift"]) < 1e-12
    assert abs(result.metadata["confidence_shift_signed"]) < 1e-12


def test_cbpe_estimates_accuracy_from_labelled_reference():
    rng = np.random.default_rng(3)
    n = 4000
    p = rng.uniform(0.5, 1.0, n)
    correct = rng.random(n) < p          # perfectly calibrated reference
    pred = np.ones(n, dtype=int)
    target = np.where(correct, 1, 0)
    ref = Dataset.from_numpy(rng.normal(size=(n, 2)), y=target, predictions=pred,
                             prediction_probabilities=np.column_stack([1 - p, p]))
    q = rng.uniform(0.9, 1.0, n)          # production: more confident rows
    test = Dataset.from_numpy(rng.normal(size=(n, 2)), predictions=pred,
                              prediction_probabilities=np.column_stack([1 - q, q]))

    result = CBPE(threshold=0.05, n_bins=10).fit_detect(ref, test)

    assert abs(result.metadata["estimated_accuracy"] - q.mean()) < 0.02
    assert abs(result.metadata["reference_accuracy"] - correct.mean()) < 1e-12


def test_cbpe_binary_1d_probability_is_read_as_positive_class():
    p = np.linspace(0.05, 0.95, 200)
    ref = Dataset.from_numpy(np.zeros((200, 1)), prediction_probabilities=p)
    test = Dataset.from_numpy(np.zeros((200, 1)), prediction_probabilities=1 - p)
    result = CBPE().fit_detect(ref, test)
    # max(p, 1-p) is symmetric, so flipping the class leaves confidence unchanged
    assert abs(result.metadata["confidence_shift_signed"]) < 1e-12
    assert not result.drift_detected
