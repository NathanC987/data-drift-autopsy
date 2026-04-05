"""Unit tests for Phase 2 embedding drift detectors."""

from __future__ import annotations

import math

import numpy as np

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.detectors import FIDDistance, PCAReconstructionError
from drift_autopsy.registry import DetectorRegistry


def _datasets(seed: int = 42):
    rng = np.random.RandomState(seed)
    ref = rng.normal(0.0, 1.0, size=(240, 64))
    test = rng.normal(0.35, 1.1, size=(240, 64))
    return Dataset.from_numpy(ref), Dataset.from_numpy(test)


def test_embedding_detectors_are_registered():
    names = DetectorRegistry.list()
    assert "pca_reconstruction" in names
    assert "fid_distance" in names


def test_pca_reconstruction_detector_finite_and_detects_shift():
    ref, test = _datasets()
    detector = PCAReconstructionError(threshold=0.01, explained_variance_ratio=0.9)
    result = detector.fit_detect(ref, test)

    assert math.isfinite(result.score)
    assert result.score >= 0.0
    assert result.drift_detected is True


def test_fid_distance_detector_finite_and_detects_shift():
    ref, test = _datasets(seed=7)
    detector = FIDDistance(threshold=1.0, covariance_eps=1e-6)
    result = detector.fit_detect(ref, test)

    assert math.isfinite(result.score)
    assert result.score >= 0.0
    assert result.drift_detected is True
