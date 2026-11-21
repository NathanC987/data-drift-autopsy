"""Unit tests for the embedding drift detectors (permutation tests)."""

from __future__ import annotations

import math

import numpy as np

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.detectors import FrechetDistance, PCAReconstructionError
from drift_autopsy.registry import DetectorRegistry


def _datasets(seed: int, shift: float):
    rng = np.random.RandomState(seed)
    ref = rng.normal(0.0, 1.0, size=(240, 16))
    test = rng.normal(shift, 1.0 + shift / 3, size=(240, 16))
    return Dataset.from_numpy(ref), Dataset.from_numpy(test)


def test_embedding_detectors_are_registered():
    names = DetectorRegistry.list()
    assert {"pca_reconstruction", "fid_distance", "frechet_distance"} <= set(names)


def test_pca_reconstruction_detects_shift_and_not_null():
    ref, test = _datasets(1, 0.6)
    shifted = PCAReconstructionError(explained_variance_ratio=0.9, n_permutations=60).fit_detect(ref, test)
    assert math.isfinite(shifted.score) and shifted.drift_detected
    ref, test = _datasets(2, 0.0)
    null = PCAReconstructionError(explained_variance_ratio=0.9, n_permutations=60).fit_detect(ref, test)
    assert not null.drift_detected


def test_frechet_distance_detects_shift_and_not_null():
    ref, test = _datasets(3, 0.4)
    shifted = FrechetDistance(n_permutations=60).fit_detect(ref, test)
    assert shifted.score > 0 and shifted.drift_detected
    ref, test = _datasets(4, 0.0)
    assert not FrechetDistance(n_permutations=60).fit_detect(ref, test).drift_detected
