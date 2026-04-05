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

    detector = CBPE(threshold=0.05, n_bins=10, min_bin_count=5)
    result = detector.fit_detect(ref, test)

    assert math.isfinite(result.score)
    assert result.metadata is not None
    assert result.metadata.get("smoothing_epsilon") is not None
