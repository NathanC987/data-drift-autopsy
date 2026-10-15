"""Label-free accuracy estimation (Confidence Based Performance Estimation).

CBPE (Kivimaki et al., 2025; NannyML) estimates accuracy on unlabelled
production data as the mean calibrated probability that each prediction is
correct. Here the calibration map from top-class confidence to correctness is
fitted by histogram binning (Zadrozny and Elkan, 2001) on a labelled
calibration set the model was not trained on:

    acc_hat = mean_i  a[b(c_i)],   a[k] = accuracy of calibration rows in bin k,

where c_i is the production confidence and b(.) its bin. A bin with fewer
than MIN_BIN_COUNT calibration rows (the package's minimum cell count, as in
the chi-square tests) holds no reliable accuracy estimate; a prediction that
falls in such a bin keeps its own confidence, i.e. the model is assumed
calibrated where the calibration data are silent. The estimate
is unbiased only while P(correct | confidence) is unchanged between the
calibration data and production, so it cannot see concept drift or label
noise, and it goes stale as the calibration data ages.

Baselines (all fitted on the same calibration set, all label free):

* AC  - average confidence: the mean top-class confidence (Hendrycks and
  Gimpel, 2017); unbiased only for a perfectly calibrated model.
* DoC - difference of confidences (Guillory et al., 2021): calibration accuracy
  plus the change in mean confidence from calibration to production.
* ATC - average thresholded confidence (Garg et al., 2022) with the negative
  entropy score: the threshold t is chosen so that the share of calibration
  rows scoring above t equals the calibration accuracy; the estimate is the
  share of production rows above t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from drift_autopsy.stats import MIN_EXPECTED_COUNT

MIN_BIN_COUNT = MIN_EXPECTED_COUNT


def top_class_confidence(probabilities: np.ndarray) -> np.ndarray:
    """Top-class confidence from (n,) positive-class or (n, k) class probabilities."""
    proba = np.asarray(probabilities, dtype=float)
    if proba.ndim == 1:
        return np.maximum(proba, 1.0 - proba)
    return proba.max(axis=1)


@dataclass
class BinnedAccuracyCalibration:
    """Histogram-binning map from confidence to probability of being correct."""

    bin_edges: np.ndarray
    bin_accuracy: np.ndarray
    bin_counts: np.ndarray
    overall_accuracy: float

    @classmethod
    def fit(cls, confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> "BinnedAccuracyCalibration":
        confidence = np.asarray(confidence, dtype=float)
        correct = np.asarray(correct, dtype=float)
        if confidence.shape != correct.shape or confidence.size == 0:
            raise ValueError("confidence and correct must be non-empty and the same shape")
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bins = _bin_index(confidence, edges)
        counts = np.bincount(bins, minlength=n_bins).astype(float)
        sums = np.bincount(bins, weights=correct, minlength=n_bins)
        overall = float(correct.mean())
        accuracy = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
        return cls(bin_edges=edges, bin_accuracy=accuracy, bin_counts=counts, overall_accuracy=overall)

    @property
    def reliable(self) -> np.ndarray:
        """Bins with at least MIN_BIN_COUNT calibration rows."""
        return self.bin_counts >= MIN_BIN_COUNT

    def predict_correct(self, confidence: np.ndarray) -> np.ndarray:
        """Calibrated probability that each prediction is correct (its own
        confidence in a bin with too few calibration rows)."""
        confidence = np.asarray(confidence, dtype=float)
        idx = _bin_index(confidence, self.bin_edges)
        return np.where(self.reliable[idx], np.nan_to_num(self.bin_accuracy[idx]), confidence)

    def estimate_accuracy(self, confidence: np.ndarray) -> float:
        confidence = np.asarray(confidence, dtype=float)
        if confidence.size == 0:
            raise ValueError("cannot estimate accuracy of an empty sample")
        return float(self.predict_correct(confidence).mean())

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "bin_edges": self.bin_edges.tolist(),
            "bin_accuracy": [None if not np.isfinite(a) else float(a) for a in self.bin_accuracy],
            "bin_reliable": self.reliable.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "overall_accuracy": self.overall_accuracy,
        }


def _bin_index(confidence: np.ndarray, edges: np.ndarray) -> np.ndarray:
    n_bins = len(edges) - 1
    return np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, n_bins - 1)


def negative_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Negative Shannon entropy of each row of (n, k) or positive-class (n,) probabilities."""
    proba = np.asarray(probabilities, dtype=float)
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    p = np.clip(proba, 1e-12, 1.0)
    return (proba * np.log(p)).sum(axis=1)


def atc_threshold(calibration_score: np.ndarray, calibration_correct: np.ndarray) -> float:
    """Threshold t with mean(score > t) = calibration accuracy (Garg et al., 2022)."""
    score = np.asarray(calibration_score, dtype=float)
    accuracy = float(np.mean(calibration_correct))
    return float(np.quantile(score, 1.0 - accuracy))


def label_free_estimates(
    calibration_probabilities: np.ndarray,
    calibration_correct: np.ndarray,
    production_probabilities: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """CBPE and the three baselines of the module docstring."""
    cal_conf = top_class_confidence(calibration_probabilities)
    prod_conf = top_class_confidence(production_probabilities)
    correct = np.asarray(calibration_correct, dtype=float)
    t = atc_threshold(negative_entropy(calibration_probabilities), correct)
    return {
        "cbpe": cbpe_accuracy_estimate(cal_conf, correct, prod_conf, n_bins),
        "ac": float(prod_conf.mean()),
        "doc": float(correct.mean() + prod_conf.mean() - cal_conf.mean()),
        "atc": float(np.mean(negative_entropy(production_probabilities) > t)),
    }


def cbpe_accuracy_estimate(
    calibration_confidence: np.ndarray,
    calibration_correct: np.ndarray,
    production_confidence: np.ndarray,
    n_bins: int = 10,
) -> float:
    """One-call CBPE estimate; see the module docstring."""
    calibration = BinnedAccuracyCalibration.fit(calibration_confidence, calibration_correct, n_bins)
    return calibration.estimate_accuracy(production_confidence)
