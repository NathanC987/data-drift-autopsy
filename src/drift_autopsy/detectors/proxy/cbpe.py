"""Confidence-based performance monitoring: a confidence-shift test plus the CBPE estimate."""

import logging
from typing import Optional

import numpy as np

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.result import DetectionResult, DriftSeverity
from drift_autopsy.estimation import BinnedAccuracyCalibration, top_class_confidence
from drift_autopsy.registry import DetectorRegistry
from drift_autopsy.stats import chi2_homogeneity_counts, severity_from_p_value

logger = logging.getLogger(__name__)


@DetectorRegistry.register("cbpe")
class CBPE(BaseDriftDetector):
    """
    Confidence-based performance monitoring.

    Two outputs, both label free on the production side:

    1. A confidence-shift test: the top-class confidences of the reference and
       production predictions are binned on fixed edges and compared with a
       two-sample chi-square test of homogeneity. This is the detector's
       decision (``drift_detected``, ``p_value``, ``score`` = chi-square).
    2. The CBPE accuracy estimate (see :mod:`drift_autopsy.estimation`), reported
       as ``metadata["estimated_accuracy"]`` when the reference dataset carries
       targets and predictions. The reference must then be a labelled
       calibration set the model was not trained on; otherwise the estimate
       inherits the model's training optimism.

    Args:
        threshold: Significance level of the confidence-shift test (default: 0.05)
        n_bins: Number of equal-width confidence bins on [0, 1] (default: 10)

    Note:
        Both datasets must have ``prediction_probabilities``. A 1-D array is read
        as the positive-class probability of a binary model.
    """

    def __init__(self, threshold: float = 0.05, n_bins: int = 10):
        super().__init__(name="cbpe")
        self.threshold = threshold
        self.n_bins = n_bins
        self._bin_edges: Optional[np.ndarray] = None
        self._reference_bins: Optional[np.ndarray] = None
        self._ref_mean_confidence: Optional[float] = None
        self._calibration: Optional[BinnedAccuracyCalibration] = None

    def _bin_counts(self, confidence: np.ndarray) -> np.ndarray:
        idx = np.clip(np.digitize(confidence, self._bin_edges[1:-1], right=False), 0, self.n_bins - 1)
        return np.bincount(idx, minlength=self.n_bins).astype(float)

    def fit(self, reference_data: Dataset) -> None:
        """
        Learn the reference confidence histogram (and the accuracy calibration map
        when the reference has targets and predictions).

        Raises:
            ValueError: If prediction probabilities are not available
        """
        super().fit(reference_data)
        if reference_data.prediction_probabilities is None:
            raise ValueError(
                "CBPE requires prediction_probabilities in the dataset. "
                "Please run your model and add predictions to the Dataset."
            )

        confidence = top_class_confidence(reference_data.prediction_probabilities)
        self._ref_mean_confidence = float(np.mean(confidence))
        self._bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        self._reference_bins = self._bin_counts(confidence)

        self._calibration = None
        if reference_data.target is not None and reference_data.predictions is not None:
            correct = (np.asarray(reference_data.predictions) == np.asarray(reference_data.target)).astype(float)
            self._calibration = BinnedAccuracyCalibration.fit(confidence, correct, self.n_bins)

        logger.info(
            f"CBPE fitted on {reference_data.n_samples} samples, "
            f"confidence range: [{confidence.min():.3f}, {confidence.max():.3f}], "
            f"accuracy calibration: {'yes' if self._calibration is not None else 'no'}"
        )

    def detect(self, test_data: Dataset) -> DetectionResult:
        """
        Test for a confidence-distribution shift and estimate production accuracy.

        Raises:
            ValueError: If prediction probabilities are not available
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        if test_data.prediction_probabilities is None:
            raise ValueError("CBPE requires prediction_probabilities in the test dataset.")

        logger.info(f"Running CBPE on {test_data.n_samples} test samples")
        confidence = top_class_confidence(test_data.prediction_probabilities)
        test_bins = self._bin_counts(confidence)

        test = chi2_homogeneity_counts(self._reference_bins, test_bins)
        chi2_stat, p_value = float(test["statistic"]), float(test["p_value"])
        drift_detected = test["dof"] > 0 and p_value < self.threshold
        severity = severity_from_p_value(p_value, self.threshold) if drift_detected else DriftSeverity.NONE

        test_mean_conf = float(np.mean(confidence))
        confidence_shift_signed = test_mean_conf - self._ref_mean_confidence

        metadata = {
            "chi2_statistic": chi2_stat,
            "dof": int(test["dof"]),
            "n_bins": self.n_bins,
            "n_tested_cells": int(test["n_cells"]),
            "ref_mean_confidence": float(self._ref_mean_confidence),
            "test_mean_confidence": test_mean_conf,
            "confidence_shift": float(abs(confidence_shift_signed)),
            "confidence_shift_signed": float(confidence_shift_signed),
            "reference_bin_counts": self._reference_bins.astype(int).tolist(),
            "test_bin_counts": test_bins.astype(int).tolist(),
        }
        if self._calibration is not None:
            metadata["estimated_accuracy"] = self._calibration.estimate_accuracy(confidence)
            metadata["reference_accuracy"] = self._calibration.overall_accuracy
            metadata["estimated_accuracy_change"] = (
                metadata["estimated_accuracy"] - self._calibration.overall_accuracy
            )

        logger.info(
            f"CBPE result: drift={drift_detected}, p-value={p_value:.4g}, chi2={chi2_stat:.2f}, "
            f"confidence_shift={confidence_shift_signed:+.4f}, severity={severity.value}"
        )

        return DetectionResult(
            detector_name=self.name,
            drift_detected=bool(drift_detected),
            severity=severity,
            score=chi2_stat,
            threshold=self.threshold,
            p_value=p_value,
            statistic=chi2_stat,
            metadata=metadata,
        )
