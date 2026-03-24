"""Confidence-Based Performance Estimation (CBPE) for proxy performance monitoring."""

import numpy as np
from scipy import stats
from typing import Optional
import logging

from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import DetectionResult, DriftSeverity
from drift_autopsy.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("cbpe")
class CBPE(BaseDriftDetector):
    """
    Confidence-Based Performance Estimation.
    
    CBPE estimates model performance degradation without ground truth labels
    by monitoring shifts in the model's confidence (prediction probability)
    distribution. Works for classification tasks.
    
    The key insight: if the model's confidence distribution shifts, its
    performance has likely changed. CBPE bins the confidence values and
    compares the distribution using chi-square test.
    
    Args:
        threshold: P-value threshold for detecting performance drift (default: 0.05)
        n_bins: Number of bins for confidence distribution (default: 10)
        min_bin_count: Minimum count per bin to be included in test (default: 5)
    
    Note:
        The dataset must have prediction_probabilities populated.
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        n_bins: int = 10,
        min_bin_count: int = 5,
        smoothing_epsilon: float = 1e-6,
    ):
        super().__init__(name="cbpe")
        self.threshold = threshold
        self.n_bins = n_bins
        self.min_bin_count = min_bin_count
        self.smoothing_epsilon = smoothing_epsilon
        self._reference_bins = None
        self._bin_edges = None
    
    def fit(self, reference_data: Dataset) -> None:
        """
        Learn confidence distribution from reference data.
        
        Args:
            reference_data: Reference dataset with predictions
        
        Raises:
            ValueError: If prediction probabilities not available
        """
        super().fit(reference_data)
        
        if reference_data.prediction_probabilities is None:
            raise ValueError(
                "CBPE requires prediction_probabilities in the dataset. "
                "Please run your model and add predictions to the Dataset."
            )
        
        # Get confidence (max probability for multi-class, or positive class prob for binary)
        pred_proba = reference_data.prediction_probabilities
        
        if len(pred_proba.shape) == 1:
            # Binary classification - single probability
            confidence = pred_proba
        else:
            # Multi-class - use max probability
            confidence = np.max(pred_proba, axis=1)
        
        # Create bins
        self._bin_edges = np.linspace(0, 1, self.n_bins + 1)
        
        # Bin the reference confidence
        ref_binned = np.digitize(confidence, self._bin_edges[:-1]) - 1
        ref_binned = np.clip(ref_binned, 0, self.n_bins - 1)
        
        # Count per bin
        self._reference_bins = np.bincount(ref_binned, minlength=self.n_bins)
        
        logger.info(
            f"CBPE fitted on {reference_data.n_samples} samples, "
            f"confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]"
        )
        logger.debug(f"Reference bin counts: {self._reference_bins}")
    
    def detect(self, test_data: Dataset) -> DetectionResult:
        """
        Detect performance drift by comparing confidence distributions.
        
        Args:
            test_data: Test dataset with predictions
        
        Returns:
            DetectionResult indicating performance drift
        
        Raises:
            ValueError: If prediction probabilities not available
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        
        if test_data.prediction_probabilities is None:
            raise ValueError(
                "CBPE requires prediction_probabilities in the test dataset."
            )
        
        logger.info(f"Running CBPE on {test_data.n_samples} test samples")
        
        # Get test confidence
        pred_proba = test_data.prediction_probabilities
        
        if len(pred_proba.shape) == 1:
            confidence = pred_proba
        else:
            confidence = np.max(pred_proba, axis=1)
        
        # Bin the test confidence using same edges
        test_binned = np.digitize(confidence, self._bin_edges[:-1]) - 1
        test_binned = np.clip(test_binned, 0, self.n_bins - 1)
        
        test_bins = np.bincount(test_binned, minlength=self.n_bins)
        
        logger.debug(f"Test bin counts: {test_bins}")
        
        # Filter bins with sufficient counts
        valid_bins = (self._reference_bins >= self.min_bin_count) | (test_bins >= self.min_bin_count)
        
        ref_counts = self._reference_bins[valid_bins]
        test_counts = test_bins[valid_bins]
        
        if len(ref_counts) < 2:
            logger.warning("Not enough bins with sufficient data for chi-square test")
            return DetectionResult(
                detector_name=self.name,
                drift_detected=False,
                severity=DriftSeverity.NONE,
                score=0.0,
                threshold=self.threshold,
                metadata={"message": "Insufficient data in bins"}
            )
        
        # Add small smoothing so bins with zero reference count have non-zero expectation.
        # This avoids inf statistics in chi-square while preserving strong shift signals.
        smoothed_ref_counts = ref_counts.astype(float) + self.smoothing_epsilon

        # Normalize to get expected proportions
        ref_proportions = smoothed_ref_counts / smoothed_ref_counts.sum()
        expected_counts = ref_proportions * test_counts.sum()
        
        # Chi-square test
        try:
            chi2_stat, p_value = stats.chisquare(test_counts, expected_counts)
            chi2_stat = float(chi2_stat)
            p_value = float(p_value)
        except Exception as e:
            logger.warning(f"Chi-square test failed: {e}")
            return DetectionResult(
                detector_name=self.name,
                drift_detected=False,
                severity=DriftSeverity.NONE,
                score=0.0,
                threshold=self.threshold,
            )
        
        # Compute additional metrics
        # Mean confidence shift
        ref_confidence = []
        for i, count in enumerate(self._reference_bins):
            ref_confidence.extend([self._bin_edges[i]] * count)
        ref_mean_conf = np.mean(ref_confidence) if ref_confidence else 0.5
        
        test_mean_conf = np.mean(confidence)
        confidence_shift = abs(test_mean_conf - ref_mean_conf)
        
        # Determine drift
        drift_detected = p_value < self.threshold
        
        # Determine severity
        if p_value >= self.threshold:
            severity = DriftSeverity.NONE
        elif p_value >= self.threshold / 2:
            severity = DriftSeverity.LOW
        elif p_value >= self.threshold / 10:
            severity = DriftSeverity.MEDIUM
        elif p_value >= self.threshold / 100:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL
        
        logger.info(
            f"CBPE result: drift={drift_detected}, "
            f"p-value={p_value:.4f}, chi2={chi2_stat:.2f}, "
            f"confidence_shift={confidence_shift:.4f}, severity={severity.value}"
        )
        
        return DetectionResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            severity=severity,
            score=chi2_stat,
            threshold=self.threshold,
            p_value=p_value,
            statistic=chi2_stat,
            metadata={
                "chi2_statistic": chi2_stat,
                "n_bins": self.n_bins,
                "n_valid_bins": len(ref_counts),
                "smoothing_epsilon": float(self.smoothing_epsilon),
                "ref_mean_confidence": float(ref_mean_conf),
                "test_mean_confidence": float(test_mean_conf),
                "confidence_shift": float(confidence_shift),
                "interpretation": (
                    "Confidence distribution has shifted, suggesting performance change"
                    if drift_detected else
                    "Confidence distribution stable, performance likely unchanged"
                )
            }
        )
