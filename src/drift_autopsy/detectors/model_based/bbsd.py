"""Black box shift detection on the model's outputs (BBSDs)."""

import logging

import numpy as np
from scipy import stats

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.result import DetectionResult
from drift_autopsy.registry import DetectorRegistry
from drift_autopsy.stats import severity_from_p_value

logger = logging.getLogger(__name__)


@DetectorRegistry.register("bbsd")
class BBSD(BaseDriftDetector):
    """
    BBSDs (Lipton et al., 2018; Rabanser et al., 2019): two-sample KS tests on
    each class probability of the monitored model, reference vs production,
    with a Bonferroni correction over the classes. For a binary model the two
    class probabilities carry the same information, so one test is run.
    This was the best single detector in the comparison of Rabanser et al.

    Args:
        threshold: Family-wise significance level (default: 0.05)
    """

    def __init__(self, threshold: float = 0.05):
        super().__init__(name="bbsd")
        self.threshold = threshold

    def fit(self, reference_data: Dataset) -> None:
        super().fit(reference_data)
        if reference_data.prediction_probabilities is None:
            raise ValueError("BBSD needs prediction_probabilities")

    @staticmethod
    def _columns(proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 1:
            return proba[:, None]
        return proba[:, 1:2] if proba.shape[1] == 2 else proba

    def detect(self, test_data: Dataset) -> DetectionResult:
        if test_data.prediction_probabilities is None:
            raise ValueError("BBSD needs prediction_probabilities")
        ref = self._columns(self._reference_data.prediction_probabilities)
        prod = self._columns(test_data.prediction_probabilities)
        results = [stats.ks_2samp(ref[:, k], prod[:, k]) for k in range(ref.shape[1])]
        p_values = np.array([r.pvalue for r in results])
        m = len(p_values)
        p_min = float(p_values.min())
        p_adj = float(min(1.0, m * p_min))
        statistic = float(max(r.statistic for r in results))
        return DetectionResult(
            detector_name=self.name,
            drift_detected=bool(p_adj < self.threshold),
            severity=severity_from_p_value(p_adj, self.threshold),
            score=statistic,
            threshold=self.threshold,
            p_value=p_adj,
            statistic=statistic,
            metadata={"per_class_p_values": p_values.tolist(), "per_class_statistics":
                      [float(r.statistic) for r in results], "correction": "bonferroni"},
        )
