"""Classifier two-sample test (domain classifier) for drift detection."""

import logging

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.result import DetectionResult
from drift_autopsy.registry import DetectorRegistry
from drift_autopsy.stats import severity_from_p_value

logger = logging.getLogger(__name__)


@DetectorRegistry.register("domain_classifier")
class DomainClassifier(BaseDriftDetector):
    """
    Classifier two-sample test (Lopez-Paz and Oquab, 2017), the "domain
    classifier" of Rabanser et al. (2019).

    Equal-sized samples of the reference and production inputs (at most
    ``max_samples`` each, drawn at random) are labelled 0 and 1 and split in
    half, stratified. A gradient boosted tree classifier (scikit-learn
    defaults, no early stopping) is fitted on one half and its accuracy is
    measured on the other. Under no drift the held-out accuracy is Binomial(n,
    1/2), so the p-value is the exact one-sided binomial tail. Nominal codes
    are passed as categorical features.

    Args:
        threshold: Significance level (default: 0.05)
        max_samples: Cap on the rows taken from each side (default: 3000)
        random_state: Seed for subsampling, splitting and the classifier
    """

    def __init__(self, threshold: float = 0.05, max_samples: int = 3000, random_state: int = 42):
        super().__init__(name="domain_classifier")
        self.threshold = threshold
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, reference_data: Dataset) -> None:
        super().fit(reference_data)

    def detect(self, test_data: Dataset) -> DetectionResult:
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        names = list(self._reference_data.feature_names)
        categorical = set(self._reference_data.categorical_features or [])
        rng = np.random.default_rng(self.random_state)
        ref = self._reference_data.to_pandas()[names].to_numpy(dtype=float)
        prod = test_data.to_pandas()[names].to_numpy(dtype=float)
        n = int(min(len(ref), len(prod), self.max_samples))
        ref = ref[rng.choice(len(ref), n, replace=False)]
        prod = prod[rng.choice(len(prod), n, replace=False)]

        X = np.vstack([ref, prod])
        y = np.r_[np.zeros(n), np.ones(n)]
        # categorical codes must be 0..254 for native splits: rank-encode them
        mask = np.array([f in categorical for f in names])
        for j in np.where(mask)[0]:
            levels, codes = np.unique(X[:, j], return_inverse=True)
            if len(levels) > 254:
                mask[j] = False          # too many levels: split on the rank order instead
            X[:, j] = codes
        # split each domain in half, so both halves are balanced
        idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
        rng.shuffle(idx0)
        rng.shuffle(idx1)
        fit_idx = np.r_[idx0[: n // 2], idx1[: n // 2]]
        test_idx = np.r_[idx0[n // 2:], idx1[n // 2:]]
        clf = HistGradientBoostingClassifier(
            categorical_features=mask if mask.any() else None,
            early_stopping=False, random_state=self.random_state,
        ).fit(X[fit_idx], y[fit_idx])
        hits = int((clf.predict(X[test_idx]) == y[test_idx]).sum())
        m = len(test_idx)
        accuracy = hits / m
        p_value = float(stats.binom.sf(hits - 1, m, 0.5))
        drift = p_value < self.threshold
        return DetectionResult(
            detector_name=self.name,
            drift_detected=bool(drift),
            severity=severity_from_p_value(p_value, self.threshold),
            score=float(accuracy),
            threshold=self.threshold,
            p_value=p_value,
            statistic=float(accuracy),
            metadata={"held_out_accuracy": accuracy, "n_held_out": m, "n_per_side": n},
        )
