"""Frechet distance between Gaussians fitted to reference and test embeddings."""

from __future__ import annotations

import logging

import numpy as np
from scipy.linalg import sqrtm

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.result import DetectionResult
from drift_autopsy.registry import DetectorRegistry
from drift_autopsy.stats import permutation_test, severity_from_p_value

logger = logging.getLogger(__name__)


def frechet_distance(X: np.ndarray, Y: np.ndarray, covariance_eps: float = 1e-6) -> float:
    """||mu_x - mu_y||^2 + tr(S_x + S_y - 2 (S_x S_y)^{1/2}) for Gaussian fits.

    This is the statistic behind FID (Heusel et al., 2017), here computed on the
    monitored model's own embedding space rather than Inception features.
    """
    d = X.shape[1]
    cx = np.atleast_2d(np.cov(X, rowvar=False)) + np.eye(d) * covariance_eps
    cy = np.atleast_2d(np.cov(Y, rowvar=False)) + np.eye(d) * covariance_eps
    diff = X.mean(axis=0) - Y.mean(axis=0)
    covmean = sqrtm(cx @ cy)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cx + cy - 2.0 * covmean))


@DetectorRegistry.register("frechet_distance")
@DetectorRegistry.register("fid_distance")
class FrechetDistance(BaseDriftDetector):
    """Frechet distance with a permutation test.

    Args:
        threshold: Significance level of the permutation test (default: 0.05)
        n_permutations: Permutations for the p-value (default: 200)
        max_samples: Per-sample cap; larger samples are randomly subsampled (default: 3000)
        covariance_eps: Ridge added to each covariance (default: 1e-6)
        random_state: Seed for subsampling and permutations (default: 42)
    """

    def __init__(self, threshold: float = 0.05, n_permutations: int = 200, max_samples: int = 3000,
                 covariance_eps: float = 1e-6, random_state: int = 42):
        super().__init__(name="frechet_distance")
        self.threshold = threshold
        self.n_permutations = n_permutations
        self.max_samples = max_samples
        self.covariance_eps = covariance_eps
        self.random_state = random_state

    def _matrix(self, dataset: Dataset, rng: np.random.Generator) -> np.ndarray:
        cols = [c for c in dataset.numeric_features]
        X = dataset.to_pandas()[cols].to_numpy(dtype=float)
        if len(X) > self.max_samples:
            X = X[rng.choice(len(X), self.max_samples, replace=False)]
        return X

    def detect(self, test_data: Dataset) -> DetectionResult:
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        rng = np.random.default_rng(self.random_state)
        X = self._matrix(self._reference_data, rng)
        Y = self._matrix(test_data, rng)
        test = permutation_test(lambda a, b: frechet_distance(a, b, self.covariance_eps), X, Y,
                                self.n_permutations, random_state=self.random_state)
        drift = test["p_value"] < self.threshold
        return DetectionResult(
            detector_name=self.name,
            drift_detected=bool(drift),
            severity=severity_from_p_value(test["p_value"], self.threshold),
            score=test["statistic"],
            threshold=self.threshold,
            p_value=test["p_value"],
            statistic=test["statistic"],
            metadata={"n_permutations": self.n_permutations, "n_reference": len(X), "n_test": len(Y),
                      "n_features": X.shape[1], "covariance_eps": self.covariance_eps},
        )


FIDDistance = FrechetDistance  # backwards-compatible name
