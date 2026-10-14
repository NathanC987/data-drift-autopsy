"""PCA reconstruction-error detector for embedding drift."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.result import DetectionResult
from drift_autopsy.registry import DetectorRegistry
from drift_autopsy.stats import permutation_test, severity_from_p_value

logger = logging.getLogger(__name__)


def reconstruction_increase(X: np.ndarray, Y: np.ndarray, explained_variance_ratio: float,
                            random_state: int) -> float:
    """Relative rise in mean squared PCA reconstruction error of Y over held-out X.

    X is split at random into halves. PCA (keeping ``explained_variance_ratio``
    of the variance) is fitted on the standardised first half; the reference
    error is measured on the second half, so both errors are out of sample and
    the statistic is near zero when Y comes from the reference distribution.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(X))
    fit, hold = X[idx[: len(X) // 2]], X[idx[len(X) // 2:]]
    mean = fit.mean(axis=0)
    std = fit.std(axis=0)
    std[std == 0] = 1.0
    pca = PCA(n_components=explained_variance_ratio, svd_solver="full", random_state=random_state)
    pca.fit((fit - mean) / std)

    def err(Z):
        Zn = (Z - mean) / std
        return float(np.mean((Zn - pca.inverse_transform(pca.transform(Zn))) ** 2))

    ref = err(hold)
    return (err(Y) - ref) / max(ref, 1e-12)


@DetectorRegistry.register("pca_reconstruction")
class PCAReconstructionError(BaseDriftDetector):
    """Relative rise in PCA reconstruction error, with a permutation test.

    Args:
        threshold: Significance level of the permutation test (default: 0.05)
        explained_variance_ratio: Variance kept by the PCA (default: 0.95)
        n_permutations: Permutations for the p-value (default: 200)
        max_samples: Per-sample cap; larger samples are randomly subsampled (default: 3000)
        random_state: Seed for subsampling, PCA and permutations (default: 42)
    """

    def __init__(self, threshold: float = 0.05, explained_variance_ratio: float = 0.95,
                 n_permutations: int = 200, max_samples: int = 3000, random_state: int = 42):
        super().__init__(name="pca_reconstruction")
        self.threshold = threshold
        self.explained_variance_ratio = explained_variance_ratio
        self.n_permutations = n_permutations
        self.max_samples = max_samples
        self.random_state = random_state

    def _matrix(self, dataset: Dataset, rng: np.random.Generator) -> np.ndarray:
        X = dataset.to_pandas()[dataset.numeric_features].to_numpy(dtype=float)
        if len(X) > self.max_samples:
            X = X[rng.choice(len(X), self.max_samples, replace=False)]
        return X

    def detect(self, test_data: Dataset) -> DetectionResult:
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        rng = np.random.default_rng(self.random_state)
        X = self._matrix(self._reference_data, rng)
        Y = self._matrix(test_data, rng)
        test = permutation_test(
            lambda a, b: reconstruction_increase(a, b, self.explained_variance_ratio, self.random_state),
            X, Y, self.n_permutations, random_state=self.random_state,
        )
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
                      "explained_variance_ratio": self.explained_variance_ratio},
        )
