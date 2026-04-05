"""FID-style Frechet distance detector for embedding drift."""

from __future__ import annotations

import logging

import numpy as np
from scipy.linalg import sqrtm

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.result import DetectionResult, DriftSeverity
from drift_autopsy.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("fid_distance")
class FIDDistance(BaseDriftDetector):
    """Detect drift using Frechet distance between reference and test embedding Gaussians."""

    def __init__(
        self,
        threshold: float = 50.0,
        covariance_eps: float = 1e-6,
    ):
        super().__init__(name="fid_distance")
        self.threshold = threshold
        self.covariance_eps = covariance_eps
        self._feature_cols: list[str] = []
        self._mu_ref: np.ndarray | None = None
        self._cov_ref: np.ndarray | None = None

    def _to_numeric_matrix(self, dataset: Dataset) -> np.ndarray:
        df = dataset.to_pandas()
        if not self._feature_cols:
            self._feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[self._feature_cols].fillna(0).to_numpy(dtype=float)

    def _regularized_cov(self, X: np.ndarray) -> np.ndarray:
        cov = np.atleast_2d(np.cov(X, rowvar=False))
        d = cov.shape[0]
        return cov + np.eye(d) * self.covariance_eps

    def fit(self, reference_data: Dataset) -> None:
        super().fit(reference_data)

        X = self._to_numeric_matrix(reference_data)
        if X.shape[1] == 0:
            raise ValueError("FID detector requires numeric features")

        self._mu_ref = X.mean(axis=0)
        self._cov_ref = self._regularized_cov(X)

        logger.info("FID distance fitted on %d samples with %d features", X.shape[0], X.shape[1])

    def detect(self, test_data: Dataset) -> DetectionResult:
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        assert self._mu_ref is not None
        assert self._cov_ref is not None

        X = self._to_numeric_matrix(test_data)
        mu_test = X.mean(axis=0)
        cov_test = self._regularized_cov(X)

        diff = self._mu_ref - mu_test
        cov_prod = self._cov_ref @ cov_test
        covmean = sqrtm(cov_prod)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = float(diff @ diff + np.trace(self._cov_ref + cov_test - 2.0 * covmean))
        if not np.isfinite(fid):
            fid = float("inf")

        drift_detected = fid >= self.threshold
        if fid < self.threshold:
            severity = DriftSeverity.NONE
        elif fid < self.threshold * 1.5:
            severity = DriftSeverity.LOW
        elif fid < self.threshold * 2.5:
            severity = DriftSeverity.MEDIUM
        elif fid < self.threshold * 4:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL

        return DetectionResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            severity=severity,
            score=float(fid),
            threshold=float(self.threshold),
            statistic=float(fid),
            metadata={
                "n_features": len(self._feature_cols),
                "covariance_eps": float(self.covariance_eps),
            },
        )
