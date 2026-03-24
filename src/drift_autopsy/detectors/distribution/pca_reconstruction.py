"""PCA reconstruction-error detector for embedding drift."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.decomposition import PCA

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.result import DetectionResult, DriftSeverity
from drift_autopsy.registry import DetectorRegistry

logger = logging.getLogger(__name__)


@DetectorRegistry.register("pca_reconstruction")
class PCAReconstructionError(BaseDriftDetector):
    """Detect drift by increase in PCA reconstruction error on test embeddings."""

    def __init__(
        self,
        threshold: float = 0.15,
        explained_variance_ratio: float = 0.95,
        random_state: int = 42,
    ):
        super().__init__(name="pca_reconstruction")
        self.threshold = threshold
        self.explained_variance_ratio = explained_variance_ratio
        self.random_state = random_state
        self._feature_cols: list[str] = []
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._pca: PCA | None = None
        self._reference_error: float | None = None

    def _to_numeric_matrix(self, dataset: Dataset) -> np.ndarray:
        df = dataset.to_pandas()
        if not self._feature_cols:
            self._feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[self._feature_cols].fillna(0).to_numpy(dtype=float)

    def _reconstruction_error(self, X: np.ndarray) -> float:
        assert self._mean is not None
        assert self._std is not None
        assert self._pca is not None

        Xn = (X - self._mean) / self._std
        Z = self._pca.transform(Xn)
        Xr = self._pca.inverse_transform(Z)
        mse_per_row = np.mean((Xn - Xr) ** 2, axis=1)
        return float(np.mean(mse_per_row))

    def fit(self, reference_data: Dataset) -> None:
        super().fit(reference_data)

        X = self._to_numeric_matrix(reference_data)
        if X.shape[1] == 0:
            raise ValueError("PCA reconstruction detector requires numeric features")

        self._mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        self._std = std

        Xn = (X - self._mean) / self._std
        self._pca = PCA(
            n_components=self.explained_variance_ratio,
            svd_solver="full",
            random_state=self.random_state,
        )
        self._pca.fit(Xn)

        self._reference_error = self._reconstruction_error(X)
        logger.info(
            "PCA reconstruction fitted: n_features=%d n_components=%d ref_error=%.6f",
            X.shape[1],
            int(self._pca.n_components_),
            self._reference_error,
        )

    def detect(self, test_data: Dataset) -> DetectionResult:
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        assert self._reference_error is not None

        X_test = self._to_numeric_matrix(test_data)
        test_error = self._reconstruction_error(X_test)

        denom = max(self._reference_error, 1e-12)
        relative_increase = max(0.0, (test_error - self._reference_error) / denom)

        drift_detected = relative_increase >= self.threshold
        if relative_increase < self.threshold:
            severity = DriftSeverity.NONE
        elif relative_increase < self.threshold * 1.5:
            severity = DriftSeverity.LOW
        elif relative_increase < self.threshold * 2.5:
            severity = DriftSeverity.MEDIUM
        elif relative_increase < self.threshold * 4:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.CRITICAL

        return DetectionResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            severity=severity,
            score=float(relative_increase),
            threshold=float(self.threshold),
            statistic=float(relative_increase),
            metadata={
                "reference_reconstruction_error": float(self._reference_error),
                "test_reconstruction_error": float(test_error),
                "n_features": len(self._feature_cols),
                "n_components": int(self._pca.n_components_) if self._pca is not None else 0,
                "explained_variance_ratio_target": float(self.explained_variance_ratio),
            },
        )
