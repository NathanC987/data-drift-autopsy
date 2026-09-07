"""Out-of-distribution detection for reliability analysis."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest


def _normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low = np.min(values) if len(values) else 0.0
    high = np.max(values) if len(values) else 1.0
    if high - low < eps:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - low) / (high - low + eps), 0.0, 1.0)


class OODDetector:
    """
    Generic OOD detector with tabular and embedding-based backends.

    The returned score is OOD risk in [0, 1], where higher means more anomalous.
    """

    def __init__(
        self,
        data_type: str = "tabular",
        method: str = "auto",
        embedding_extractor: Optional[Callable[[Any], np.ndarray]] = None,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.data_type = data_type
        self.method = method
        self.embedding_extractor = embedding_extractor
        self.contamination = contamination
        self.random_state = random_state

        self._reference_vectors: Optional[np.ndarray] = None
        self._ref_mean: Optional[np.ndarray] = None
        self._ref_std: Optional[np.ndarray] = None
        self._iforest: Optional[IsolationForest] = None
        self._distance_p95: float = 1.0
        self._distance_p50: float = 0.0

    @staticmethod
    def _to_2d_numeric(x: Any) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(float)

    def _vectorize(self, x: Any) -> np.ndarray:
        if self.embedding_extractor is not None:
            vectors = np.asarray(self.embedding_extractor(x), dtype=float)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            return vectors

        if self.data_type == "text":
            if isinstance(x, str):
                x = [x]
            if isinstance(x, list) and x and isinstance(x[0], str):
                lengths = np.array([len(t.split()) for t in x], dtype=float).reshape(-1, 1)
                chars = np.array([len(t) for t in x], dtype=float).reshape(-1, 1)
                return np.hstack([lengths, chars])

        return self._to_2d_numeric(x)

    def fit(self, reference_data: Any) -> "OODDetector":
        """Fit OOD detector on reference data."""
        reference_vectors = self._vectorize(reference_data)
        self._reference_vectors = reference_vectors

        auto_method = "isolation_forest" if self.data_type == "tabular" else "embedding_distance"
        selected_method = auto_method if self.method == "auto" else self.method

        if selected_method == "isolation_forest":
            self._iforest = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
            )
            self._iforest.fit(reference_vectors)
        else:
            self._ref_mean = np.mean(reference_vectors, axis=0)
            self._ref_std = np.std(reference_vectors, axis=0) + 1e-8
            dists = np.linalg.norm((reference_vectors - self._ref_mean) / self._ref_std, axis=1)
            self._distance_p50 = float(np.percentile(dists, 50)) if len(dists) else 0.0
            self._distance_p95 = float(np.percentile(dists, 95)) if len(dists) else 1.0

        return self

    def compute_ood_score_batch(self, x: Any, reference_data: Optional[Any] = None) -> Dict[str, Any]:
        """Compute OOD score for batch samples in [0, 1]."""
        if self._reference_vectors is None:
            if reference_data is None:
                raise ValueError("Reference data is required before computing OOD scores")
            self.fit(reference_data)

        vectors = self._vectorize(x)

        if self._iforest is not None:
            decision = self._iforest.decision_function(vectors)
            raw_risk = -decision
            scores = _normalize(raw_risk)
            method = "isolation_forest"
        else:
            centered = (vectors - self._ref_mean) / self._ref_std
            dist = np.linalg.norm(centered, axis=1)
            denom = max(self._distance_p95 - self._distance_p50, 1e-8)
            scores = np.clip((dist - self._distance_p50) / denom, 0.0, 1.0)
            method = "embedding_distance"

        return {
            "scores": scores.astype(float),
            "mean_score": float(np.mean(scores)) if len(scores) else 0.0,
            "metadata": {"method": method},
        }

    def compute_ood_score(self, x: Any, reference_data: Optional[Any] = None) -> float:
        """Compute OOD score for a single sample in [0, 1]."""
        batch = self.compute_ood_score_batch([x] if not isinstance(x, np.ndarray) else x, reference_data=reference_data)
        if len(batch["scores"]) == 0:
            return 0.0
        return float(np.clip(batch["scores"][0], 0.0, 1.0))