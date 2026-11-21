"""Out-of-distribution scoring for feature vectors (tabular rows or embeddings).

Representation. Each row is mapped to a compact vector before scoring:
numeric features are z-scored with the reference mean and spread, and each
nominal feature is replaced by the rarity of its value in the reference,
-log((count + 0.5) / (n + 0.5 * K)) with K reference categories (a Jeffreys
pseudo-count, so a category never seen in the reference gets the largest,
finite rarity). Embeddings have no nominal features and are simply z-scored.

Score. Two detectors are fitted on the reference representation:

* an Isolation Forest (Liu et al., 2008), anomaly score = -decision_function;
* the Euclidean norm of the representation (distance from the reference mean
  in standard units), which catches inputs whose values have been rescaled
  beyond the reference range, a case Isolation Forests under-flag.

Each raw score s is mapped to [0, 1] as clip((s - q_lo) / (q_hi - q_lo), 0, 1),
where q_lo and q_hi are the ``zero_percentile`` and ``one_percentile`` of the
same score over the reference. By construction, (100 - zero_percentile)% of
reference rows score above 0 and (100 - one_percentile)% score 1. The OOD score
is the maximum of the two mapped scores.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest


class TabularOODEncoder:
    """Numeric z-scores plus per-category rarity (see module docstring)."""

    def __init__(self, categorical_indices: Optional[Sequence[int]] = None):
        self.categorical_indices = list(categorical_indices or [])
        self._numeric_indices: List[int] = []
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._rarity: List[Dict[float, float]] = []
        self._unseen_rarity: List[float] = []

    def fit(self, X: np.ndarray) -> "TabularOODEncoder":
        X = np.asarray(X, dtype=float)
        cat = set(self.categorical_indices)
        self._numeric_indices = [j for j in range(X.shape[1]) if j not in cat]
        if self._numeric_indices:
            num = X[:, self._numeric_indices]
            self._mean = num.mean(axis=0)
            std = num.std(axis=0)
            std[std == 0] = 1.0
            self._std = std
        n = X.shape[0]
        self._rarity, self._unseen_rarity = [], []
        for j in self.categorical_indices:
            values, counts = np.unique(X[:, j], return_counts=True)
            denom = n + 0.5 * len(values)
            self._rarity.append({float(v): float(-np.log((c + 0.5) / denom)) for v, c in zip(values, counts)})
            self._unseen_rarity.append(float(-np.log(0.5 / denom)))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        blocks = []
        if self._numeric_indices:
            blocks.append((X[:, self._numeric_indices] - self._mean) / self._std)
        for k, j in enumerate(self.categorical_indices):
            table, unseen = self._rarity[k], self._unseen_rarity[k]
            blocks.append(np.array([table.get(float(v), unseen) for v in X[:, j]]).reshape(-1, 1))
        return np.hstack(blocks) if blocks else np.empty((X.shape[0], 0))


class OODDetector:
    """Out-of-distribution score in [0, 1] for feature vectors (see module docstring)."""

    def __init__(
        self,
        categorical_indices: Optional[Sequence[int]] = None,
        zero_percentile: float = 90.0,
        one_percentile: float = 99.0,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        if not 0.0 <= zero_percentile < one_percentile <= 100.0:
            raise ValueError("need 0 <= zero_percentile < one_percentile <= 100")
        self.encoder = TabularOODEncoder(categorical_indices)
        self.zero_percentile = zero_percentile
        self.one_percentile = one_percentile
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._iforest: Optional[IsolationForest] = None
        self._iforest_range = (0.0, 1.0)
        self._distance_range = (0.0, 1.0)

    def _range(self, scores: np.ndarray):
        lo = float(np.percentile(scores, self.zero_percentile))
        hi = float(np.percentile(scores, self.one_percentile))
        return (lo, max(hi, lo + 1e-12))

    @staticmethod
    def _map(scores: np.ndarray, rng) -> np.ndarray:
        lo, hi = rng
        return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)

    def fit(self, reference_X: np.ndarray) -> "OODDetector":
        Z = self.encoder.fit(reference_X).transform(reference_X)
        self._iforest = IsolationForest(n_estimators=self.n_estimators, random_state=self.random_state)
        self._iforest.fit(Z)
        self._iforest_range = self._range(-self._iforest.decision_function(Z))
        self._distance_range = self._range(np.linalg.norm(Z, axis=1))
        return self

    def score_components(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        if self._iforest is None:
            raise RuntimeError("OODDetector must be fitted before scoring")
        Z = self.encoder.transform(X)
        raw_iforest = -self._iforest.decision_function(Z)
        raw_distance = np.linalg.norm(Z, axis=1)
        iforest = self._map(raw_iforest, self._iforest_range)
        distance = self._map(raw_distance, self._distance_range)
        # Unclipped score on the same scale (1 = the reference's one_percentile),
        # for ranking inputs that all score 1 after clipping.
        raw = np.maximum((raw_iforest - self._iforest_range[0]) / (self._iforest_range[1] - self._iforest_range[0]),
                         (raw_distance - self._distance_range[0]) / (self._distance_range[1] - self._distance_range[0]))
        return {"isolation_forest": iforest, "distance": distance, "ood": np.maximum(iforest, distance),
                "ood_unclipped": raw}

    def score(self, X: np.ndarray) -> np.ndarray:
        """OOD score in [0, 1] per row."""
        return self.score_components(X)["ood"]
