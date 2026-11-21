"""Prediction instability under small input perturbations.

For each row x the model is re-run on ``n_perturbations`` noisy copies
x + e, with e_j ~ N(0, (strength * sd_j)^2) for every perturbed feature j and
e_j = 0 otherwise. sd_j is the reference standard deviation of feature j, so
the noise is a fixed fraction of each feature's natural spread. Nominal
features are never perturbed (a perturbed code is a different category, not a
nearby value). Instability is the mean absolute change in the predicted class
probabilities, averaged over classes and perturbations; it lies in [0, 1].
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


class StabilityChecker:
    """Vectorised instability score (see module docstring)."""

    def __init__(
        self,
        model: Any,
        perturb_indices: Optional[Sequence[int]] = None,
        strength: float = 0.1,
        n_perturbations: int = 5,
        random_state: int = 42,
    ):
        if strength <= 0 or n_perturbations < 1:
            raise ValueError("strength must be > 0 and n_perturbations >= 1")
        self.model = model
        self.perturb_indices = None if perturb_indices is None else list(perturb_indices)
        self.strength = strength
        self.n_perturbations = n_perturbations
        self.random_state = random_state
        self._scale: Optional[np.ndarray] = None

    def fit(self, reference_X: np.ndarray) -> "StabilityChecker":
        X = np.asarray(reference_X, dtype=float)
        sd = X.std(axis=0)
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[self.perturb_indices if self.perturb_indices is not None else slice(None)] = True
        self._scale = np.where(mask, self.strength * sd, 0.0)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Instability in [0, 1] per row."""
        if self._scale is None:
            raise RuntimeError("StabilityChecker must be fitted on reference data first")
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        base = np.asarray(self.model.predict_proba(X), dtype=float)
        total = np.zeros(len(X))
        for _ in range(self.n_perturbations):
            noisy = X + rng.normal(size=X.shape) * self._scale
            total += np.abs(np.asarray(self.model.predict_proba(noisy), dtype=float) - base).mean(axis=1)
        return np.clip(total / self.n_perturbations, 0.0, 1.0)
