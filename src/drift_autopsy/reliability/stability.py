"""Prediction stability checks via input perturbations."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


class StabilityChecker:
    """
    Evaluate prediction sensitivity under small perturbations.

    Higher score means lower stability and higher reliability risk.
    """

    def __init__(
        self,
        model: Any,
        data_type: str,
        perturbation_strength: float = 0.01,
        n_perturbations: int = 5,
        random_state: int = 42,
    ):
        self.model = model
        self.data_type = data_type
        self.perturbation_strength = perturbation_strength
        self.n_perturbations = n_perturbations
        self.random_state = random_state

    def _predict_output(self, x: Any) -> np.ndarray:
        model_input = self._prepare_model_input(x)
        if hasattr(self.model, "predict_proba"):
            out = np.asarray(self.model.predict_proba(model_input), dtype=float)
        else:
            out = np.asarray(self.model.predict(model_input), dtype=float)
            if out.ndim == 1:
                out = out.reshape(-1, 1)
        return out

    def _prepare_model_input(self, x: Any) -> Any:
        """Prepare model input shape for stable predict/predict_proba calls."""
        if self.data_type == "text":
            if isinstance(x, str):
                return [x]
            if isinstance(x, list):
                return x
            return [str(x)]

        if self.data_type == "tabular":
            if hasattr(x, "to_numpy"):
                arr = np.asarray(x.to_numpy(), dtype=float)
            else:
                arr = np.asarray(x, dtype=float)

            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr

        arr = np.asarray(x)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _perturb_tabular(self, x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        std = np.std(x, axis=0, keepdims=True) + 1e-8
        noise = rng.normal(loc=0.0, scale=self.perturbation_strength * std, size=x.shape)
        return x + noise

    def _perturb_image(self, x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        noise = rng.normal(loc=0.0, scale=self.perturbation_strength, size=x.shape)
        return np.clip(x + noise, 0.0, 1.0)

    def _perturb_text(self, x: Any, rng: np.random.RandomState) -> Any:
        if isinstance(x, str):
            tokens = x.split()
            if len(tokens) <= 1:
                return x
            keep_mask = rng.rand(len(tokens)) > self.perturbation_strength
            if not keep_mask.any():
                keep_mask[rng.randint(0, len(tokens))] = True
            return " ".join([tok for tok, keep in zip(tokens, keep_mask) if keep])

        if isinstance(x, list) and x and isinstance(x[0], str):
            return [self._perturb_text(item, rng) for item in x]

        return x

    def _perturb(self, x: Any, rng: np.random.RandomState) -> Any:
        if self.data_type == "tabular":
            arr = np.asarray(x, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return self._perturb_tabular(arr, rng)

        if self.data_type == "image":
            arr = np.asarray(x, dtype=float)
            return self._perturb_image(arr, rng)

        if self.data_type == "text":
            return self._perturb_text(x, rng)

        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return self._perturb_tabular(arr, rng)

    @staticmethod
    def _prediction_distance(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return 1.0
        diff = np.abs(a - b)
        if diff.size == 0:
            return 0.0
        return float(np.mean(diff))

    def compute_stability(self, x: Any) -> Dict[str, Any]:
        """Compute normalized stability score in [0, 1] for a single input."""
        rng = np.random.RandomState(self.random_state)
        original = self._predict_output(x)

        distances = []
        for _ in range(self.n_perturbations):
            perturbed_x = self._perturb(x, rng)
            perturbed_pred = self._predict_output(perturbed_x)
            distances.append(self._prediction_distance(original, perturbed_pred))

        avg_distance = float(np.mean(distances)) if distances else 0.0
        score = _clip01(avg_distance)

        return {
            "stability_score": score,
            "raw_difference": avg_distance,
            "metadata": {
                "n_perturbations": self.n_perturbations,
                "perturbation_strength": self.perturbation_strength,
            },
        }
