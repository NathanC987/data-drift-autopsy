"""Explanation consistency checks using SHAP and Grad-CAM style artifacts."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.rca import SHAPAnalyzer


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    similarity = float(np.dot(a, b) / denom)
    return float(np.clip(1.0 - similarity, 0.0, 1.0))


class ExplanationConsistencyChecker:
    """
    Compute explanation consistency risk in [0, 1].

    For tabular data, reuses SHAPAnalyzer importance changes.
    For image data, compares Grad-CAM style heatmaps via adapter callback.
    """

    def __init__(
        self,
        model: Any,
        data_type: str,
        gradcam_extractor: Optional[Callable[[Any, Any], np.ndarray]] = None,
        shap_background_samples: int = 100,
        shap_test_samples: int = 100,
    ):
        self.model = model
        self.data_type = data_type
        self.gradcam_extractor = gradcam_extractor
        self.shap_background_samples = shap_background_samples
        self.shap_test_samples = shap_test_samples

    @staticmethod
    def _vector_from_shap_distribution(distribution_changes: Dict[str, Any]) -> np.ndarray:
        ordered = []
        for feature in sorted(distribution_changes.keys()):
            payload = distribution_changes.get(feature, {})
            if not isinstance(payload, dict):
                continue
            ordered.append(float(payload.get("change", 0.0)))
        return np.asarray(ordered, dtype=float)

    def _tabular_score(
        self,
        reference_data: Dataset,
        current_data: Dataset,
        baseline_explanation_vector: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        analyzer = SHAPAnalyzer(
            n_background_samples=self.shap_background_samples,
            n_test_samples=self.shap_test_samples,
        )
        rca_result = analyzer.analyze(reference_data, current_data, model=self.model)

        distribution_changes = rca_result.distribution_changes or {}
        current_vector = self._vector_from_shap_distribution(distribution_changes)

        if baseline_explanation_vector is None:
            score = float(np.clip(np.mean(np.abs(current_vector)) if len(current_vector) else 0.0, 0.0, 1.0))
            method = "shap_absolute_change"
        else:
            score = _cosine_distance(np.asarray(baseline_explanation_vector, dtype=float), current_vector)
            method = "shap_cosine_shift"

        return {
            "explanation_score": float(np.clip(score, 0.0, 1.0)),
            "metadata": {
                "method": method,
                "n_features": int(len(current_vector)),
            },
            "details": {
                "top_importance_changes": (rca_result.explanations or {}).get("top_importance_changes", []),
            },
        }

    def _image_score(
        self,
        baseline_input: Any,
        current_input: Any,
        baseline_heatmap: Optional[np.ndarray] = None,
        current_heatmap: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if self.gradcam_extractor is not None:
            if baseline_heatmap is None:
                baseline_heatmap = self.gradcam_extractor(self.model, baseline_input)
            if current_heatmap is None:
                current_heatmap = self.gradcam_extractor(self.model, current_input)

        if baseline_heatmap is None or current_heatmap is None:
            return {
                "explanation_score": 0.5,
                "metadata": {"method": "missing_gradcam"},
                "details": {},
            }

        score = _cosine_distance(np.asarray(baseline_heatmap), np.asarray(current_heatmap))
        return {
            "explanation_score": float(np.clip(score, 0.0, 1.0)),
            "metadata": {"method": "gradcam_cosine_shift"},
            "details": {},
        }

    def compute(
        self,
        reference_data: Optional[Dataset] = None,
        current_data: Optional[Dataset] = None,
        baseline_input: Optional[Any] = None,
        current_input: Optional[Any] = None,
        baseline_explanation_vector: Optional[np.ndarray] = None,
        baseline_heatmap: Optional[np.ndarray] = None,
        current_heatmap: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute explanation consistency score in [0, 1]."""
        if self.data_type == "tabular" and reference_data is not None and current_data is not None:
            return self._tabular_score(
                reference_data=reference_data,
                current_data=current_data,
                baseline_explanation_vector=baseline_explanation_vector,
            )

        if self.data_type == "image" and baseline_input is not None and current_input is not None:
            return self._image_score(
                baseline_input=baseline_input,
                current_input=current_input,
                baseline_heatmap=baseline_heatmap,
                current_heatmap=current_heatmap,
            )

        return {
            "explanation_score": 0.5,
            "metadata": {"method": "fallback"},
            "details": {},
        }
