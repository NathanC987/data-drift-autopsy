"""Confidence extraction utilities for model-agnostic reliability checks."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _safe_clip(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


class ConfidenceExtractor:
    """
    Unified confidence extraction for classification, regression, and sequence models.

    Confidence is always normalized to [0, 1].
    """

    def __init__(
        self,
        model: Any,
        task_type: str = "auto",
        regression_std_estimator: Optional[Callable[[Any], np.ndarray]] = None,
        logits_extractor: Optional[Callable[[Any], np.ndarray]] = None,
    ):
        self.model = model
        self.task_type = task_type
        self.regression_std_estimator = regression_std_estimator
        self.logits_extractor = logits_extractor

    def _infer_task_type(self) -> str:
        if self.task_type != "auto":
            return self.task_type

        if hasattr(self.model, "predict_proba"):
            return "classification"

        if hasattr(self.model, "estimators_"):
            return "regression"

        return "generic"

    @staticmethod
    def _ensure_2d(array: Any) -> np.ndarray:
        x = np.asarray(array)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x

    def _classification_confidence(self, x: Any) -> tuple[np.ndarray, Dict[str, Any]]:
        if not hasattr(self.model, "predict_proba"):
            return np.zeros(1, dtype=float), {"method": "missing_predict_proba"}

        proba = np.asarray(self.model.predict_proba(x), dtype=float)
        if proba.ndim == 1:
            confidence = np.clip(proba, 0.0, 1.0)
        else:
            confidence = np.clip(np.max(proba, axis=1), 0.0, 1.0)

        return confidence, {"method": "predict_proba"}

    def _regression_confidence(self, x: Any) -> tuple[np.ndarray, Dict[str, Any]]:
        if callable(self.regression_std_estimator):
            std = np.asarray(self.regression_std_estimator(x), dtype=float)
            confidence = 1.0 / (1.0 + np.maximum(std, 0.0))
            return np.clip(confidence, 0.0, 1.0), {"method": "external_std"}

        if hasattr(self.model, "estimators_"):
            estimator_predictions = []
            for estimator in getattr(self.model, "estimators_", []):
                if hasattr(estimator, "predict"):
                    estimator_predictions.append(np.asarray(estimator.predict(x), dtype=float))

            if estimator_predictions:
                stack = np.vstack(estimator_predictions)
                std = np.std(stack, axis=0)
                confidence = 1.0 / (1.0 + np.maximum(std, 0.0))
                return np.clip(confidence, 0.0, 1.0), {"method": "ensemble_variance"}

        preds = np.asarray(self.model.predict(x), dtype=float)
        abs_centered = np.abs(preds - np.median(preds))
        scale = np.std(preds) + 1e-8
        confidence = 1.0 - np.clip(abs_centered / (3.0 * scale + 1e-8), 0.0, 1.0)
        return np.clip(confidence, 0.0, 1.0), {"method": "heuristic_prediction_dispersion"}

    def _sequence_confidence(self, x: Any) -> tuple[np.ndarray, Dict[str, Any]]:
        if callable(self.logits_extractor):
            logits = np.asarray(self.logits_extractor(x), dtype=float)
        elif hasattr(self.model, "predict_logits"):
            logits = np.asarray(self.model.predict_logits(x), dtype=float)
        else:
            logits = None

        if logits is None:
            return np.zeros(1, dtype=float), {"method": "missing_logits"}

        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits) / (np.sum(np.exp(logits), axis=1, keepdims=True) + 1e-8)
        confidence = np.max(probs, axis=1)
        return np.clip(confidence, 0.0, 1.0), {"method": "logits_softmax"}

    def extract_batch(self, x: Any) -> Dict[str, Any]:
        """Extract confidence scores for a batch of samples."""
        inferred = self._infer_task_type()

        if inferred == "classification":
            scores, metadata = self._classification_confidence(x)
        elif inferred == "regression":
            scores, metadata = self._regression_confidence(x)
        elif inferred in {"transformer", "llm", "text_generation", "sequence"}:
            scores, metadata = self._sequence_confidence(x)
        else:
            if hasattr(self.model, "predict_proba"):
                scores, metadata = self._classification_confidence(x)
            elif hasattr(self.model, "predict"):
                scores, metadata = self._regression_confidence(x)
            else:
                scores = np.zeros(len(np.asarray(x)) if np.asarray(x).ndim > 1 else 1, dtype=float)
                metadata = {"method": "unsupported_model"}

        scores = np.asarray(scores, dtype=float).reshape(-1)
        scores = np.array([_safe_clip(v) for v in scores], dtype=float)

        return {
            "scores": scores,
            "mean_score": float(scores.mean()) if len(scores) else 0.0,
            "metadata": metadata,
            "task_type": inferred,
        }

    def extract(self, x: Any) -> Dict[str, Any]:
        """Extract confidence for a single sample."""
        single = self._ensure_2d(x) if not isinstance(x, (str, bytes, dict, list)) else [x]
        batch_output = self.extract_batch(single)
        score = float(batch_output["scores"][0]) if len(batch_output["scores"]) else 0.0
        return {
            "confidence_score": _safe_clip(score),
            "metadata": batch_output["metadata"],
            "task_type": batch_output["task_type"],
        }