"""Baseline classifier utilities for embedding-derived image pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class EmbeddingBaselineClassifier:
    """Train and apply a baseline classifier on embedding feature columns."""

    def __init__(self, model_name: str = "logistic_regression", model_params: Dict[str, Any] | None = None):
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = self._build_model(model_name, self.model_params)
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _feature_columns(df: pd.DataFrame, prefix: str = "feature_") -> List[str]:
        cols = [col for col in df.columns if col.startswith(prefix)]
        if not cols:
            raise ValueError("No embedding columns found with prefix 'feature_'")
        return cols

    def _build_model(self, model_name: str, model_params: Dict[str, Any]):
        if model_name != "logistic_regression":
            raise ValueError(
                f"Unsupported baseline model '{model_name}'. Supported: logistic_regression"
            )

        params = {
            "max_iter": 1000,
            "random_state": 42,
        }
        params.update(model_params)
        return LogisticRegression(**params)

    def fit(self, reference_df: pd.DataFrame) -> None:
        """Fit baseline model on a full reference bucket dataframe."""
        if "y_true" not in reference_df.columns:
            raise ValueError("reference_df must include y_true")

        feature_cols = self._feature_columns(reference_df)
        x = reference_df[feature_cols].to_numpy(dtype=float)
        y = reference_df["y_true"].to_numpy(dtype=int)

        self.model.fit(x, y)
        self.classes_ = np.array(self.model.classes_, dtype=int)

    def fit_with_split(
        self,
        reference_df: pd.DataFrame,
        train_fraction: float = 0.7,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """Fit baseline model with train/eval split and return baseline metrics."""
        if "y_true" not in reference_df.columns:
            raise ValueError("reference_df must include y_true")

        feature_cols = self._feature_columns(reference_df)
        x = reference_df[feature_cols].to_numpy(dtype=float)
        y = reference_df["y_true"].to_numpy(dtype=int)

        can_stratify = len(np.unique(y)) > 1 and np.min(np.bincount(y)) > 1

        if len(reference_df) < 4 or not can_stratify:
            # Fallback for very small or degenerate class distributions.
            logger.warning(
                "Skipping train/eval split due to low samples or class support; fitting on full reference"
            )
            self.model.fit(x, y)
            self.classes_ = np.array(self.model.classes_, dtype=int)
            preds = self.model.predict(x)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y,
                preds,
                average="macro",
                zero_division=0,
            )
            return {
                "train_samples": float(len(reference_df)),
                "test_samples": float(len(reference_df)),
                "accuracy": float(accuracy_score(y, preds)),
                "precision_macro": float(precision),
                "recall_macro": float(recall),
                "f1_macro": float(f1),
            }

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=train_fraction,
            random_state=random_state,
            stratify=y,
        )

        self.model.fit(x_train, y_train)
        self.classes_ = np.array(self.model.classes_, dtype=int)

        preds = self.model.predict(x_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            preds,
            average="macro",
            zero_division=0,
        )

        return {
            "train_samples": float(len(x_train)),
            "test_samples": float(len(x_test)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        }

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class labels and probabilities from embedding dataframe."""
        if self.classes_ is None:
            raise RuntimeError("Baseline model is not fitted")

        feature_cols = self._feature_columns(df)
        x = df[feature_cols].to_numpy(dtype=float)

        y_pred = self.model.predict(x).astype(int)
        proba = self.model.predict_proba(x).astype(float)
        return y_pred, proba

    def attach_predictions(self, df: pd.DataFrame, class_count: int) -> pd.DataFrame:
        """Return a new dataframe with y_pred and pred_proba_* columns."""
        y_pred, proba = self.predict(df)

        out = df.copy()
        out["y_pred"] = y_pred

        # Ensure stable full class-space probabilities for downstream contract.
        full_proba = np.zeros((len(out), class_count), dtype=float)
        for model_idx, class_id in enumerate(self.classes_):
            if 0 <= int(class_id) < class_count:
                full_proba[:, int(class_id)] = proba[:, model_idx]

        row_sums = full_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        full_proba = full_proba / row_sums

        for class_idx in range(class_count):
            out[f"pred_proba_{class_idx}"] = full_proba[:, class_idx]

        return out

    def save(self, path: str) -> None:
        """Persist baseline model and metadata to disk."""
        payload = {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "classes": self.classes_,
            "model": self.model,
        }
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str) -> "EmbeddingBaselineClassifier":
        """Load baseline model from disk."""
        with open(path, "rb") as handle:
            payload = pickle.load(handle)

        instance = cls(
            model_name=payload["model_name"],
            model_params=payload["model_params"],
        )
        instance.classes_ = payload["classes"]
        instance.model = payload["model"]
        return instance


def create_monitored_model(
    model_name: str = "logistic_regression",
    model_params: Dict[str, Any] | None = None,
) -> EmbeddingBaselineClassifier:
    """Create a monitored-model adapter for prediction generation.

    This factory keeps current behavior while making model-role separation explicit:
    - System extractor generates embedding feature columns.
    - Monitored model generates y_pred and pred_proba_* columns.
    """
    return EmbeddingBaselineClassifier(model_name=model_name, model_params=model_params)
