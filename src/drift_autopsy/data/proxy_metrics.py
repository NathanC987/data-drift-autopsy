"""Proxy performance estimation for multiclass embedding pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass
class ProxyMetrics:
    """Estimated and actual metrics for one bucket."""

    estimated: Dict[str, float]
    actual: Dict[str, float | None]
    class_wise_estimated: Dict[str, Dict[str, float]] = field(default_factory=dict)
    class_wise_actual: Dict[str, Dict[str, float]] = field(default_factory=dict)
    proxy_quality_gap: Dict[str, float] = field(default_factory=dict)
    class_wise_proxy_quality_gap: Dict[str, Dict[str, float]] = field(default_factory=dict)


class MulticlassProxyEstimator:
    """Confidence-calibration proxy estimator for multiclass predictions."""

    def __init__(self, n_bins: int = 10):
        if n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        self.n_bins = n_bins
        self.bin_edges: np.ndarray | None = None
        self.bin_accuracy: np.ndarray | None = None
        self.class_count: int | None = None

    @staticmethod
    def _proba_columns(df: pd.DataFrame, prefix: str = "pred_proba_") -> List[str]:
        cols = sorted([c for c in df.columns if c.startswith(prefix)], key=lambda x: int(x.split("_")[-1]))
        if not cols:
            raise ValueError("No pred_proba_* columns found")
        return cols

    @staticmethod
    def _actual_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def fit(self, reference_df: pd.DataFrame) -> None:
        """Fit confidence->correctness calibration on reference bucket."""
        if "y_true" not in reference_df.columns or "y_pred" not in reference_df.columns:
            raise ValueError("reference_df must include y_true and y_pred")

        proba_cols = self._proba_columns(reference_df)
        self.class_count = len(proba_cols)

        proba = reference_df[proba_cols].to_numpy(dtype=float)
        confidence = np.max(proba, axis=1)
        correct = (reference_df["y_true"].to_numpy(dtype=int) == reference_df["y_pred"].to_numpy(dtype=int)).astype(float)

        self.bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_ids = np.clip(np.digitize(confidence, self.bin_edges[1:-1], right=False), 0, self.n_bins - 1)

        bin_acc = np.zeros(self.n_bins, dtype=float)
        global_acc = float(correct.mean()) if len(correct) else 0.0

        for bin_idx in range(self.n_bins):
            mask = bin_ids == bin_idx
            if np.any(mask):
                bin_acc[bin_idx] = float(correct[mask].mean())
            else:
                bin_acc[bin_idx] = global_acc

        self.bin_accuracy = bin_acc

    def estimate(self, bucket_df: pd.DataFrame) -> ProxyMetrics:
        """Estimate metrics for a bucket and return estimated-vs-actual pair."""
        if self.bin_edges is None or self.bin_accuracy is None or self.class_count is None:
            raise RuntimeError("Estimator must be fitted before estimate()")

        if "y_pred" not in bucket_df.columns:
            raise ValueError("bucket_df must include y_pred")

        proba_cols = self._proba_columns(bucket_df)
        if len(proba_cols) != self.class_count:
            raise ValueError("Probability column count mismatch between fit and estimate")

        proba = bucket_df[proba_cols].to_numpy(dtype=float)
        y_pred = bucket_df["y_pred"].to_numpy(dtype=int)
        confidence = np.max(proba, axis=1)
        bin_ids = np.clip(np.digitize(confidence, self.bin_edges[1:-1], right=False), 0, self.n_bins - 1)
        q = self.bin_accuracy[bin_ids]

        # Estimated TP by class uses calibrated correctness per predicted class.
        pred_counts = np.bincount(y_pred, minlength=self.class_count).astype(float)
        tp_est = np.bincount(y_pred, weights=q, minlength=self.class_count).astype(float)

        # Approximate true class support via soft probabilities.
        support_est = proba.sum(axis=0)

        precision_est = np.divide(
            tp_est,
            pred_counts,
            out=np.zeros_like(tp_est),
            where=pred_counts > 0,
        )
        recall_est = np.divide(
            tp_est,
            support_est,
            out=np.zeros_like(tp_est),
            where=support_est > 0,
        )
        f1_est = np.divide(
            2 * precision_est * recall_est,
            precision_est + recall_est,
            out=np.zeros_like(precision_est),
            where=(precision_est + recall_est) > 0,
        )

        estimated = {
            "accuracy": float(np.mean(q)) if len(q) else 0.0,
            "precision": float(np.mean(precision_est)),
            "recall": float(np.mean(recall_est)),
            "f1": float(np.mean(f1_est)),
        }

        class_wise_estimated = {}
        for class_idx in range(self.class_count):
            class_wise_estimated[f"class_{class_idx}"] = {
                "precision": float(precision_est[class_idx]),
                "recall": float(recall_est[class_idx]),
                "f1": float(f1_est[class_idx]),
                "support": float(support_est[class_idx]),
            }

        has_actual_labels = (
            "y_true" in bucket_df.columns
            and bucket_df["y_true"].notna().any()
        )

        actual: Dict[str, float | None] = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
        class_wise_actual: Dict[str, Dict[str, float]] = {}
        proxy_quality_gap: Dict[str, float] = {}
        class_wise_proxy_quality_gap: Dict[str, Dict[str, float]] = {}

        if has_actual_labels:
            valid_mask = bucket_df["y_true"].notna().to_numpy()
            y_true = bucket_df.loc[valid_mask, "y_true"].to_numpy(dtype=int)
            y_pred_valid = y_pred[valid_mask]

            actual = self._actual_metrics(y_true=y_true, y_pred=y_pred_valid)
            proxy_quality_gap = {
                metric_name: float(float(actual[metric_name]) - estimated[metric_name])
                for metric_name in ("accuracy", "precision", "recall", "f1")
            }

            precision_cls, recall_cls, f1_cls, support_cls = precision_recall_fscore_support(
                y_true,
                y_pred_valid,
                labels=list(range(self.class_count)),
                average=None,
                zero_division=0,
            )

            for class_idx in range(self.class_count):
                class_key = f"class_{class_idx}"
                class_wise_actual[class_key] = {
                    "precision": float(precision_cls[class_idx]),
                    "recall": float(recall_cls[class_idx]),
                    "f1": float(f1_cls[class_idx]),
                    "support": float(support_cls[class_idx]),
                }
                class_wise_proxy_quality_gap[class_key] = {
                    "precision": float(precision_cls[class_idx] - precision_est[class_idx]),
                    "recall": float(recall_cls[class_idx] - recall_est[class_idx]),
                    "f1": float(f1_cls[class_idx] - f1_est[class_idx]),
                }

        return ProxyMetrics(
            estimated=estimated,
            actual=actual,
            class_wise_estimated=class_wise_estimated,
            class_wise_actual=class_wise_actual,
            proxy_quality_gap=proxy_quality_gap,
            class_wise_proxy_quality_gap=class_wise_proxy_quality_gap,
        )
