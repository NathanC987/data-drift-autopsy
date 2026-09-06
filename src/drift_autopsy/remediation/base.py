"""Shared types and helpers for localisation-driven remediation.

A remediation strategy takes a :class:`RemediationContext` (the data splits plus
the drift diagnosis) and returns a :class:`RemediationResult` describing how much
accuracy it recovered and what it cost. Cost is reported as a vector -- wall
clock, training rows, a hardware-independent FLOP proxy, and the number of
production labels required -- because "cheap" means different things depending
on whether labels or compute are the bottleneck.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelFactory = Callable[[], Any]


def default_model_factory() -> Pipeline:
    """A standardised logistic regression -- the paper's monitored model."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


@dataclass
class RemediationContext:
    """Everything a strategy needs, plus the drift diagnosis that drives it."""

    reference_X: np.ndarray
    reference_y: np.ndarray
    production_X: np.ndarray          # unlabelled production sample (adaptation only)
    holdout_X: np.ndarray             # held-out target partition, no strategy sees this
    holdout_y: np.ndarray
    feature_names: List[str]
    drifted_features: List[str]       # from localisation
    shift_name: str
    base_model: Any                   # the deployed, un-remediated model
    recent_windows: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    production_y: Optional[np.ndarray] = None  # only for label-using strategies
    model_factory: ModelFactory = default_model_factory
    reference_accuracy: Optional[float] = None  # ceiling: base model on held-out reference

    def __post_init__(self) -> None:
        self.feature_names = list(self.feature_names)
        self.drifted_features = [f for f in self.drifted_features if f in self.feature_names]

    @property
    def drifted_index(self) -> np.ndarray:
        idx = [self.feature_names.index(f) for f in self.drifted_features]
        return np.asarray(idx, dtype=int)

    @property
    def baseline_before(self) -> float:
        return float(accuracy_score(self.holdout_y, self.base_model.predict(self.holdout_X)))


@dataclass
class RemediationResult:
    strategy: str
    shift_name: str
    accuracy_before: float
    accuracy_after: float
    reference_accuracy: float
    wall_clock_seconds: float
    train_samples: int
    n_features_used: int
    n_production_labels_required: int
    fit_flops_proxy: float
    effective_sample_size: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy_recovered(self) -> float:
        return self.accuracy_after - self.accuracy_before

    @property
    def fraction_of_gap_recovered(self) -> float:
        gap = self.reference_accuracy - self.accuracy_before
        return float(self.accuracy_recovered / gap) if abs(gap) > 1e-9 else float("nan")

    @property
    def recovery_per_second(self) -> float:
        return float(self.accuracy_recovered / max(self.wall_clock_seconds, 1e-6))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "shift_name": self.shift_name,
            "accuracy_before": round(self.accuracy_before, 5),
            "accuracy_after": round(self.accuracy_after, 5),
            "accuracy_recovered": round(self.accuracy_recovered, 5),
            "fraction_of_gap_recovered": round(self.fraction_of_gap_recovered, 4),
            "reference_accuracy": round(self.reference_accuracy, 5),
            "wall_clock_seconds": round(self.wall_clock_seconds, 4),
            "train_samples": int(self.train_samples),
            "n_features_used": int(self.n_features_used),
            "n_production_labels_required": int(self.n_production_labels_required),
            "fit_flops_proxy": float(self.fit_flops_proxy),
            "effective_sample_size": (
                None if self.effective_sample_size is None else round(self.effective_sample_size, 1)
            ),
            "recovery_per_second": round(self.recovery_per_second, 6),
            "extra": self.extra,
        }


def timed_fit(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    repeats: int = 3,
) -> Tuple[Any, float, int]:
    """Fit ``estimator`` ``repeats`` times, return (last fit, median seconds, n_iter)."""
    times: List[float] = []
    fitted = estimator
    for _ in range(repeats):
        fitted = _clone_like(estimator)
        start = time.perf_counter()
        if sample_weight is not None:
            fitted.fit(X, y, **_weight_kwarg(fitted, sample_weight))
        else:
            fitted.fit(X, y)
        times.append(time.perf_counter() - start)
    return fitted, float(np.median(times)), _n_iter(fitted)


def _clone_like(estimator: Any) -> Any:
    from sklearn.base import clone

    return clone(estimator)


def _weight_kwarg(estimator: Any, weights: np.ndarray) -> Dict[str, np.ndarray]:
    if isinstance(estimator, Pipeline):
        return {f"{estimator.steps[-1][0]}__sample_weight": weights}
    return {"sample_weight": weights}


def _n_iter(estimator: Any) -> int:
    clf = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    n_iter = getattr(clf, "n_iter_", None)
    if n_iter is None:
        return 1
    return int(np.max(np.asarray(n_iter)))


def flops_proxy_lr(n_rows: int, n_features: int, n_iter: int) -> float:
    """Rough cost of fitting a linear model: iterations x rows x features."""
    return float(max(n_iter, 1) * n_rows * n_features)


def flops_proxy_rf(n_trees: int, n_rows: int, n_features: int) -> float:
    """Rough cost of fitting a random forest for the density-ratio weights."""
    if n_rows <= 1:
        return 0.0
    return float(n_trees * n_rows * np.log2(n_rows) * np.sqrt(max(n_features, 1)))


def accuracy_on(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    return float(accuracy_score(y, model.predict(X)))
