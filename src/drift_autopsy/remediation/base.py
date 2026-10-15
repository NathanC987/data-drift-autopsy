"""Shared types and helpers for remediation.

A remediation strategy takes a :class:`RemediationContext` (the data the team
has, plus the drift diagnosis) and returns a :class:`RemediationResult`: the
accuracy it reaches on a labelled production holdout that no strategy trains
on, a paired-bootstrap test of the recovery against the deployed model, and a
cost vector (wall clock, training rows, a FLOP proxy and the number of fresh
production labels it needs).

Every fit runs with BLAS limited to one thread, so results are bit-for-bit
reproducible and timings are comparable across strategies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from threadpoolctl import threadpool_limits

from drift_autopsy.stats import paired_accuracy_difference

ModelFactory = Callable[[], Any]


@dataclass
class RemediationContext:
    """Everything a strategy may use. ``holdout_*`` is for grading only."""

    reference_X: np.ndarray            # labelled data the deployed model was trained on
    reference_y: np.ndarray
    production_X: np.ndarray           # unlabelled production sample
    holdout_X: np.ndarray              # labelled production holdout (grading only)
    holdout_y: np.ndarray
    feature_names: List[str]
    drifted_features: List[str]        # from localisation
    shift_name: str
    base_model: Any                    # the deployed model
    model_factory: ModelFactory        # builds an unfitted model of the deployed kind
    reference_accuracy: float          # deployed model on held-out reference data
    recent_windows: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)  # delayed-label windows, oldest first
    production_y: Optional[np.ndarray] = None   # only for strategies that buy fresh labels
    seed: int = 42

    def __post_init__(self) -> None:
        self.feature_names = list(self.feature_names)
        self.drifted_features = [f for f in self.drifted_features if f in self.feature_names]
        self._holdout_base_correct: Optional[np.ndarray] = None

    @property
    def holdout_base_correct(self) -> np.ndarray:
        if self._holdout_base_correct is None:
            self._holdout_base_correct = (self.base_model.predict(self.holdout_X) == self.holdout_y).astype(int)
        return self._holdout_base_correct

    @property
    def baseline_before(self) -> float:
        return float(self.holdout_base_correct.mean())


@dataclass
class RemediationResult:
    strategy: str
    shift_name: str
    accuracy_before: float
    accuracy_after: float
    reference_accuracy: float
    recovery_test: Dict[str, float]
    wall_clock_seconds: float
    train_samples: int
    n_features_used: int
    n_production_labels_required: int
    fit_flops_proxy: float
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy_recovered(self) -> float:
        return self.accuracy_after - self.accuracy_before

    @property
    def gap(self) -> float:
        return self.reference_accuracy - self.accuracy_before

    @property
    def fraction_of_gap_recovered(self) -> Optional[float]:
        """Recovered / gap; undefined (None) when there is no significant gap to recover."""
        return float(self.accuracy_recovered / self.gap) if self.gap > 0 else None

    @property
    def recovery_significant(self) -> bool:
        return bool(self.recovery_test["ci_low"] > 0)

    def to_dict(self) -> Dict[str, Any]:
        frac = self.fraction_of_gap_recovered
        return {
            "strategy": self.strategy,
            "shift_name": self.shift_name,
            "accuracy_before": self.accuracy_before,
            "accuracy_after": self.accuracy_after,
            "accuracy_recovered": self.accuracy_recovered,
            "recovery_test": self.recovery_test,
            "recovery_significant": self.recovery_significant,
            "gap": self.gap,
            "fraction_of_gap_recovered": frac,
            "reference_accuracy": self.reference_accuracy,
            "wall_clock_seconds": self.wall_clock_seconds,
            "train_samples": int(self.train_samples),
            "n_features_used": int(self.n_features_used),
            "n_production_labels_required": int(self.n_production_labels_required),
            "fit_flops_proxy": float(self.fit_flops_proxy),
            "extra": self.extra,
        }


def fit_model(factory: ModelFactory, X: np.ndarray, y: np.ndarray,
              sample_weight: Optional[np.ndarray] = None, repeats: int = 3) -> Tuple[Any, float, int]:
    """Fit ``repeats`` times on one BLAS thread; return (model, median seconds, iterations)."""
    times: List[float] = []
    model = None
    with threadpool_limits(limits=1):
        for _ in range(repeats):
            model = factory()
            start = time.perf_counter()
            if sample_weight is None:
                model.fit(X, y)
            else:
                model.fit(X, y, **_weight_kwarg(model, sample_weight))
            times.append(time.perf_counter() - start)
    return model, float(np.median(times)), _n_iter(model)


def _weight_kwarg(model: Any, weights: np.ndarray) -> Dict[str, np.ndarray]:
    if hasattr(model, "steps"):
        return {f"{model.steps[-1][0]}__sample_weight": weights}
    return {"sample_weight": weights}


def _n_iter(model: Any) -> int:
    """Optimiser iterations (logistic regression) or trees grown (boosting, one per class per iteration)."""
    clf = model.steps[-1][1] if hasattr(model, "steps") else model
    n_iter = getattr(clf, "n_iter_", None)
    if n_iter is None:
        return 1
    return int(np.max(np.asarray(n_iter))) * int(getattr(clf, "n_trees_per_iteration_", 1))


def flops_proxy_linear(n_rows: int, n_features: int, n_iter: int) -> float:
    """Rough fitting cost: iterations (or boosted trees) x rows x features. Each
    optimiser iteration and each histogram tree touches every row and feature once."""
    return float(max(n_iter, 1) * n_rows * n_features)


def flops_proxy_forest(n_trees: int, n_rows: int, n_features: int) -> float:
    """Rough cost of fitting a random forest (density-ratio classifier)."""
    if n_rows <= 1:
        return 0.0
    return float(n_trees * n_rows * np.log2(n_rows) * np.sqrt(max(n_features, 1)))


def grade(ctx: RemediationContext, name: str, model: Any, seconds: float, n_iter: int,
          n_rows: int, n_features: int, n_labels: int, keep: Optional[np.ndarray] = None,
          extra_flops: float = 0.0, **extra: Any) -> RemediationResult:
    """Score a remediated model on the production holdout against the deployed model."""
    X_hold = ctx.holdout_X if keep is None else ctx.holdout_X[:, keep]
    after_correct = (model.predict(X_hold) == ctx.holdout_y).astype(int)
    test = paired_accuracy_difference(ctx.holdout_base_correct, after_correct, random_state=ctx.seed)
    return RemediationResult(
        strategy=name,
        shift_name=ctx.shift_name,
        accuracy_before=ctx.baseline_before,
        accuracy_after=float(after_correct.mean()),
        reference_accuracy=float(ctx.reference_accuracy),
        recovery_test=test,
        wall_clock_seconds=seconds,
        train_samples=n_rows,
        n_features_used=n_features,
        n_production_labels_required=n_labels,
        fit_flops_proxy=flops_proxy_linear(n_rows, n_features, n_iter) + extra_flops,
        extra=extra,
    )
