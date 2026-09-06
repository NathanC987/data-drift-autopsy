"""Remediation strategies, each driven by the drift diagnosis.

Every strategy returns a :class:`RemediationResult`. The label cost differs:
``full_retrain``/``feature_drop_retrain``/``importance_weighted_retrain`` and
``retrain_on_recent`` reuse labels the team already has (reference or delayed);
``head_refit``/``calibration_only`` need a small slice of fresh production
labels.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from drift_autopsy.remediation.base import (
    RemediationContext,
    RemediationResult,
    accuracy_on,
    flops_proxy_lr,
    flops_proxy_rf,
    timed_fit,
)
from drift_autopsy.remediation.importance_weighting import density_ratio_weights


def _result(ctx: RemediationContext, name: str, model, seconds: float, n_iter: int,
            n_rows: int, n_features: int, n_labels: int, **extra) -> RemediationResult:
    return RemediationResult(
        strategy=name,
        shift_name=ctx.shift_name,
        accuracy_before=ctx.baseline_before,
        accuracy_after=accuracy_on(model, _select(ctx.holdout_X, extra.get("_keep")), ctx.holdout_y),
        reference_accuracy=float(ctx.reference_accuracy),
        wall_clock_seconds=seconds,
        train_samples=n_rows,
        n_features_used=n_features,
        n_production_labels_required=n_labels,
        fit_flops_proxy=flops_proxy_lr(n_rows, n_features, n_iter) + extra.get("_extra_flops", 0.0),
        effective_sample_size=extra.get("effective_sample_size"),
        extra={k: v for k, v in extra.items() if not k.startswith("_")},
    )


def _select(X: np.ndarray, keep: Optional[np.ndarray]) -> np.ndarray:
    return X if keep is None else X[:, keep]


def full_retrain(ctx: RemediationContext) -> RemediationResult:
    """Retrain on reference + every recent window with (delayed) labels."""
    Xs = [ctx.reference_X] + [w[0] for w in ctx.recent_windows]
    ys = [ctx.reference_y] + [w[1] for w in ctx.recent_windows]
    X, y = np.vstack(Xs), np.concatenate(ys)
    model, sec, n_iter = timed_fit(ctx.model_factory(), X, y)
    return _result(ctx, "full_retrain", model, sec, n_iter, len(X), X.shape[1], 0)


def feature_drop_retrain(ctx: RemediationContext, drop_features=None) -> RemediationResult:
    """Drop the localised drifted features, then retrain on reference + recent."""
    drop = list(drop_features) if drop_features is not None else list(ctx.drifted_features)
    drop_idx = [ctx.feature_names.index(f) for f in drop if f in ctx.feature_names]
    keep = np.array([i for i in range(len(ctx.feature_names)) if i not in drop_idx], dtype=int)

    Xs = [ctx.reference_X] + [w[0] for w in ctx.recent_windows]
    ys = [ctx.reference_y] + [w[1] for w in ctx.recent_windows]
    X, y = np.vstack(Xs)[:, keep], np.concatenate(ys)
    model, sec, n_iter = timed_fit(ctx.model_factory(), X, y)
    return _result(
        ctx, "feature_drop_retrain", model, sec, n_iter, len(X), len(keep), 0,
        dropped_features=drop, _keep=keep,
    )


def importance_weighted_retrain(ctx: RemediationContext, clip: float = 20.0) -> RemediationResult:
    """Covariate-shift correction: weighted ERM on reference data, zero prod labels."""
    weights, diag = density_ratio_weights(ctx.reference_X, ctx.production_X, clip=clip)
    model, sec, n_iter = timed_fit(
        ctx.model_factory(), ctx.reference_X, ctx.reference_y, sample_weight=weights
    )
    rf_flops = flops_proxy_rf(diag["n_estimators"], len(ctx.reference_X) + len(ctx.production_X),
                              ctx.reference_X.shape[1])
    return _result(
        ctx, "importance_weighted_retrain", model, sec, n_iter,
        len(ctx.reference_X), ctx.reference_X.shape[1], 0,
        effective_sample_size=diag["effective_sample_size"],
        domain_classifier_auc=diag["domain_classifier_auc"],
        _extra_flops=rf_flops,
    )


def retrain_on_recent(ctx: RemediationContext, window: str = "previous") -> RemediationResult:
    """Retrain on reference + the most recent window(s). The cheap strategy for streams."""
    if not ctx.recent_windows:
        raise ValueError("retrain_on_recent needs ctx.recent_windows")
    picked = ctx.recent_windows[-1:] if window == "previous" else ctx.recent_windows
    Xs = [ctx.reference_X] + [w[0] for w in picked]
    ys = [ctx.reference_y] + [w[1] for w in picked]
    X, y = np.vstack(Xs), np.concatenate(ys)
    model, sec, n_iter = timed_fit(ctx.model_factory(), X, y)
    return _result(
        ctx, f"retrain_on_recent[{window}]", model, sec, n_iter, len(X), X.shape[1], 0,
        windows_used=len(picked),
    )


def head_refit(ctx: RemediationContext, n_recent_labels: int = 2000, seed: int = 42) -> RemediationResult:
    """Refit the model on reference + a small fresh slice of production labels."""
    if ctx.production_y is None:
        raise ValueError("head_refit needs ctx.production_y")
    rng = np.random.default_rng(seed)
    n = min(n_recent_labels, len(ctx.production_X))
    idx = rng.choice(len(ctx.production_X), n, replace=False)
    X = np.vstack([ctx.reference_X, ctx.production_X[idx]])
    y = np.concatenate([ctx.reference_y, ctx.production_y[idx]])
    model, sec, n_iter = timed_fit(ctx.model_factory(), X, y)
    return _result(ctx, "head_refit", model, sec, n_iter, len(X), X.shape[1], n)


def calibration_only(ctx: RemediationContext, n_recent_labels: int = 2000, seed: int = 42) -> RemediationResult:
    """Isotonic recalibration of the deployed model's score on a small labelled slice.

    For a binary linear model this barely moves argmax accuracy; it is reported
    to show that cheap label-free-ish fixes recover calibration, not accuracy.
    """
    if ctx.production_y is None:
        raise ValueError("calibration_only needs ctx.production_y")
    rng = np.random.default_rng(seed)
    n = min(n_recent_labels, len(ctx.production_X))
    idx = rng.choice(len(ctx.production_X), n, replace=False)

    import time as _t

    scores_cal = ctx.base_model.predict_proba(ctx.production_X[idx])[:, 1]
    start = _t.perf_counter()
    iso = IsotonicRegression(out_of_bounds="clip").fit(scores_cal, ctx.production_y[idx])
    seconds = _t.perf_counter() - start

    scores_hold = ctx.base_model.predict_proba(ctx.holdout_X)[:, 1]
    preds = (iso.predict(scores_hold) >= 0.5).astype(int)
    after = float(np.mean(preds == ctx.holdout_y))

    return RemediationResult(
        strategy="calibration_only",
        shift_name=ctx.shift_name,
        accuracy_before=ctx.baseline_before,
        accuracy_after=after,
        reference_accuracy=float(ctx.reference_accuracy),
        wall_clock_seconds=seconds,
        train_samples=n,
        n_features_used=1,
        n_production_labels_required=n,
        fit_flops_proxy=float(n),
        extra={"note": "recovers calibration, not accuracy"},
    )
