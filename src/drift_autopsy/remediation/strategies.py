"""Remediation strategies.

Label cost differs, and each result reports it:

* ``importance_weighted_retrain`` - reference data only, reweighted toward
  production (no labels).
* ``feature_drop_retrain``       - reference (+ delayed windows) without the
  localised features (no fresh labels).
* ``retrain_on_recent``          - reference + the most recent delayed-label
  window (labels the team already receives with a delay).
* ``full_retrain``               - reference + every delayed-label window.
* ``label_retrain``              - reference + a fresh labelled production
  sample of ``n_labels`` rows (buys labels).
* ``recalibrate``                - isotonic recalibration of the deployed
  binary model on ``n_labels`` fresh production labels (buys labels).

A strategy that needs data the context lacks (no delayed windows, no
production labels, a non-binary model for recalibration) raises
``NotApplicable`` and is skipped, rather than silently returning the deployed
model.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from drift_autopsy.remediation.base import (
    RemediationContext,
    RemediationResult,
    fit_model,
    flops_proxy_forest,
    grade,
)
from drift_autopsy.remediation.importance_weighting import density_ratio_weights


class NotApplicable(Exception):
    """The strategy needs data the context does not have."""


def importance_weighted_retrain(ctx: RemediationContext) -> RemediationResult:
    weights, diag = density_ratio_weights(ctx.reference_X, ctx.production_X, random_state=ctx.seed)
    model, sec, n_iter = fit_model(ctx.model_factory, ctx.reference_X, ctx.reference_y, sample_weight=weights)
    rf = flops_proxy_forest(diag["n_estimators"] * diag["n_splits"], diag["n_rows_fit"], ctx.reference_X.shape[1])
    return grade(ctx, "importance_weighted_retrain", model, sec + diag["seconds"], n_iter,
                 len(ctx.reference_X), ctx.reference_X.shape[1], 0, extra_flops=rf,
                 domain_classifier_auc=diag["domain_classifier_auc"],
                 effective_sample_fraction=diag["effective_sample_fraction"])


def feature_drop_retrain(ctx: RemediationContext) -> RemediationResult:
    if not ctx.drifted_features:
        raise NotApplicable("no localised features to drop")
    drop = {ctx.feature_names.index(f) for f in ctx.drifted_features}
    keep = np.array([i for i in range(len(ctx.feature_names)) if i not in drop], dtype=int)
    if keep.size == 0:
        raise NotApplicable("every feature is localised")
    X = np.vstack([ctx.reference_X] + [w[0] for w in ctx.recent_windows])[:, keep]
    y = np.concatenate([ctx.reference_y] + [w[1] for w in ctx.recent_windows])
    factory = _column_subset_factory(ctx, keep)
    model, sec, n_iter = fit_model(factory, X, y)
    return grade(ctx, "feature_drop_retrain", model, sec, n_iter, len(X), len(keep), 0, keep=keep,
                 dropped_features=list(ctx.drifted_features))


def retrain_on_recent(ctx: RemediationContext) -> RemediationResult:
    if not ctx.recent_windows:
        raise NotApplicable("no delayed-label window")
    X_new, y_new = ctx.recent_windows[-1]
    X = np.vstack([ctx.reference_X, X_new])
    y = np.concatenate([ctx.reference_y, y_new])
    model, sec, n_iter = fit_model(ctx.model_factory, X, y)
    return grade(ctx, "retrain_on_recent", model, sec, n_iter, len(X), X.shape[1], 0)


def full_retrain(ctx: RemediationContext) -> RemediationResult:
    if len(ctx.recent_windows) < 2:
        raise NotApplicable("needs two or more delayed-label windows (else it equals retrain_on_recent)")
    X = np.vstack([ctx.reference_X] + [w[0] for w in ctx.recent_windows])
    y = np.concatenate([ctx.reference_y] + [w[1] for w in ctx.recent_windows])
    model, sec, n_iter = fit_model(ctx.model_factory, X, y)
    return grade(ctx, "full_retrain", model, sec, n_iter, len(X), X.shape[1], 0,
                 windows_used=len(ctx.recent_windows))


def label_retrain(ctx: RemediationContext, n_labels: int = 2000) -> RemediationResult:
    if ctx.production_y is None:
        raise NotApplicable("no production labels")
    idx = _label_sample(ctx, n_labels)
    X = np.vstack([ctx.reference_X, ctx.production_X[idx]])
    y = np.concatenate([ctx.reference_y, ctx.production_y[idx]])
    model, sec, n_iter = fit_model(ctx.model_factory, X, y)
    return grade(ctx, "label_retrain", model, sec, n_iter, len(X), X.shape[1], len(idx))


def recalibrate(ctx: RemediationContext, n_labels: int = 2000) -> RemediationResult:
    if ctx.production_y is None:
        raise NotApplicable("no production labels")
    if len(getattr(ctx.base_model, "classes_", [])) != 2:
        raise NotApplicable("isotonic recalibration here is for binary models")
    idx = _label_sample(ctx, n_labels)
    scores = ctx.base_model.predict_proba(ctx.production_X[idx])[:, 1]
    start = time.perf_counter()
    iso = IsotonicRegression(out_of_bounds="clip").fit(scores, ctx.production_y[idx])
    seconds = time.perf_counter() - start
    model = _Recalibrated(ctx.base_model, iso)
    return grade(ctx, "recalibrate", model, seconds, 1, len(idx), 1, len(idx))


# --------------------------------------------------------------------------- #

def _label_sample(ctx: RemediationContext, n_labels: int) -> np.ndarray:
    rng = np.random.default_rng(ctx.seed)
    return rng.choice(len(ctx.production_X), min(n_labels, len(ctx.production_X)), replace=False)


class _Recalibrated:
    """Deployed binary model with an isotonic map on its positive-class score."""

    def __init__(self, base, iso):
        self.base, self.iso = base, iso
        self.classes_ = base.classes_

    def predict(self, X):
        p = self.iso.predict(self.base.predict_proba(X)[:, 1])
        return np.where(p >= 0.5, self.classes_[1], self.classes_[0])


def _column_subset_factory(ctx: RemediationContext, keep: np.ndarray):
    """Factory for a model on a column subset. Uses ``ctx.model_factory`` when it
    accepts a ``keep_columns`` argument, else the factory as is."""
    def factory():
        try:
            return ctx.model_factory(keep_columns=keep)
        except TypeError:
            return ctx.model_factory()
    return factory


STRATEGIES = {
    "importance_weighted_retrain": importance_weighted_retrain,
    "feature_drop_retrain": feature_drop_retrain,
    "retrain_on_recent": retrain_on_recent,
    "full_retrain": full_retrain,
    "label_retrain": label_retrain,
    "recalibrate": recalibrate,
}
