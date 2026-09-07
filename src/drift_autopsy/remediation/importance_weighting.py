"""Density-ratio sample weights for covariate-shift adaptation.

A classifier is trained to tell reference rows from production rows; its
predicted odds give w(x) = p(prod | x) / p(ref | x), up to the base-rate
constant. Training the deployed model on the reference data with these weights
is the classic covariate-shift correction (Shimodaira 2000; Sugiyama 2007) and
needs no production labels. Probabilities are cross-fitted so a row is never
weighted by a model that trained on it.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def density_ratio_weights(
    X_ref: np.ndarray,
    X_prod: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 5,
    n_splits: int = 5,
    clip: float = 20.0,
    max_rows_for_fit: int = 5000,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return per-reference-row weights (mean 1) and diagnostics.

    When the stacked data is large the domain classifier is cross-fitted on a
    balanced subsample and its probabilities applied to every reference row,
    which keeps the estimate stable but bounds the cost on wide embeddings.
    """
    X_ref = np.asarray(X_ref, dtype=float)
    X_prod = np.asarray(X_prod, dtype=float)
    n_ref, n_prod = len(X_ref), len(X_prod)

    X = np.vstack([X_ref, X_prod])
    y = np.r_[np.zeros(n_ref), np.ones(n_prod)]

    rng = np.random.RandomState(random_state)
    fit_cap = max(1000, min(len(X), max_rows_for_fit))
    fit_idx = np.arange(len(X)) if len(X) <= fit_cap else rng.choice(len(X), fit_cap, replace=False)

    oof_fit = np.zeros(len(fit_idx))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    Xf, yf = X[fit_idx], y[fit_idx]
    for train_idx, test_idx in skf.split(Xf, yf):
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1,
        )
        clf.fit(Xf[train_idx], yf[train_idx])
        oof_fit[test_idx] = clf.predict_proba(Xf[test_idx])[:, 1]

    if len(fit_idx) == len(X):
        oof = oof_fit
    else:
        final = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1,
        ).fit(Xf, yf)
        oof = final.predict_proba(X)[:, 1]
        oof[fit_idx] = oof_fit  # keep the honest out-of-fold values where we have them

    auc = _auc(yf, oof_fit)
    p_ref = np.clip(oof[:n_ref], 1e-3, 1 - 1e-3)
    weights = (p_ref / (1.0 - p_ref)) * (n_ref / n_prod)
    weights = np.clip(weights, 0.0, clip)
    weights = weights / weights.mean()

    ess = float(weights.sum() ** 2 / np.sum(weights ** 2))
    return weights, {
        "domain_classifier_auc": float(auc),
        "effective_sample_size": ess,
        "effective_sample_fraction": ess / n_ref,
        "n_estimators": n_estimators,
        "clip": clip,
    }


def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return 0.5
