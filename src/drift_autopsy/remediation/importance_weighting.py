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
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return per-reference-row weights (mean 1) and diagnostics."""
    X_ref = np.asarray(X_ref, dtype=float)
    X_prod = np.asarray(X_prod, dtype=float)
    n_ref, n_prod = len(X_ref), len(X_prod)

    X = np.vstack([X_ref, X_prod])
    y = np.r_[np.zeros(n_ref), np.ones(n_prod)]

    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, test_idx in skf.split(X, y):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        oof[test_idx] = clf.predict_proba(X[test_idx])[:, 1]

    auc = _auc(y, oof)
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
