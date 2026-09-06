"""One controlled covariate shift, for checking that the machinery works.

The real ACS shifts turn out to be largely irreducible, so we also inject a
shift whose cause is known and purely covariate: a fixed scale and mean offset
applied to a named subset of features, with the label rule untouched. Here
feature-drop and importance weighting *should* recover the gap; if they do, the
failure on the real shifts is about the shift, not the method.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def inject_covariate_shift(
    X: np.ndarray,
    feature_names: Sequence[str],
    shifted_features: Sequence[str],
    scale: float = 1.6,
    offset: float = 1.0,
) -> np.ndarray:
    """Return a copy of ``X`` with the named features linearly transformed.

    ``x' = offset * sd(x) + scale * (x - mean(x)) + mean(x)`` per feature, so the
    marginal spreads and means move but the feature stays on its original axis.
    """
    X = np.asarray(X, dtype=float).copy()
    names = list(feature_names)
    idx = [names.index(f) for f in shifted_features if f in names]
    for j in idx:
        col = X[:, j]
        mu, sd = col.mean(), col.std() + 1e-9
        X[:, j] = mu + scale * (col - mu) + offset * sd
    return X


def shifted_feature_list(feature_names: Sequence[str], k: int = 2) -> List[str]:
    """Default choice: the first ``k`` names (kept explicit for the paper)."""
    return list(feature_names)[:k]
