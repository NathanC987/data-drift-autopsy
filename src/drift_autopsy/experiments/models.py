"""The two monitored model families shared by every study.

``lr``  - L2-regularised logistic regression (C = 1) on standardised numeric
          inputs and one-hot nominal codes. This is the primary model.
``hgb`` - scikit-learn's histogram gradient boosting with its default settings,
          no early stopping (so a fit is deterministic given the seed and uses
          every row) and native categorical splits for nominal codes.

Nothing in either family is tuned on any evaluation window. The last pipeline
step is always named ``classifier`` so sample weights can be routed to it.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_KINDS = ("lr", "hgb")
MODEL_DESCRIPTIONS = {
    "lr": "L2 logistic regression (C=1); standardised numeric, one-hot nominal",
    "hgb": "histogram gradient boosting (scikit-learn defaults, no early stopping); native categorical splits",
}

# HistGradientBoosting bins every feature into at most 255 bins, so a nominal
# feature can carry at most 255 categories (codes 0..254).
HGB_MAX_CATEGORIES = 255


class CategoryCodes(BaseEstimator, TransformerMixin):
    """Map nominal codes to 0..k-1 for native categorical splits.

    Each nominal column keeps its ``max_categories - 1`` most frequent training
    codes (ties broken by code); every other code, including codes never seen
    in training, maps to one shared "other" level. Numeric columns pass
    through unchanged. Output columns are numeric first, then nominal.
    """

    def __init__(self, numeric_idx: Sequence[int], categorical_idx: Sequence[int],
                 max_categories: int = HGB_MAX_CATEGORIES):
        self.numeric_idx = numeric_idx
        self.categorical_idx = categorical_idx
        self.max_categories = max_categories

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.kept_: List[np.ndarray] = []
        for j in self.categorical_idx:
            codes, counts = np.unique(X[:, j], return_counts=True)
            order = np.lexsort((codes, -counts))[: self.max_categories - 1]
            self.kept_.append(np.sort(codes[order]))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = [X[:, list(self.numeric_idx)]]
        for j, kept in zip(self.categorical_idx, self.kept_):
            values = X[:, j]
            pos = np.clip(np.searchsorted(kept, values), 0, len(kept) - 1)
            codes = np.where(kept[pos] == values, pos, len(kept))
            out.append(codes.astype(float)[:, None])
        return np.hstack(out)


def make_tabular_model(kind: str, numeric_idx: Sequence[int], categorical_idx: Sequence[int],
                       random_state: int = 42) -> Pipeline:
    """A model for a raw feature matrix with the given numeric and nominal column positions."""
    numeric_idx, categorical_idx = list(numeric_idx), list(categorical_idx)
    if kind == "lr":
        preprocess = ColumnTransformer([
            ("numeric", StandardScaler(), numeric_idx),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_idx),
        ])
        return Pipeline([
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=5000, random_state=random_state)),
        ])
    if kind == "hgb":
        n_num = len(numeric_idx)
        mask = [False] * n_num + [True] * len(categorical_idx)
        return Pipeline([
            ("preprocess", CategoryCodes(numeric_idx, categorical_idx)),
            ("classifier", HistGradientBoostingClassifier(
                categorical_features=mask if categorical_idx else None,
                early_stopping=False, random_state=random_state)),
        ])
    raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")


def make_dense_model(kind: str, random_state: int = 42) -> Pipeline:
    """A model for an all-numeric feature matrix (image embeddings, sensor features)."""
    if kind == "lr":
        return Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000, random_state=random_state)),
        ])
    if kind == "hgb":
        return Pipeline([
            ("classifier", HistGradientBoostingClassifier(early_stopping=False, random_state=random_state)),
        ])
    raise ValueError(f"unknown model kind {kind!r}; expected one of {MODEL_KINDS}")
