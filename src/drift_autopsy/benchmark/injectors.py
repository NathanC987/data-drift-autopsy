"""Controlled drift injectors for the verification benchmark.

Each injector takes a clean feature frame and its true labels and returns a
production window plus an :class:`InjectedDriftSpec` answer key. The four
injectors move different parts of the joint distribution:

* ``inject_covariate``   -- P(X) moves on a named feature subset, P(Y|X) fixed.
* ``inject_prior``       -- P(Y) moves, P(X|Y) preserved (whole rows resampled).
* ``inject_concept``     -- P(Y|X) is re-written in a feature-defined region,
  every X left untouched.
* ``inject_label_noise`` -- P(Y|X) is corrupted by unstructured random flips,
  every X left untouched.

``inject_none`` is the control: the window is returned unchanged and the
pipeline is expected to report no drift.

Held-out mechanisms (design B). The four injectors above partly match what the
tests look for (a feature-wise shift, a region defined by one feature,
uniform flips). Design B moves the same parts of the joint distribution in
ways the tests were not built around:

* ``inject_selection``       -- covariate shift by selection bias: each row is
  kept with probability sigmoid(beta * s(x)) for a random score s over all
  inputs. Values stay real rows and P(Y|X) is exactly unchanged.
* ``inject_prior`` with a lower target -- the prior decreases instead.
* ``inject_concept_oblique`` -- labels inverted in the top fraction of a second
  random score: an oblique region no single feature defines.
* ``inject_threshold``       -- the label rule itself changes (income threshold
  raised); X untouched, P(Y|X) moves everywhere near the old boundary.
* ``inject_class_noise``     -- class-conditional label noise: only positives
  are flipped, so the noise depends on the true class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from drift_autopsy.benchmark.spec import InjectedDriftSpec


def _scale_shift(X: np.ndarray, feature_names: Sequence[str], shifted_features: Sequence[str],
                 scale: float, offset: float) -> np.ndarray:
    """Copy of ``X`` with each named feature mapped to mu + scale * (x - mu) + offset * sd."""
    X = np.asarray(X, dtype=float).copy()
    names = list(feature_names)
    for f in shifted_features:
        j = names.index(f)
        col = X[:, j]
        mu, sd = col.mean(), col.std()
        X[:, j] = mu + scale * (col - mu) + offset * sd
    return X


@dataclass
class InjectionResult:
    """A production window and the answer key that produced it."""

    X: pd.DataFrame
    y: np.ndarray
    spec: InjectedDriftSpec


def _positive_rate(y: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean(y)) if len(y) else 0.0


def inject_none(X: pd.DataFrame, y: np.ndarray, intensity_label: str = "none") -> InjectionResult:
    """Control window: no change at all."""
    X = X.reset_index(drop=True).copy()
    y = np.asarray(y).astype(int)
    spec = InjectedDriftSpec(
        drift_type="none",
        intensity_label=intensity_label,
        params={},
        affected_features=[],
        px_changed=False,
        pygivenx_changed=False,
        p_y_reference=_positive_rate(y),
        p_y_production=_positive_rate(y),
    )
    return InjectionResult(X=X, y=y, spec=spec)


def inject_covariate(
    X: pd.DataFrame,
    y: np.ndarray,
    features: Sequence[str],
    scale: float,
    offset: float,
    intensity_label: str,
    fraction: float = 0.6,
    seed: int = 42,
) -> InjectionResult:
    """Scale and mean-offset a named feature subset for a subpopulation.

    ``x' = mean + scale*(x - mean) + offset*sd`` per feature, applied to a random ``fraction``
    of rows, so the marginal spread and mean move, the shift is heterogeneous
    across the population (a realistic new segment), and the target rule P(Y|X)
    is untouched.
    """
    X = X.reset_index(drop=True).copy()
    y = np.asarray(y).astype(int)
    names = list(X.columns)
    features = [f for f in features if f in names]
    rng = np.random.default_rng(seed)

    fraction = float(np.clip(fraction, 0.05, 1.0))
    n = len(X)
    n_shift = int(round(fraction * n))
    shift_rows = np.zeros(n, dtype=bool)
    shift_rows[rng.choice(n, n_shift, replace=False)] = True

    arr = X.to_numpy(dtype=float)
    shifted_full = _scale_shift(arr, names, features, scale=scale, offset=offset)
    arr[shift_rows] = shifted_full[shift_rows]
    X_new = pd.DataFrame(arr, columns=names)

    spec = InjectedDriftSpec(
        drift_type="covariate",
        intensity_label=intensity_label,
        params={"scale": scale, "offset": offset, "fraction": fraction},
        affected_features=list(features),
        px_changed=True,
        pxgiveny_changed=True,
        pygivenx_changed=False,
        p_y_reference=_positive_rate(y),
        p_y_production=_positive_rate(y),
    )
    return InjectionResult(X=X_new, y=y, spec=spec)


def inject_prior(
    X: pd.DataFrame,
    y: np.ndarray,
    target_positive_rate: float,
    intensity_label: str,
    seed: int = 42,
) -> InjectionResult:
    """Resample whole rows so P(Y) hits ``target_positive_rate``, P(X|Y) fixed.

    Rows are drawn (without replacement where possible) from the existing
    positive and negative pools in the ratio the target prior implies, so each
    class-conditional feature distribution is exactly a subsample of the
    original -- only the mixing weight changes.
    """
    X = X.reset_index(drop=True).copy()
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    base_rate = _positive_rate(y)
    target = float(np.clip(target_positive_rate, 0.02, 0.98))

    # Keep the larger class whole, downsample the other to hit the ratio.
    if target >= base_rate:
        n_pos = len(pos_idx)
        n_neg = int(round(n_pos * (1 - target) / max(target, 1e-6)))
        n_neg = min(n_neg, len(neg_idx))
        keep = np.concatenate([pos_idx, rng.choice(neg_idx, n_neg, replace=False)])
    else:
        n_neg = len(neg_idx)
        n_pos = int(round(n_neg * target / max(1 - target, 1e-6)))
        n_pos = min(n_pos, len(pos_idx))
        keep = np.concatenate([rng.choice(pos_idx, n_pos, replace=False), neg_idx])

    keep = np.sort(keep)
    X_new = X.iloc[keep].reset_index(drop=True)
    y_new = y[keep]

    spec = InjectedDriftSpec(
        drift_type="prior",
        intensity_label=intensity_label,
        params={"target_positive_rate": target},
        affected_features=[],
        px_changed=True,
        pxgiveny_changed=False,
        pygivenx_changed=False,
        p_y_reference=base_rate,
        p_y_production=_positive_rate(y_new),
    )
    return InjectionResult(X=X_new, y=y_new, spec=spec)


def inject_concept(
    X: pd.DataFrame,
    y: np.ndarray,
    region_feature: str,
    region_fraction: float,
    intensity_label: str,
    seed: int = 42,
) -> InjectionResult:
    """Deterministically invert the label for the top ``region_fraction`` of rows
    ranked by ``region_feature``.

    Rank selection (rather than a quantile cutoff) keeps the region size exactly
    ``region_fraction`` even when the feature is coarse or discrete. X is not
    touched at all, so P(X) is identical to the reference; inside the region the
    target rule is the opposite of what the model learned -- a genuine,
    feature-localised P(Y|X) change.
    """
    X = X.reset_index(drop=True).copy()
    y_ref = np.asarray(y).astype(int)

    if region_feature not in X.columns:
        raise ValueError(f"region_feature {region_feature!r} not in columns")

    rng = np.random.default_rng(seed)
    col = X[region_feature].to_numpy(dtype=float)
    # rank high-to-low with a deterministic random tie-break
    order = np.lexsort((rng.random(len(col)), -col))
    n_region = int(round(float(np.clip(region_fraction, 0.02, 0.9)) * len(col)))
    region = np.zeros(len(col), dtype=bool)
    region[order[:n_region]] = True

    y_new = y_ref.copy()
    y_new[region] = 1 - y_new[region]

    spec = InjectedDriftSpec(
        drift_type="concept",
        intensity_label=intensity_label,
        params={
            "region_feature": region_feature,
            "region_fraction": round(float(np.mean(region)), 4),
        },
        affected_features=[region_feature],
        px_changed=False,
        pxgiveny_changed=True,
        pygivenx_changed=True,
        p_y_reference=_positive_rate(y_ref),
        p_y_production=_positive_rate(y_new),
    )
    return InjectionResult(X=X, y=y_new, spec=spec)


def inject_label_noise(
    X: pd.DataFrame,
    y: np.ndarray,
    rate: float,
    intensity_label: str,
    seed: int = 42,
) -> InjectionResult:
    """Flip a random ``rate`` fraction of labels, uniformly, with no structure.

    X is untouched; the corruption does not depend on any feature, so unlike
    ``inject_concept`` there is no region for localisation to find.
    """
    X = X.reset_index(drop=True).copy()
    y_ref = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)

    rate = float(np.clip(rate, 0.0, 0.5))
    flip = rng.random(len(y_ref)) < rate
    y_new = y_ref.copy()
    y_new[flip] = 1 - y_new[flip]

    spec = InjectedDriftSpec(
        drift_type="label_noise",
        intensity_label=intensity_label,
        params={"rate": rate, "observed_flip_fraction": float(np.mean(flip))},
        affected_features=[],
        px_changed=False,
        pxgiveny_changed=True,
        pygivenx_changed=True,
        p_y_reference=_positive_rate(y_ref),
        p_y_production=_positive_rate(y_new),
    )
    return InjectionResult(X=X, y=y_new, spec=spec)


# --------------------------------------------------------------------------- #
# design B: held-out mechanisms

class RandomDirection:
    """A seeded random score s(x) over every input, fixed per replicate.

    Numeric features are standardised with the training statistics and get an
    N(0, 1) weight each; every category of a nominal feature gets its own
    N(0, 1) effect (categories are drawn in sorted order, unseen ones on first
    use, so the score is a deterministic function of x). The sum is divided
    by its training standard deviation and centred on the training mean, so
    s has mean 0 and unit variance on the training data.
    """

    def __init__(self, train_X: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str], seed: int):
        self.numeric, self.categorical = list(numeric), list(categorical)
        self.rng = np.random.default_rng(seed)
        self.mean = train_X[self.numeric].mean().to_numpy(dtype=float)
        sd = train_X[self.numeric].std(ddof=0).to_numpy(dtype=float)
        self.sd = np.where(sd > 0, sd, 1.0)
        self.weights = self.rng.standard_normal(len(self.numeric))
        self.effects: Dict[str, Dict[float, float]] = {}
        for f in self.categorical:
            codes = np.sort(train_X[f].unique())
            self.effects[f] = dict(zip(codes.astype(float), self.rng.standard_normal(len(codes))))
        raw = self._raw(train_X)
        self.centre, self.scale = float(raw.mean()), float(raw.std() or 1.0)

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        z = (X[self.numeric].to_numpy(dtype=float) - self.mean) / self.sd
        score = z @ self.weights
        for f in self.categorical:
            table = self.effects[f]
            for code in np.sort(X[f].unique()).astype(float):
                if code not in table:
                    table[code] = float(self.rng.standard_normal())
            score = score + X[f].astype(float).map(table).to_numpy()
        return score

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return (self._raw(X) - self.centre) / self.scale


def inject_selection(X: pd.DataFrame, y: np.ndarray, direction: RandomDirection, beta: float,
                     intensity_label: str, seed: int = 42) -> InjectionResult:
    """Keep each row with probability sigmoid(beta * s(x)) (see module docstring)."""
    X = X.reset_index(drop=True).copy()
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    keep_prob = 1.0 / (1.0 + np.exp(-beta * direction.score(X)))
    keep = rng.random(len(X)) < keep_prob
    X_new, y_new = X[keep].reset_index(drop=True), y[keep]
    spec = InjectedDriftSpec(
        drift_type="covariate", intensity_label=intensity_label,
        params={"beta": beta, "kept_fraction": float(keep.mean())},
        affected_features=[], px_changed=True, pxgiveny_changed=True, pygivenx_changed=False,
        p_y_reference=_positive_rate(y), p_y_production=_positive_rate(y_new),
    )
    return InjectionResult(X=X_new, y=y_new, spec=spec)


def inject_concept_oblique(X: pd.DataFrame, y: np.ndarray, direction: RandomDirection, region_fraction: float,
                           intensity_label: str, seed: int = 42) -> InjectionResult:
    """Invert the label in the top ``region_fraction`` of rows ranked by s(x); X untouched."""
    X = X.reset_index(drop=True).copy()
    y_ref = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    s = direction.score(X)
    order = np.lexsort((rng.random(len(s)), -s))
    region = np.zeros(len(s), dtype=bool)
    region[order[: int(round(region_fraction * len(s)))]] = True
    y_new = y_ref.copy()
    y_new[region] = 1 - y_new[region]
    spec = InjectedDriftSpec(
        drift_type="concept", intensity_label=intensity_label,
        params={"region_fraction": round(float(region.mean()), 4)},
        affected_features=[], px_changed=False, pxgiveny_changed=True, pygivenx_changed=True,
        p_y_reference=_positive_rate(y_ref), p_y_production=_positive_rate(y_new),
    )
    return InjectionResult(X=X, y=y_new, spec=spec)


def inject_threshold(X: pd.DataFrame, y: np.ndarray, income: np.ndarray, threshold: float,
                     intensity_label: str) -> InjectionResult:
    """Relabel with a higher income threshold, y = income > threshold; X untouched."""
    X = X.reset_index(drop=True).copy()
    y_ref = np.asarray(y).astype(int)
    y_new = (np.asarray(income, dtype=float) > threshold).astype(int)
    spec = InjectedDriftSpec(
        drift_type="concept", intensity_label=intensity_label,
        params={"threshold": float(threshold), "relabelled_fraction": float(np.mean(y_new != y_ref))},
        affected_features=[], px_changed=False, pxgiveny_changed=True, pygivenx_changed=True,
        p_y_reference=_positive_rate(y_ref), p_y_production=_positive_rate(y_new),
    )
    return InjectionResult(X=X, y=y_new, spec=spec)


def inject_class_noise(X: pd.DataFrame, y: np.ndarray, rate: float, intensity_label: str,
                       seed: int = 42) -> InjectionResult:
    """Flip each positive label to negative with probability ``rate``; X untouched."""
    X = X.reset_index(drop=True).copy()
    y_ref = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    flip = (y_ref == 1) & (rng.random(len(y_ref)) < rate)
    y_new = y_ref.copy()
    y_new[flip] = 0
    spec = InjectedDriftSpec(
        drift_type="label_noise", intensity_label=intensity_label,
        params={"rate": rate, "observed_flip_fraction": float(np.mean(flip))},
        affected_features=[], px_changed=False, pxgiveny_changed=True, pygivenx_changed=True,
        p_y_reference=_positive_rate(y_ref), p_y_production=_positive_rate(y_new),
    )
    return InjectionResult(X=X, y=y_new, spec=spec)
