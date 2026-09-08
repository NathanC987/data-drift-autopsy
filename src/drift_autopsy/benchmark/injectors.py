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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from drift_autopsy.benchmark.spec import InjectedDriftSpec
from drift_autopsy.remediation.synthetic import inject_covariate_shift as _scale_shift


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

    ``x' = mean + scale*(x - mean) + offset*sd`` per feature (delegated to the
    remediation module's shared implementation) applied to a random ``fraction``
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
        px_changed=False,
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
        pygivenx_changed=True,
        p_y_reference=_positive_rate(y_ref),
        p_y_production=_positive_rate(y_new),
    )
    return InjectionResult(X=X, y=y_new, spec=spec)
