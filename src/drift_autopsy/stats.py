"""Statistical tests shared by the detectors, the localizer and the benchmark.

Every decision rule in the package goes through these functions, so the
assumptions are stated once:

* Multiple testing is controlled with Bonferroni or the Holm step-down
  procedure (Holm, 1979), both of which bound the family-wise error rate.
* Categorical shift is a two-sample chi-square test of homogeneity on a
  2 x K contingency table. Categories whose expected count is below five in
  either sample are pooled into one "other" cell, the usual validity rule for
  the chi-square approximation (Cochran, 1954).
* Weighted tests (used to ask whether a reweighted reference explains the
  production data) replace the reference sample size with Kish's effective
  sample size (Kish, 1965).
* Severity labels for p-value tests are orders of magnitude below the
  significance level; see :func:`severity_from_p_value`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from drift_autopsy.core.result import DriftSeverity

MIN_EXPECTED_COUNT = 5.0


# --------------------------------------------------------------------------- #
# multiple testing

def bonferroni_reject(p_values: Sequence[float], alpha: float) -> np.ndarray:
    """Reject H0_i when p_i < alpha / m."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    return p < alpha / p.size


def holm_reject(p_values: Sequence[float], alpha: float) -> np.ndarray:
    """Holm step-down: sort p ascending, reject while p_(k) < alpha / (m - k)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(np.argsort(p, kind="stable")):
        if p[idx] < alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject


def adjusted_threshold(alpha: float, m: int, correction: Optional[str]) -> float:
    """Per-test threshold for the most significant test of a family of m tests.

    For the global "is anything drifting" decision Holm's first step and
    Bonferroni coincide, so both return alpha / m.
    """
    if correction in ("bonferroni", "holm") and m > 0:
        return alpha / m
    return alpha


def reject(p_values: Sequence[float], alpha: float, correction: Optional[str]) -> np.ndarray:
    """Per-test rejections under the named correction (None, bonferroni, holm)."""
    if correction == "bonferroni":
        return bonferroni_reject(p_values, alpha)
    if correction == "holm":
        return holm_reject(p_values, alpha)
    return np.asarray(p_values, dtype=float) < alpha


# --------------------------------------------------------------------------- #
# severity

def severity_from_p_value(p_value: Optional[float], threshold: float) -> DriftSeverity:
    """Map a p-value to a severity label relative to its rejection threshold t.

    none: p >= t; low: t/2 <= p < t; medium: t/10 <= p < t/2;
    high: t/100 <= p < t/10; critical: p < t/100.
    """
    if p_value is None or not np.isfinite(p_value) or p_value >= threshold:
        return DriftSeverity.NONE
    if p_value >= threshold / 2:
        return DriftSeverity.LOW
    if p_value >= threshold / 10:
        return DriftSeverity.MEDIUM
    if p_value >= threshold / 100:
        return DriftSeverity.HIGH
    return DriftSeverity.CRITICAL


def severity_from_ratio(score: float, threshold: float) -> DriftSeverity:
    """Severity for effect-size detectors as a multiple of their threshold t.

    none: s < t; low: t <= s < 1.5t; medium: 1.5t <= s < 2.5t;
    high: 2.5t <= s < 5t; critical: s >= 5t.
    """
    if not np.isfinite(score) or score < threshold:
        return DriftSeverity.NONE if np.isfinite(score) else DriftSeverity.CRITICAL
    if score < 1.5 * threshold:
        return DriftSeverity.LOW
    if score < 2.5 * threshold:
        return DriftSeverity.MEDIUM
    if score < 5.0 * threshold:
        return DriftSeverity.HIGH
    return DriftSeverity.CRITICAL


# --------------------------------------------------------------------------- #
# categorical tests

def _pool_sparse(ref_counts: np.ndarray, test_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Pool columns whose expected count is below MIN_EXPECTED_COUNT in either row."""
    table = np.vstack([ref_counts, test_counts]).astype(float)
    total = table.sum()
    if total <= 0:
        return ref_counts, test_counts, 0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    keep = (expected >= MIN_EXPECTED_COUNT).all(axis=0)
    n_pooled = int((~keep).sum())
    if n_pooled == 0:
        return ref_counts, test_counts, 0
    ref_kept = np.append(ref_counts[keep], ref_counts[~keep].sum())
    test_kept = np.append(test_counts[keep], test_counts[~keep].sum())
    # the pooled cell itself may still be sparse; drop it if so
    pooled_table = np.vstack([ref_kept, test_kept]).astype(float)
    exp_pooled = np.outer(pooled_table.sum(axis=1), pooled_table.sum(axis=0)) / pooled_table.sum()
    if (exp_pooled[:, -1] < MIN_EXPECTED_COUNT).any():
        ref_kept, test_kept = ref_kept[:-1], test_kept[:-1]
    return ref_kept, test_kept, n_pooled


def chi2_homogeneity_counts(ref_counts: np.ndarray, test_counts: np.ndarray) -> Dict[str, Any]:
    """Two-sample chi-square test of homogeneity on aligned category counts."""
    ref_counts = np.asarray(ref_counts, dtype=float)
    test_counts = np.asarray(test_counts, dtype=float)
    ref_kept, test_kept, n_pooled = _pool_sparse(ref_counts, test_counts)
    if len(ref_kept) < 2 or ref_kept.sum() <= 0 or test_kept.sum() <= 0:
        return {"statistic": 0.0, "p_value": 1.0, "dof": 0, "n_cells": int(len(ref_kept)),
                "n_pooled": n_pooled, "cramers_v": 0.0}
    table = np.vstack([ref_kept, test_kept])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return {"statistic": 0.0, "p_value": 1.0, "dof": 0, "n_cells": int(table.shape[1]),
                "n_pooled": n_pooled, "cramers_v": 0.0}
    stat, p, dof, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    cramers_v = float(np.sqrt(stat / (n * (min(table.shape) - 1)))) if n > 0 else 0.0
    return {"statistic": float(stat), "p_value": float(p), "dof": int(dof),
            "n_cells": int(table.shape[1]), "n_pooled": n_pooled, "cramers_v": cramers_v}


def chi2_homogeneity(ref_values: Sequence[Any], test_values: Sequence[Any],
                     ref_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Two-sample chi-square homogeneity test on raw category values.

    With ``ref_weights`` the reference counts are weighted and rescaled to
    Kish's effective sample size, which keeps the test honest about the
    information lost to weighting.
    """
    ref_values = np.asarray(ref_values, dtype=object)
    test_values = np.asarray(test_values, dtype=object)
    categories = sorted(set(ref_values.tolist()) | set(test_values.tolist()), key=str)
    index = {c: i for i, c in enumerate(categories)}
    ref_idx = np.fromiter((index[v] for v in ref_values), dtype=int, count=len(ref_values))
    test_idx = np.fromiter((index[v] for v in test_values), dtype=int, count=len(test_values))
    if ref_weights is None:
        ref_counts = np.bincount(ref_idx, minlength=len(categories)).astype(float)
    else:
        w = np.asarray(ref_weights, dtype=float)
        ref_counts = np.bincount(ref_idx, weights=w, minlength=len(categories))
        ref_counts = ref_counts / ref_counts.sum() * kish_effective_sample_size(w)
    test_counts = np.bincount(test_idx, minlength=len(categories)).astype(float)
    out = chi2_homogeneity_counts(ref_counts, test_counts)
    out["n_categories"] = len(categories)
    return out


# --------------------------------------------------------------------------- #
# weighted two-sample tests

def kish_effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    s2 = float(np.sum(w ** 2))
    return float(np.sum(w) ** 2 / s2) if s2 > 0 else 0.0


def weighted_ks(ref_values: np.ndarray, ref_weights: np.ndarray, test_values: np.ndarray) -> Dict[str, float]:
    """Two-sample KS statistic between a weighted reference and a test sample.

    The p-value uses the asymptotic Kolmogorov distribution with the
    reference size replaced by its Kish effective sample size.
    """
    x = np.asarray(ref_values, dtype=float)
    w = np.asarray(ref_weights, dtype=float)
    y = np.asarray(test_values, dtype=float)
    ok = np.isfinite(x)
    x, w = x[ok], w[ok]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0 or w.sum() <= 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_eff_reference": 0.0}
    grid = np.union1d(x, y)
    order = np.argsort(x, kind="stable")
    cw = np.cumsum(w[order]) / w.sum()
    cdf_x = np.concatenate([[0.0], cw])[np.searchsorted(x[order], grid, side="right")]
    cdf_y = np.searchsorted(np.sort(y), grid, side="right") / y.size
    d = float(np.max(np.abs(cdf_x - cdf_y)))
    n_eff = kish_effective_sample_size(w)
    en = n_eff * y.size / (n_eff + y.size)
    p = float(stats.kstwobign.sf(np.sqrt(en) * d)) if en > 0 else 1.0
    return {"statistic": d, "p_value": min(max(p, 0.0), 1.0), "n_eff_reference": n_eff}


# --------------------------------------------------------------------------- #
# proportions

def two_proportion_test(k1: int, n1: int, k2: int, n2: int, alternative: str = "two-sided") -> Dict[str, float]:
    """Pooled two-proportion z-test of p1 vs p2.

    ``alternative``: "two-sided", "greater" (p1 > p2) or "less" (p1 < p2).
    """
    if n1 <= 0 or n2 <= 0:
        return {"z": 0.0, "p_value": 1.0, "p1": 0.0, "p2": 0.0}
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se <= 0:
        return {"z": 0.0, "p_value": 1.0, "p1": float(p1), "p2": float(p2)}
    z = (p1 - p2) / se
    if alternative == "greater":
        p = stats.norm.sf(z)
    elif alternative == "less":
        p = stats.norm.cdf(z)
    else:
        p = 2 * stats.norm.sf(abs(z))
    return {"z": float(z), "p_value": float(p), "p1": float(p1), "p2": float(p2)}


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(0.5 + confidence / 2)
    phat = k / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    lo = 0.0 if k == 0 else float(max(0.0, centre - half))
    hi = 1.0 if k == n else float(min(1.0, centre + half))
    return (lo, hi)


# --------------------------------------------------------------------------- #
# paired comparison of two classifiers on the same labelled rows

def paired_accuracy_difference(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> Dict[str, float]:
    """Accuracy of B minus accuracy of A on the same rows, with a paired bootstrap.

    ``weights`` turns both accuracies into weighted means (importance-weighted
    validation). Rows are resampled jointly, so the pairing is kept. Returns the
    point difference, the two-sided (1 - alpha) percentile interval and the
    one-sided bootstrap p-value for H0: difference <= 0.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, dtype=float)
    if a.shape != b.shape or a.shape != w.shape or a.size == 0:
        raise ValueError("correct_a, correct_b and weights must be non-empty and the same shape")
    diff_rows = b - a
    point = float(np.sum(w * diff_rows) / np.sum(w))
    rng = np.random.default_rng(random_state)
    boot = np.empty(n_bootstrap)
    n = a.size
    for k in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        ww = w[idx]
        boot[k] = np.sum(ww * diff_rows[idx]) / np.sum(ww)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_greater = float((1 + np.sum(boot <= 0)) / (1 + n_bootstrap))
    return {"difference": point, "ci_low": float(lo), "ci_high": float(hi),
            "p_value_greater": p_greater, "n": int(n)}


# --------------------------------------------------------------------------- #
# generic permutation test

def permutation_test(
    statistic,
    X: np.ndarray,
    Y: np.ndarray,
    n_permutations: int,
    random_state: int = 0,
) -> Dict[str, float]:
    """Two-sample permutation test of ``statistic(X, Y)`` (larger = more different).

    The pooled rows are reassigned at random to groups of the original sizes
    ``n_permutations`` times; p = (1 + #{T_perm >= T_obs}) / (1 + n_permutations).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    observed = float(statistic(X, Y))
    Z = np.vstack([X, Y])
    n = len(X)
    rng = np.random.default_rng(random_state)
    exceed = 0
    for _ in range(n_permutations):
        idx = rng.permutation(len(Z))
        exceed += statistic(Z[idx[:n]], Z[idx[n:]]) >= observed
    return {"statistic": observed, "p_value": (1 + exceed) / (1 + n_permutations),
            "n_permutations": int(n_permutations)}


# --------------------------------------------------------------------------- #
# paired comparison of two methods over replicates

def sign_flip_test(
    differences: Sequence[float],
    n_flips: int = 10000,
    random_state: int = 0,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Paired test of H0: mean difference = 0 over independent replicates.

    ``differences`` holds one value per replicate (method A minus method B,
    e.g. the difference in the share of correctly diagnosed windows). Under H0
    each difference is symmetric about zero, so its sign is exchangeable: the
    p-value is (1 + #{|mean of randomly sign-flipped values| >= |observed|}) /
    (1 + n_flips), two-sided. The interval is a percentile bootstrap over
    replicates. Replicates, not windows, are the unit, because windows of one
    replicate share a model and a reference.
    """
    d = np.asarray(differences, dtype=float)
    if d.size == 0:
        raise ValueError("no replicates")
    observed = float(d.mean())
    rng = np.random.default_rng(random_state)
    signs = rng.choice([-1.0, 1.0], size=(n_flips, d.size))
    flipped = np.abs((signs * d).mean(axis=1))
    p = float((1 + np.sum(flipped >= abs(observed) - 1e-12)) / (1 + n_flips))
    boot = d[rng.integers(0, d.size, (n_bootstrap, d.size))].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean_difference": observed, "ci_low": float(lo), "ci_high": float(hi),
            "p_value": p, "n_replicates": int(d.size), "n_flips": int(n_flips)}


def macro_f1(true_labels: Sequence[Any], predicted_labels: Sequence[Any], labels: Sequence[Any]) -> float:
    """Unweighted mean of the per-class F1 scores over ``labels`` (0 for a class never predicted correctly)."""
    t = np.asarray(true_labels, dtype=object)
    p = np.asarray(predicted_labels, dtype=object)
    scores = []
    for c in labels:
        tp = float(np.sum((t == c) & (p == c)))
        fp = float(np.sum((t != c) & (p == c)))
        fn = float(np.sum((t == c) & (p != c)))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))
