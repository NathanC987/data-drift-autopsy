"""Tests for drift_autopsy.stats and categorical-aware detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.detectors import KSTest, MMD, PSI
from drift_autopsy.localizers import UnivariateLocalizer
from drift_autopsy.stats import (
    chi2_homogeneity,
    holm_reject,
    kish_effective_sample_size,
    two_proportion_test,
    weighted_ks,
    wilson_interval,
)


def test_holm_step_down_stops_at_first_non_rejection():
    # alpha=0.05, m=4: thresholds 0.0125, 0.0167, 0.025, 0.05
    p = [0.01, 0.04, 0.03, 0.005]
    assert holm_reject(p, 0.05).tolist() == [True, False, False, True]
    # 0.02 fails its 0.0167 threshold, so 0.03 (< 0.025? no) and 0.04 are not tested
    assert holm_reject([0.001, 0.02, 0.024, 0.04], 0.05).tolist() == [True, False, False, False]


def test_chi2_homogeneity_same_vs_shifted_and_pooling():
    rng = np.random.default_rng(0)
    cats = np.array(list("abcde"))
    ref = rng.choice(cats, 3000, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    same = rng.choice(cats, 3000, p=[0.3, 0.3, 0.2, 0.1, 0.1])
    shifted = rng.choice(cats, 3000, p=[0.1, 0.1, 0.2, 0.3, 0.3])
    assert chi2_homogeneity(ref, same)["p_value"] > 0.01
    assert chi2_homogeneity(ref, shifted)["p_value"] < 1e-10
    # rare categories are pooled so the test stays valid
    rare = np.concatenate([ref, ["z1", "z2", "z3"]])
    out = chi2_homogeneity(rare, same)
    assert out["n_pooled"] >= 3


def test_weighted_ks_matches_scipy_with_unit_weights_and_detects_reweighting():
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    y = rng.normal(loc=0.5, size=1500)
    ours = weighted_ks(x, np.ones_like(x), y)
    assert abs(ours["statistic"] - sps.ks_2samp(x, y).statistic) < 1e-12
    # importance weights exp(0.5 x) tilt N(0,1) to N(0.5,1): the shift disappears
    tilted = weighted_ks(x, np.exp(0.5 * x), y)
    assert tilted["p_value"] > 0.01
    assert tilted["n_eff_reference"] < 2000


def test_kish_and_proportions_and_wilson():
    assert kish_effective_sample_size(np.ones(10)) == 10
    assert two_proportion_test(60, 100, 40, 100, "greater")["p_value"] < 0.01
    lo, hi = wilson_interval(13, 13)
    assert hi == 1.0 and lo > 0.7


def _mixed_frame(n, rng, occ_p, age_loc):
    return pd.DataFrame({
        "AGE": rng.normal(age_loc, 10, n),
        "OCC": rng.choice([1010, 2020, 3030, 4040], n, p=occ_p).astype(float),
    })


def test_ks_skips_categorical_and_localizer_uses_chi2():
    rng = np.random.default_rng(2)
    ref = Dataset.from_pandas(_mixed_frame(3000, rng, [0.25] * 4, 40), categorical_features=["OCC"])
    test = Dataset.from_pandas(_mixed_frame(3000, rng, [0.1, 0.1, 0.4, 0.4], 40), categorical_features=["OCC"])

    ks = KSTest(threshold=0.05, correction="bonferroni").fit_detect(ref, test)
    assert list(ks.metadata["feature_results"]) == ["AGE"]
    assert not ks.drift_detected

    loc = UnivariateLocalizer(threshold=0.05, correction="bonferroni").localize(ref, test)
    by_name = {fd.feature_name: fd for fd in loc.feature_drifts}
    assert by_name["OCC"].drift_detected and not by_name["AGE"].drift_detected
    assert by_name["OCC"].distribution_shift["test"] == "categorical"
    assert 0.0 < by_name["OCC"].score <= 1.0          # Cramer's V


def test_psi_counts_values_outside_reference_range():
    rng = np.random.default_rng(4)
    ref = Dataset.from_pandas(pd.DataFrame({"x": rng.uniform(0, 1, 2000)}))
    test = Dataset.from_pandas(pd.DataFrame({"x": rng.uniform(5, 6, 2000)}))
    result = PSI(threshold=0.2).fit_detect(ref, test)
    assert result.score > 2.0 and result.drift_detected


def test_mmd_categorical_shift_is_seen_without_numeric_change():
    rng = np.random.default_rng(5)
    ref = Dataset.from_pandas(_mixed_frame(600, rng, [0.25] * 4, 40), categorical_features=["OCC"])
    test = Dataset.from_pandas(_mixed_frame(600, rng, [0.7, 0.1, 0.1, 0.1], 40), categorical_features=["OCC"])
    out = MMD(n_permutations=100, standardize=True).fit_detect(ref, test)
    assert out.p_value < 0.05
    assert out.metadata["n_categorical_features"] == 1
