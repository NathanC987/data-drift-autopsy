"""Regression tests for the SHAP explained output and MMD standardisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.detectors import MMD
from drift_autopsy.rca import SHAPAnalyzer


def _scaled_problem(n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    informative = rng.normal(size=n) * 1000.0 + 5000.0   # large raw scale, drives the label
    nuisance = rng.normal(size=(n, 3))
    X = np.column_stack([informative, nuisance])
    y = (informative + rng.normal(scale=300.0, size=n) > 5000.0).astype(int)
    return pd.DataFrame(X, columns=["big", "n1", "n2", "n3"]), y


def test_shap_explains_positive_class_probability_of_full_pipeline():
    pytest.importorskip("shap")
    X, y = _scaled_problem()
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(X.to_numpy(), y)

    np.random.seed(0)
    analyzer = SHAPAnalyzer(n_background_samples=50, n_test_samples=50)
    result = analyzer.analyze(Dataset.from_pandas(X), Dataset.from_pandas(X), model=model)

    assert result.metadata["explained_output"] == "predict_proba[:,1]"
    ref_imp = {k: v["ref_importance"] for k, v in result.distribution_changes.items()}
    assert max(ref_imp, key=ref_imp.get) == "big"
    assert 0.0 < ref_imp["big"] < 1.0


def test_mmd_standardize_exposes_shift_hidden_by_large_scale_feature():
    rng = np.random.default_rng(1)
    n = 800
    ref = np.column_stack([rng.normal(size=n) * 5000.0, rng.normal(size=n)])
    test = np.column_stack([rng.normal(size=n) * 5000.0, rng.normal(loc=1.0, size=n)])
    ref_ds, test_ds = Dataset.from_numpy(ref), Dataset.from_numpy(test)

    raw = MMD(n_permutations=100).fit_detect(ref_ds, test_ds)
    std = MMD(n_permutations=100, standardize=True).fit_detect(ref_ds, test_ds)

    assert not raw.drift_detected          # the large-scale nuisance feature hides the shift
    assert std.drift_detected and std.p_value < 0.05
    assert std.metadata["standardized"] is True


def test_mmd_permutation_pvalue_is_seeded():
    rng = np.random.default_rng(2)
    ref = Dataset.from_numpy(rng.normal(size=(200, 3)))
    test = Dataset.from_numpy(rng.normal(loc=0.2, size=(200, 3)))

    p1 = MMD(n_permutations=30, random_state=7).fit_detect(ref, test).p_value
    p2 = MMD(n_permutations=30, random_state=7).fit_detect(ref, test).p_value
    assert p1 == p2


def test_mmd_permutation_test_controls_type_one_error():
    rng = np.random.default_rng(11)
    rejections = 0
    n_trials = 60
    for t in range(n_trials):
        ref = Dataset.from_numpy(rng.normal(size=(80, 3)))
        test = Dataset.from_numpy(rng.normal(size=(80, 3)))
        rejections += MMD(n_permutations=99, random_state=t).fit_detect(ref, test).drift_detected
    # nominal 5%; with 60 trials the 99th percentile of Binomial(60, 0.05) is 8
    assert rejections <= 8


def test_shap_change_intervals_separate_noise_from_real_change():
    pytest.importorskip("shap")
    X, y = _scaled_problem(n=3000, seed=5)
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(X.to_numpy(), y)
    ref = Dataset.from_pandas(X.iloc[:1500])
    same = Dataset.from_pandas(X.iloc[1500:])
    analyzer = SHAPAnalyzer(n_background_samples=60, n_test_samples=200, random_state=1)

    null = analyzer.analyze(ref, same, model=model)
    assert null.metadata["n_significant_changes"] == 0

    moved = X.iloc[1500:].copy()
    moved["big"] = 5000.0 + (moved["big"] - 5000.0) * 3.0   # wider spread -> larger |SHAP|
    changed = analyzer.analyze(ref, Dataset.from_pandas(moved), model=model)
    assert changed.distribution_changes["big"]["change_significant"]
    assert changed.distribution_changes["big"]["change"] > 0
