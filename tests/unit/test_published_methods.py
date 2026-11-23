"""Tests for the published diagnosis methods and the noise mechanisms added to design A."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from drift_autopsy.benchmark.injectors import inject_class_conditional_noise, inject_instance_noise
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.published_methods import (
    confident_learning_audit,
    cpmcn_prevalence,
    label_free_then_cl,
    roschewitz_identification,
)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 9000
    cols = [f"f{i}" for i in range(4)]
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=cols)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * X["f0"] - X["f1"])))).astype(int)
    model = LogisticRegression().fit(X.iloc[:4000].to_numpy(), y[:4000])
    return X, y, cols, model


def _ds(model, X, y=None):
    proba = model.predict_proba(X.to_numpy())
    return Dataset(data=X.reset_index(drop=True), feature_names=list(X.columns), target=y,
                   predictions=model.classes_[proba.argmax(1)], prediction_probabilities=proba)


def test_cpmcn_recovers_prevalence_under_label_shift(toy):
    X, y, _, model = toy
    ref_X, ref_y = X.iloc[4000:6500], y[4000:6500]
    pool_X, pool_y = X.iloc[6500:].reset_index(drop=True), y[6500:]
    rng = np.random.default_rng(1)
    pos, neg = np.flatnonzero(pool_y == 1), np.flatnonzero(pool_y == 0)
    idx = np.r_[rng.choice(pos, 900, replace=False), rng.choice(neg, 300, replace=False)]   # 75% positive
    q = cpmcn_prevalence(np.array([np.mean(ref_y == 0), np.mean(ref_y == 1)]),
                         model.predict_proba(pool_X.iloc[idx].to_numpy()))
    assert abs(q[1] - 0.75) < 0.05


def test_roschewitz_names_prevalence_shift_and_no_shift(toy):
    X, y, _, model = toy
    ref = _ds(model, X.iloc[4000:6500], y[4000:6500])
    pool_X, pool_y = X.iloc[6500:].reset_index(drop=True), y[6500:]
    same = roschewitz_identification(ref, _ds(model, pool_X.iloc[:1200]), n_permutations=200, seed=0)
    assert same["verdict"] == "none"
    rng = np.random.default_rng(2)
    pos, neg = np.flatnonzero(pool_y == 1), np.flatnonzero(pool_y == 0)
    idx = np.r_[rng.choice(pos, 900, replace=False), rng.choice(neg, 300, replace=False)]
    shifted = roschewitz_identification(ref, _ds(model, pool_X.iloc[idx]), n_permutations=200, seed=0)
    assert shifted["shift"] == "prevalence" and shifted["verdict"] == "prior"


def test_class_conditional_noise_flips_positives_at_twice_the_rate():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.normal(size=40000)})
    y = (rng.random(40000) < 0.4).astype(int)
    res = inject_class_conditional_noise(X, y, 0.2, "moderate", seed=0)
    flipped = res.y != y
    assert abs(flipped[y == 1].mean() - 0.2) < 0.01
    assert abs(flipped[y == 0].mean() - 0.1) < 0.01
    assert res.X.equals(X)


def test_instance_noise_follows_uncertainty_with_the_given_mean():
    rng = np.random.default_rng(4)
    n = 40000
    X = pd.DataFrame({"a": rng.normal(size=n)})
    y = (rng.random(n) < 0.5).astype(int)
    u = rng.uniform(0, 1, size=n)
    res = inject_instance_noise(X, y, u, 0.1, "mild", seed=0)
    flipped = res.y != y
    assert abs(flipped.mean() - 0.1) < 0.01
    assert flipped[u > 0.8].mean() > 2 * flipped[u < 0.2].mean()


def test_confident_learning_flags_noisy_audit(toy):
    X, y, _, model = toy
    ref = _ds(model, X.iloc[4000:6500], y[4000:6500])
    aX, ay = X.iloc[6500:7100].reset_index(drop=True), y[6500:7100].copy()
    clean = confident_learning_audit(model, ref, aX, ay)
    assert not clean["significant"]
    noisy_y = ay.copy()
    flip = np.random.default_rng(5).random(len(ay)) < 0.25
    noisy_y[flip] = 1 - noisy_y[flip]
    noisy = confident_learning_audit(model, ref, aX, noisy_y)
    assert noisy["significant"]
    assert label_free_then_cl("none", noisy) == "label_noise"
    assert label_free_then_cl("covariate", noisy) == "covariate"


def test_detectshift_names_concept_drift_and_no_shift(toy):
    from drift_autopsy.published_methods import detectshift_diagnosis

    X, y, cols, _ = toy
    ref_X, ref_y = X.iloc[4000:5200].reset_index(drop=True), y[4000:5200]
    tgt_X = X.iloc[6000:6600].reset_index(drop=True)
    same = detectshift_diagnosis(ref_X, ref_y, tgt_X, y[6000:6600], n_permutations=100, seed=0)
    assert same["verdict"] == "none"
    flipped = np.where(tgt_X["f0"] > 0, 1 - y[6000:6600], y[6000:6600])      # labels inverted in a region
    conc = detectshift_diagnosis(ref_X, ref_y, tgt_X, flipped, n_permutations=100, seed=0)
    assert conc["verdict"] == "concept"
