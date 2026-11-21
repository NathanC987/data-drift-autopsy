"""Tests for the drift injectors, the test-based diagnosis and the benchmark aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from drift_autopsy.benchmark import (
    inject_concept,
    inject_covariate,
    inject_label_noise,
    inject_none,
    inject_prior,
)
from drift_autopsy.benchmark.metrics import aggregate
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.diagnosis import audit_diagnosis, bbse_weights, combine, label_free_diagnosis


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 12000
    cols = [f"f{i}" for i in range(5)]
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
    X["f4"] = rng.choice([1.0, 2.0, 3.0], size=n)          # a nominal code
    logits = 1.2 * X["f0"] - 0.9 * X["f2"] + 0.5 * (X["f4"] == 3.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y, cols


def _split(X, y):
    return (X.iloc[:6000], y[:6000]), (X.iloc[6000:9000].reset_index(drop=True), y[6000:9000]), \
           (X.iloc[9000:].reset_index(drop=True), y[9000:])


def _ds(model, X, y=None):
    proba = model.predict_proba(X.to_numpy())
    return Dataset(data=X.reset_index(drop=True), target=y, predictions=proba.argmax(1),
                   prediction_probabilities=proba, categorical_features=["f4"])


# --------------------------------------------------------------------------- #
# injectors

def test_inject_none_is_identity(toy):
    X, y, _ = toy
    r = inject_none(X, y)
    pd.testing.assert_frame_equal(r.X, X.reset_index(drop=True))
    assert np.array_equal(r.y, y)
    assert not r.spec.px_changed and not r.spec.pygivenx_changed


def test_inject_covariate_touches_only_target_features(toy):
    X, y, _ = toy
    r = inject_covariate(X, y, ["f0", "f1"], scale=2.0, offset=1.5, intensity_label="strong", fraction=1.0)
    for f in ["f2", "f3", "f4"]:
        assert np.allclose(r.X[f].to_numpy(), X[f].to_numpy())
    assert r.X["f0"].std() > X["f0"].std() * 1.5
    assert np.array_equal(r.y, y)
    assert r.spec.px_changed and r.spec.pxgiveny_changed and not r.spec.pygivenx_changed


def test_inject_prior_preserves_conditionals_but_moves_px(toy):
    X, y, _ = toy
    r = inject_prior(X, y, target_positive_rate=0.65, intensity_label="strong", seed=0)
    assert abs(r.y.mean() - 0.65) < 0.03
    for cls in (0, 1):
        assert np.allclose(X.to_numpy()[y == cls].mean(0), r.X.to_numpy()[r.y == cls].mean(0), atol=0.1)
    assert r.spec.px_changed and not r.spec.pxgiveny_changed and not r.spec.pygivenx_changed


def test_inject_concept_freezes_px_and_flips_only_the_region(toy):
    X, y, _ = toy
    r = inject_concept(X, y, region_feature="f0", region_fraction=0.3, intensity_label="m", seed=0)
    pd.testing.assert_frame_equal(r.X, X.reset_index(drop=True))
    flipped = r.y != y
    assert abs(flipped.mean() - 0.3) < 0.02
    assert X["f0"].to_numpy()[flipped].min() >= np.quantile(X["f0"], 0.65)
    assert r.spec.pygivenx_changed and not r.spec.px_changed


def test_inject_label_noise_freezes_px_and_flips_about_rate(toy):
    X, y, _ = toy
    r = inject_label_noise(X, y, rate=0.2, intensity_label="strong", seed=0)
    pd.testing.assert_frame_equal(r.X, X.reset_index(drop=True))
    assert abs((r.y != y).mean() - 0.2) < 0.03


# --------------------------------------------------------------------------- #
# diagnosis

def test_bbse_recovers_a_prior_shift(toy):
    X, y, _ = toy
    (Xt, yt), (Xc, yc), (Xp, yp) = _split(X, y)
    model = LogisticRegression().fit(Xt.to_numpy(), yt)
    shifted = inject_prior(Xp, yp, target_positive_rate=0.75, intensity_label="s", seed=1)
    est = bbse_weights(model.predict(Xc.to_numpy()), yc, model.predict(shifted.X.to_numpy()), [0, 1])
    assert abs(est["estimated_production_prior"][1] - shifted.y.mean()) < 0.05


@pytest.mark.parametrize("kind,expected", [("none", "none"), ("covariate", "covariate"), ("prior", "prior"),
                                           ("concept", "concept"), ("label_noise", "label_noise")])
def test_diagnosis_names_each_injected_type(toy, kind, expected):
    X, y, _ = toy
    (Xt, yt), (Xc, yc), (Xp, yp) = _split(X, y)
    model = LogisticRegression().fit(Xt.to_numpy(), yt)
    inj = {
        "none": lambda: inject_none(Xp, yp),
        "covariate": lambda: inject_covariate(Xp, yp, ["f1", "f3"], 2.0, 1.5, "s", fraction=0.6, seed=2),
        "prior": lambda: inject_prior(Xp, yp, 0.75, "s", seed=2),
        "concept": lambda: inject_concept(Xp, yp, "f2", 0.3, "s", seed=2),
        "label_noise": lambda: inject_label_noise(Xp, yp, 0.25, "s", seed=2),
    }[kind]()
    ref = _ds(model, Xc, yc)
    lf = label_free_diagnosis(ref, _ds(model, inj.X))
    idx = np.random.default_rng(3).choice(len(inj.X), 800, replace=False)
    au = audit_diagnosis(model, ref, inj.X.iloc[idx], inj.y[idx])
    d = combine(lf, au)
    assert d.drift_type == expected
    if kind == "covariate":
        assert set(d.localized_features) <= {"f1", "f3"} and d.localized_features
    if kind == "concept":
        assert d.localized_features == ["f2"]


# --------------------------------------------------------------------------- #
# aggregation

def _run(t, pred, seed, helps=False, escalate=False, err=0.01):
    return {
        "state": "S", "seed": seed, "drift_type": t, "intensity_label": "mild", "intensity_order": 0,
        "diagnosis": {"drift_type": pred, "stage": "label-free"}, "diagnosis_correct": pred == t,
        "diagnosis_by_audit_size": {"200": pred}, "estimator": {"error": err},
        "reliability": {"auroc_risk": 0.7, "auroc_confidence_only": 0.69, "auroc_ood": 0.5,
                        "auroc_instability": 0.6},
        "triage": {"will_retraining_help": escalate,
                   "probes": [{"probe": "importance_weighted_retrain", "difference": 0.01}]},
        "triage_truth": {"retraining_helps": helps, "actual_gain": 0.02 if helps else 0.0},
        "localization": None, "detection": {"KS Test": {"drift_detected": False}},
        "label_free": {"min_input_p_value": 0.5}, "true_accuracy_drop": 0.0, "audit": {"audit_accuracy": 0.8},
    }


def test_aggregate_counts_and_intervals():
    runs = [_run("none", "none", 0), _run("none", "covariate", 1), _run("covariate", "covariate", 0, True, True)]
    s = aggregate(runs)
    assert s["type_identification"]["overall"]["k"] == 2 and s["type_identification"]["overall"]["n"] == 3
    lo, hi = s["type_identification"]["per_type"]["none"]["ci"]
    assert lo < 0.5 < hi
    assert s["triage"]["confusion"] == {"tp": 1, "tn": 2, "fp": 0, "fn": 0}
    assert s["detector_false_alarms"]["KS Test"]["rate"] == 0.0


@pytest.mark.integration
def test_run_replicate_end_to_end_small():
    from drift_autopsy.benchmark import BenchmarkConfig, run_replicate

    cfg = BenchmarkConfig(states=("WA",), seeds=(0,), production_size=1500, holdout_size=800,
                          audit_sizes=(400,), primary_audit_size=400, mmd_permutations=20,
                          drift_types=["none", "covariate"])
    records = run_replicate("WA", 0, cfg)
    assert len(records) == 4
    assert {r["drift_type"] for r in records} == {"none", "covariate"}
    assert all("diagnosis" in r and "triage_truth" in r for r in records)
