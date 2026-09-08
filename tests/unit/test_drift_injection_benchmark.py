"""Unit tests for the controlled drift-injection benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from drift_autopsy.benchmark import (
    classify_drift_type,
    estimator_error,
    inject_concept,
    inject_covariate,
    inject_label_noise,
    inject_none,
    inject_prior,
    label_free_accuracy_estimate,
    labelled_audit,
    localisation_prf,
)
from drift_autopsy.benchmark.evaluate import BenchmarkData, run_benchmark
from drift_autopsy.benchmark.spec import InjectedDriftSpec


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 4000
    cols = [f"f{i}" for i in range(6)]
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=cols)
    logits = 0.9 * X["f0"] - 0.7 * X["f2"] + 0.4 * X["f4"]
    y = (logits + rng.normal(scale=0.5, size=n) > 0).astype(int).to_numpy()
    return X, y, cols


# --------------------------------------------------------------------------- #
# injectors

def test_inject_none_is_identity(toy):
    X, y, _ = toy
    r = inject_none(X, y)
    pd.testing.assert_frame_equal(r.X, X.reset_index(drop=True))
    assert np.array_equal(r.y, y)
    assert r.spec.drift_type == "none"
    assert not r.spec.px_changed and not r.spec.pygivenx_changed


def test_inject_covariate_touches_only_target_features(toy):
    X, y, cols = toy
    r = inject_covariate(X, y, ["f0", "f1"], scale=2.0, offset=1.5, intensity_label="strong",
                         fraction=1.0)
    # untouched features are bit-identical
    for f in ["f2", "f3", "f4", "f5"]:
        assert np.allclose(r.X[f].to_numpy(), X[f].to_numpy())
    # target features moved in spread
    assert r.X["f0"].std() > X["f0"].std() * 1.5
    assert np.array_equal(r.y, y)                       # labels untouched
    assert r.spec.px_changed and not r.spec.pygivenx_changed
    assert set(r.spec.affected_features) == {"f0", "f1"}


def test_inject_covariate_fraction_limits_affected_rows(toy):
    X, y, cols = toy
    r = inject_covariate(X, y, ["f0"], scale=3.0, offset=0.0, intensity_label="m", fraction=0.5,
                         seed=1)
    changed = ~np.isclose(r.X["f0"].to_numpy(), X["f0"].to_numpy())
    assert 0.4 < changed.mean() < 0.6


def test_inject_prior_hits_target_rate_and_preserves_conditionals(toy):
    X, y, cols = toy
    r = inject_prior(X, y, target_positive_rate=0.65, intensity_label="strong", seed=0)
    assert abs(r.y.mean() - 0.65) < 0.03
    # P(X|Y) preserved: class-conditional means are a subsample of the originals
    for cls in (0, 1):
        orig = X.to_numpy()[y == cls].mean(axis=0)
        new = r.X.to_numpy()[r.y == cls].mean(axis=0)
        assert np.allclose(orig, new, atol=0.15)
    assert not r.spec.px_changed and not r.spec.pygivenx_changed


def test_inject_concept_freezes_px_and_flips_only_the_region(toy):
    X, y, cols = toy
    r = inject_concept(X, y, region_feature="f0", region_fraction=0.3, intensity_label="m", seed=0)
    pd.testing.assert_frame_equal(r.X, X.reset_index(drop=True))   # P(X) identical
    flipped = r.y != y
    assert abs(flipped.mean() - 0.3) < 0.02
    # every flipped row is in the top of f0
    assert X["f0"].to_numpy()[flipped].min() >= np.quantile(X["f0"], 0.65)
    assert r.spec.pygivenx_changed and not r.spec.px_changed
    assert r.spec.affected_features == ["f0"]


def test_inject_label_noise_freezes_px_and_flips_about_rate(toy):
    X, y, cols = toy
    r = inject_label_noise(X, y, rate=0.2, intensity_label="strong", seed=0)
    pd.testing.assert_frame_equal(r.X, X.reset_index(drop=True))
    assert abs((r.y != y).mean() - 0.2) < 0.03
    assert r.spec.pygivenx_changed and not r.spec.affected_features


# --------------------------------------------------------------------------- #
# grading helpers

def test_localisation_prf_applicable_and_not():
    hit = localisation_prf(["AGEP", "WKHP"], ["AGEP", "WKHP"])
    assert hit["precision"] == 1.0 and hit["recall"] == 1.0 and hit["f1"] == 1.0

    partial = localisation_prf(["AGEP", "SEX"], ["AGEP", "WKHP"])
    assert partial["precision"] == 0.5 and partial["recall"] == 0.5

    na = localisation_prf(["SCHL"], [], audit_feature=None)
    assert na["applicable"] is False and na["clean"] is False
    clean = localisation_prf([], [])
    assert clean["clean"] is True


def test_estimator_error_flags_optimism():
    spec = InjectedDriftSpec(drift_type="covariate", intensity_label="s",
                             reference_accuracy=0.80, production_accuracy=0.60)
    e = estimator_error(0.78, spec)
    assert e["optimistic"] is True
    assert e["true_drop"] == pytest.approx(0.20)
    assert e["implied_drop"] == pytest.approx(0.02)


def test_label_free_accuracy_estimate_is_bounded_and_sane():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, size=2000)
    correct = (rng.uniform(size=2000) < conf).astype(float)
    est = label_free_accuracy_estimate(conf, correct, conf)
    assert 0.0 <= est <= 1.0
    assert abs(est - correct.mean()) < 0.05


def test_labelled_audit_localises_a_planted_region(toy):
    X, y, cols = toy
    model = Pipeline([("s", StandardScaler()),
                      ("c", LogisticRegression(max_iter=500, random_state=0))]).fit(X.to_numpy(), y)
    inj = inject_concept(X, y, "f0", region_fraction=0.4, intensity_label="s", seed=0)
    audit = labelled_audit(model, inj.X, inj.y, cols, reference_accuracy=0.80,
                           reference_positive_rate=float(y.mean()), n_labels=1500, seed=0)
    assert audit["measured_drop"] > 0.1
    assert audit["structured_feature"] == "f0"
    assert audit["structure_score"] > 0.15


# --------------------------------------------------------------------------- #
# classifier

def _sig(**over):
    base = dict(
        input_detector_max_severity=0, input_drift_detected=False, n_features_localised=0,
        features_localised=[], max_localised_ks_statistic=0.0, cbpe_detected=False,
        cbpe_severity_rank=0, cbpe_confidence_shift=0.0, domain_classifier_auc=0.50,
        predicted_prior_shift=0.0, estimator_implied_drop=0.0, reliability_mean_ood=0.0,
        reliability_mean_calibration_risk=0.25, reliability_suspicious_pct=5.0,
        reliability_high_risk_pct=2.0, iw_probe_recovery=0.0, feature_drop_sensitivity=0.0,
        risk_region_concentration=0.0, risk_region_feature=None, label_free_stack_silent=True,
    )
    base.update(over)
    return base


def test_classify_control():
    v = classify_drift_type(_sig(), audit={"measured_drop": 0.0, "prior_shift": 0.0,
                                            "structure_score": 0.0})
    assert v["predicted_type"] == "none"


def test_classify_covariate_label_free():
    v = classify_drift_type(_sig(
        input_detector_max_severity=4, input_drift_detected=True, n_features_localised=2,
        features_localised=["AGEP", "WKHP"], max_localised_ks_statistic=0.5,
        domain_classifier_auc=0.95, iw_probe_recovery=0.5, label_free_stack_silent=False,
    ))
    assert v["predicted_type"] == "covariate" and v["stage"] == "label-free"


def test_classify_prior_via_audit():
    v = classify_drift_type(
        _sig(predicted_prior_shift=0.14, input_drift_detected=True, input_detector_max_severity=2,
             max_localised_ks_statistic=0.1, domain_classifier_auc=0.60,
             label_free_stack_silent=False),
        audit={"measured_drop": 0.04, "prior_shift": 0.25, "structure_score": 0.05, "n_labels": 600},
    )
    assert v["predicted_type"] == "prior"


def test_classify_concept_via_audit():
    v = classify_drift_type(
        _sig(),
        audit={"measured_drop": 0.20, "prior_shift": 0.03, "structure_score": 0.35,
               "structured_feature": "SCHL", "n_labels": 600},
    )
    assert v["predicted_type"] == "concept"


def test_classify_label_noise_via_audit():
    v = classify_drift_type(
        _sig(),
        audit={"measured_drop": 0.18, "prior_shift": 0.02, "structure_score": 0.04, "n_labels": 600},
    )
    assert v["predicted_type"] == "label_noise"


# --------------------------------------------------------------------------- #
# end-to-end

@pytest.mark.integration
def test_run_benchmark_end_to_end_small(toy):
    X, y, cols = toy
    Xtr, Xref, Xp, Xe = X.iloc[:2000], X.iloc[2000:2800], X.iloc[2800:3400], X.iloc[3400:]
    ytr, yref, yp, ye = y[:2000], y[2000:2800], y[2800:3400], y[3400:]
    data = BenchmarkData(
        train_X=Xtr.reset_index(drop=True), train_y=ytr,
        reference_X=Xref.reset_index(drop=True), reference_y=yref,
        prod_pool_X=Xp.reset_index(drop=True), prod_pool_y=yp,
        eval_pool_X=Xe.reset_index(drop=True), eval_pool_y=ye,
        feature_names=cols,
    )
    # concept region + covariate features must be real columns
    import drift_autopsy.benchmark.evaluate as ev
    ev.COVARIATE_FEATURES, ev.CONCEPT_REGION_FEATURE = ["f0", "f1"], "f2"

    res = run_benchmark(data=data, drift_types=["none", "label_noise"],
                        reliability_sample=25, audit_labels=120, progress=lambda m: None)
    assert set(res.keys()) == {"config", "presentation", "runs", "summary"}
    assert len(res["runs"]) == 1 + 3          # control + 3 label-noise intensities
    for r in res["runs"]:
        assert r["verdict"]["predicted_type"] in {"none", "covariate", "prior", "concept", "label_noise"}
        assert "estimator" in r["grading"] and "localisation" in r["grading"]
    assert "type_identification" in res["summary"]
    assert "dataset_card" in res["presentation"] and "label_noise" in res["presentation"]["drift_catalog"]
