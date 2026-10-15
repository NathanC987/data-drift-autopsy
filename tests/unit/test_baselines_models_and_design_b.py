"""Tests for the second model family, the ACS harmonisation, the estimator and
detector baselines, the diagnosis baselines and ablations, the paired
statistics and the held-out (design B) injectors."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from drift_autopsy.benchmark.injectors import (
    RandomDirection,
    inject_class_noise,
    inject_concept_oblique,
    inject_selection,
    inject_threshold,
)
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.detectors import BBSD, DomainClassifier
from drift_autopsy.diagnosis import audit_diagnosis, label_free_diagnosis
from drift_autopsy.diagnosis_baselines import (
    all_rule_verdicts,
    bbse_first,
    error_rate_monitoring,
    input_monitoring,
    meta_features,
    ours,
    window_tests,
)
from drift_autopsy.estimation import atc_threshold, label_free_estimates, negative_entropy
from drift_autopsy.experiments.acs import RELSHIPP_TO_RELP, harmonize_relationship
from drift_autopsy.experiments.models import CategoryCodes, make_dense_model, make_tabular_model
from drift_autopsy.stats import macro_f1, sign_flip_test

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pums_relationship_codes.csv"


# --------------------------------------------------------------------------- #
# ACS relationship harmonisation

def _labels(variable: str):
    out = {}
    for line in FIXTURE.read_text().splitlines():
        parts = [p.strip('"') for p in line.split(",", 6)]
        if parts[1] == variable:
            out[int(parts[4])] = parts[6].strip('"')
    return out


def _words(text: str):
    return set(re.findall(r"[a-z]+", text.lower()))


def test_relshipp_mapping_matches_the_dictionary_labels():
    relp, relshipp = _labels("RELP"), _labels("RELSHIPP")
    assert set(RELSHIPP_TO_RELP) == set(relshipp)          # every 2019 code is mapped
    for new, old in RELSHIPP_TO_RELP.items():
        assert old in relp
        # each RELP label's words appear in the RELSHIPP label it is mapped from
        assert _words(relp[old]) <= _words(relshipp[new]), (new, old, relshipp[new], relp[old])
    assert 11 not in RELSHIPP_TO_RELP.values()              # "Roomer or boarder" has no successor


def test_harmonize_relationship_adds_relp_and_rejects_unknown_codes():
    raw = pd.DataFrame({"RELSHIPP": [20, 21, 24, 36]})
    assert harmonize_relationship(raw)["RELP"].tolist() == [0, 1, 13, 15]
    old = pd.DataFrame({"RELP": [0, 2]})
    assert harmonize_relationship(old) is old
    with pytest.raises(ValueError):
        harmonize_relationship(pd.DataFrame({"RELSHIPP": [99]}))


# --------------------------------------------------------------------------- #
# model families

def test_category_codes_caps_levels_and_pools_unseen():
    X = np.column_stack([np.arange(10.0), [5, 5, 5, 7, 7, 9, 9, 9, 9, 11]])
    enc = CategoryCodes(numeric_idx=[0], categorical_idx=[1], max_categories=3).fit(X)
    # the two most frequent codes (9 and 5) are kept; 7, 11 and unseen codes share "other" = 2
    out = enc.transform(np.array([[0.0, 9], [1.0, 5], [2.0, 7], [3.0, 42]]))
    assert out[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert out[:, 1].tolist() == [1.0, 0.0, 2.0, 2.0]


def test_hgb_is_deterministic_and_both_families_fit():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=600), rng.integers(0, 4, 600)]).astype(float)
    y = (X[:, 0] + (X[:, 1] == 2) + rng.normal(scale=0.5, size=600) > 0.5).astype(int)
    p1 = make_tabular_model("hgb", [0], [1], random_state=3).fit(X, y).predict_proba(X)
    p2 = make_tabular_model("hgb", [0], [1], random_state=3).fit(X, y).predict_proba(X)
    assert np.array_equal(p1, p2)
    for kind in ("lr", "hgb"):
        assert make_dense_model(kind).fit(X, y).score(X, y) > 0.7
    with pytest.raises(ValueError):
        make_tabular_model("svm", [0], [1])


# --------------------------------------------------------------------------- #
# estimator baselines

def test_label_free_estimates_on_a_known_case():
    # calibration: confidence 0.8, 80% correct; production: confidence 0.9
    cal = np.column_stack([np.full(100, 0.2), np.full(100, 0.8)])
    correct = np.r_[np.ones(80), np.zeros(20)]
    prod = np.column_stack([np.full(50, 0.1), np.full(50, 0.9)])
    est = label_free_estimates(cal, correct, prod)
    assert est["ac"] == pytest.approx(0.9)
    assert est["doc"] == pytest.approx(0.8 + 0.9 - 0.8)
    assert est["cbpe"] == pytest.approx(0.9)          # empty bin: the model's own confidence is kept
    assert est["atc"] == pytest.approx(1.0)           # every production row is more confident than t


def test_atc_threshold_matches_calibration_accuracy():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([2, 2, 2], size=2000)
    correct = rng.random(2000) < 0.7
    t = atc_threshold(negative_entropy(proba), correct)
    assert np.mean(negative_entropy(proba) > t) == pytest.approx(correct.mean(), abs=0.002)


# --------------------------------------------------------------------------- #
# detection baselines

def _ds(X, proba=None):
    return Dataset(data=pd.DataFrame(X, columns=["a", "b"]), prediction_probabilities=proba,
                   predictions=None if proba is None else proba.argmax(1))


def test_bbsd_and_domain_classifier_detect_a_strong_shift_only():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(1500, 2))
    same = rng.normal(size=(1500, 2))
    moved = rng.normal(loc=[1.0, 0.0], size=(1500, 2))
    model = LogisticRegression().fit(ref, (ref[:, 0] > 0).astype(int))
    for det in (BBSD(), DomainClassifier(random_state=0)):
        null = det.fit_detect(_ds(ref, model.predict_proba(ref)), _ds(same, model.predict_proba(same)))
        alt = det.fit_detect(_ds(ref, model.predict_proba(ref)), _ds(moved, model.predict_proba(moved)))
        assert not null.drift_detected and null.p_value > 0.01
        assert alt.drift_detected and alt.p_value < 1e-6


# --------------------------------------------------------------------------- #
# statistics

def test_sign_flip_test():
    assert sign_flip_test(np.full(30, 0.1))["p_value"] < 0.001
    res = sign_flip_test(np.zeros(30))
    assert res["p_value"] == 1.0 and res["mean_difference"] == 0.0


def test_macro_f1():
    assert macro_f1(["a", "a", "b", "b"], ["a", "a", "b", "b"], ["a", "b"]) == 1.0
    assert macro_f1(["a", "a", "b", "b"], ["a", "a", "a", "a"], ["a", "b"]) == pytest.approx((2 / 3) / 2)


# --------------------------------------------------------------------------- #
# diagnosis baselines and ablations

def _record(feature_p, prediction_p, mmd_p, residual=(), drop=False, structured=()):
    return {"feature_p": dict(zip("abc", feature_p)), "prediction_p": prediction_p, "mmd_p": mmd_p,
            "residual_rejected": list(residual), "n_features": 3, "bbse_prior_tv": 0.0, "cbpe_change": 0.0,
            "audit": {"drop_p": 0.001 if drop else 0.5, "drop_significant": drop, "accuracy_change": 0.0,
                      "error_p": {"a": 0.5}, "structured_features": list(structured)},
            "audit_variants": {}}


def test_rule_baselines_on_hand_built_records():
    quiet = _record([0.5, 0.5, 0.5], 0.5, 0.5)
    prior_like = _record([1e-6, 0.5, 0.5], 1e-9, 0.5)            # inputs and predictions moved, no residual
    covariate = _record([1e-6, 0.5, 0.5], 1e-9, 1e-4, residual=["a"])
    concept = _record([0.5, 0.5, 0.5], 0.5, 0.5, drop=True, structured=["a"])
    noise = _record([0.5, 0.5, 0.5], 0.5, 0.5, drop=True)
    assert [ours(r, 0.05) for r in (quiet, prior_like, covariate, concept, noise)] == \
        ["none", "prior", "covariate", "concept", "label_noise"]
    assert [input_monitoring(r, 0.05) for r in (quiet, prior_like, covariate, concept)] == \
        ["none", "covariate", "covariate", "none"]
    assert [bbse_first(r, 0.05) for r in (prior_like, covariate)] == ["prior", "prior"]
    assert [error_rate_monitoring(r, 0.05) for r in (concept, noise)] == ["concept", "concept"]
    only_mmd = _record([0.5, 0.5, 0.5], 0.5, 1e-4)
    v = all_rule_verdicts(only_mmd, 0.05)
    assert v["ours"] == "covariate" and v["no_mmd"] == "none"
    assert len(meta_features(covariate)) == 9


@pytest.fixture
def toy_windows():
    rng = np.random.default_rng(0)
    n = 9000
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f0", "f1", "f2"])
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * X["f0"] - X["f1"])))).astype(int)
    model = LogisticRegression().fit(X.iloc[:4000].to_numpy(), y[:4000])

    def ds(Xp, yp=None):
        proba = model.predict_proba(Xp.to_numpy())
        return Dataset(data=Xp.reset_index(drop=True), target=yp, predictions=proba.argmax(1),
                       prediction_probabilities=proba)
    return X, y, model, ds


def test_recomputed_verdict_and_precomputed_mmd_match_the_method(toy_windows):
    X, y, model, ds = toy_windows
    ref = ds(X.iloc[4000:6500], y[4000:6500])
    Xp = X.iloc[6500:].reset_index(drop=True).copy()
    Xp["f2"] += 0.3
    lf = label_free_diagnosis(ref, ds(Xp), mmd_permutations=50, random_state=1)
    lf2 = label_free_diagnosis(ref, ds(Xp), mmd_permutations=50, random_state=1,
                               precomputed_mmd=lf["multivariate_test"])
    assert lf2["multivariate_test"]["p_value"] == lf["multivariate_test"]["p_value"]
    assert lf2["verdict"] == lf["verdict"]
    au = audit_diagnosis(model, ref, Xp.iloc[:500], y[6500:7000])
    au_default = audit_diagnosis(model, ref, Xp.iloc[:500], y[6500:7000], null="confidence")
    assert au == au_default
    t = window_tests(lf, au, 0.0, audit_variants={
        "no_confidence": audit_diagnosis(model, ref, Xp.iloc[:500], y[6500:7000], null="none"),
        "class_confidence": audit_diagnosis(model, ref, Xp.iloc[:500], y[6500:7000], null="class_confidence")})
    v = all_rule_verdicts(t, 0.05)
    assert v["ours"] == "covariate" == lf["verdict"]
    assert set(v) >= {"no_confidence_null", "class_confidence_null"}


# --------------------------------------------------------------------------- #
# design B injectors

@pytest.fixture
def frame():
    rng = np.random.default_rng(2)
    n = 5000
    X = pd.DataFrame({"num": rng.normal(size=n), "cat": rng.choice([1.0, 2.0, 3.0], size=n)})
    y = (rng.random(n) < 0.4).astype(int)
    return X, y


def test_random_direction_is_a_fixed_standardised_function(frame):
    X, y = frame
    d1 = RandomDirection(X, ["num"], ["cat"], seed=5)
    d2 = RandomDirection(X, ["num"], ["cat"], seed=5)
    s = d1.score(X)
    assert np.allclose(s, d2.score(X))
    assert abs(s.mean()) < 1e-9 and abs(s.std() - 1) < 1e-9
    assert np.allclose(d1.score(X.iloc[:10]), s[:10])        # row-wise, not window-dependent


def test_selection_keeps_real_rows_with_their_labels(frame):
    X, y = frame
    d = RandomDirection(X, ["num"], ["cat"], seed=1)
    r = inject_selection(X, y, d, beta=1.5, intensity_label="strong", seed=0)
    keyed = X.assign(y=y, row=np.arange(len(X)))
    merged = r.X.assign(y=r.y).merge(keyed, on=["num", "cat"], suffixes=("", "_orig"))
    assert len(merged) == len(r.X) and (merged["y"] == merged["y_orig"]).all()
    assert d.score(r.X).mean() > 0.3                          # the kept rows lean towards high scores
    assert r.spec.drift_type == "covariate" and not r.spec.pygivenx_changed


def test_oblique_concept_threshold_and_class_noise_leave_x_untouched(frame):
    X, y = frame
    d = RandomDirection(X, ["num"], ["cat"], seed=2)
    oblique = inject_concept_oblique(X, y, d, 0.3, "moderate", seed=0)
    pd.testing.assert_frame_equal(oblique.X, X)
    assert abs((oblique.y != y).mean() - 0.3) < 0.01
    income = np.linspace(10000, 100000, len(X))
    base = (income > 50000).astype(int)
    thr = inject_threshold(X, base, income, 60000.0, "moderate")
    pd.testing.assert_frame_equal(thr.X, X)
    assert ((thr.y == 1) <= (base == 1)).all() and thr.y.mean() < base.mean()
    noise = inject_class_noise(X, y, 0.2, "moderate", seed=0)
    pd.testing.assert_frame_equal(noise.X, X)
    assert ((noise.y != y) <= (y == 1)).all()                 # only positives are flipped
    assert abs((noise.y != y)[y == 1].mean() - 0.2) < 0.03


# --------------------------------------------------------------------------- #
# GPU permutation tests give the same decisions as the CPU path

def test_gpu_permutation_tests_match_cpu(monkeypatch):
    from drift_autopsy import gpu
    from drift_autopsy.detectors import MMD, FrechetDistance, PCAReconstructionError

    if not gpu.gpu_available():
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(4)
    loadings = rng.normal(size=(4, 30))            # low-rank structure, so PCA keeps fewer than 30 axes

    def sample(n, shift=0.0):
        Z = rng.normal(size=(n, 4)) @ loadings + 0.3 * rng.normal(size=(n, 30))
        Z[:, 0] += shift
        return pd.DataFrame(Z, columns=[f"e{i}" for i in range(30)])

    ref, prod = Dataset(data=sample(600)), Dataset(data=sample(500, shift=0.4))
    for make in (lambda: MMD(n_permutations=99, random_state=3),
                 lambda: FrechetDistance(n_permutations=99, random_state=3),
                 lambda: PCAReconstructionError(n_permutations=99, random_state=3)):
        on_gpu = make().fit_detect(ref, prod)
        monkeypatch.setenv("DRIFT_AUTOPSY_DEVICE", "cpu")
        on_cpu = make().fit_detect(ref, prod)
        monkeypatch.delenv("DRIFT_AUTOPSY_DEVICE")
        assert on_gpu.metadata["device"] == "cuda" and on_cpu.metadata["device"] == "cpu"
        assert on_gpu.p_value == on_cpu.p_value
        assert on_gpu.statistic == pytest.approx(on_cpu.statistic, rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------- #
# sweep fixes: sparse CBPE bins, fusion events rule, input validation

def test_cbpe_keeps_own_confidence_in_sparse_bins():
    from drift_autopsy.estimation import MIN_BIN_COUNT, BinnedAccuracyCalibration

    conf = np.r_[np.full(200, 0.95), np.full(2, 0.55)]          # the 0.5-0.6 bin holds only 2 rows
    correct = np.r_[np.ones(180), np.zeros(20), np.ones(2)]
    cal = BinnedAccuracyCalibration.fit(conf, correct, 10)
    assert cal.bin_counts[5] < MIN_BIN_COUNT and not cal.reliable[5] and cal.reliable[9]
    q = cal.predict_correct(np.array([0.55, 0.35, 0.95]))
    assert q[0] == pytest.approx(0.55) and q[1] == pytest.approx(0.35)   # own confidence where data are silent
    assert q[2] == pytest.approx(0.9)                                    # binned accuracy where data exist


def test_fusion_falls_back_without_enough_errors():
    from drift_autopsy.reliability.fusion import MIN_EVENTS_PER_SIGNAL, RiskFusion

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.random((400, 3)), columns=["confidence_risk", "ood", "instability"])
    frame["calibrated_error"] = rng.random(400)
    few = np.zeros(400, dtype=int)
    few[:MIN_EVENTS_PER_SIGNAL * 3 - 1] = 1
    f = RiskFusion(["confidence_risk", "ood", "instability"]).fit(frame, few)
    assert not f.learned and f.coefficients() == {}
    assert np.array_equal(f.predict(frame), frame["confidence_risk"].to_numpy())
    many = (rng.random(400) < 0.3).astype(int)
    assert RiskFusion(["confidence_risk", "ood", "instability"]).fit(frame, many).learned


def test_dataset_and_detector_options_are_validated():
    from drift_autopsy.detectors import PSI, KSTest

    with pytest.raises(ValueError):
        Dataset(data=pd.DataFrame({"a": [1, 2, 3]}), target=np.array([0, 1]))
    with pytest.raises(ValueError):
        PSI(aggregate_method="median")
    with pytest.raises(ValueError):
        KSTest(correction="fdr")


def test_shap_importances_are_reproducible_above_ten_features():
    # Above 10 features shap uses its random permutation algorithm; the seed must reach it.
    from sklearn.linear_model import LogisticRegression

    from drift_autopsy.rca.shap_analyzer import SHAPAnalyzer

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 16))
    model = LogisticRegression().fit(X, (X[:, 0] + X[:, 1] > 0).astype(int))
    cols = [f"f{i}" for i in range(16)]
    ref = Dataset(data=pd.DataFrame(X[:200], columns=cols))
    test = Dataset(data=pd.DataFrame(X[200:] + 0.3, columns=cols))
    runs = [SHAPAnalyzer(n_background_samples=30, n_test_samples=60, n_bootstrap=50)
            .analyze(ref, test, None, model=model).feature_importances for _ in range(2)]
    assert runs[0] == runs[1]
