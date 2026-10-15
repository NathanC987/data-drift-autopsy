"""Dashboard tests against the generated study outputs.

Every loader must return a usable frame, every chart builder a figure, and
every computed sentence must be free of missing values; the app must render
all four tabs without an exception.
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path("examples/dashboard").resolve()))
import charts as C  # noqa: E402
import loaders as L  # noqa: E402
import narrative as N  # noqa: E402

pytestmark = pytest.mark.integration


def _need(name):
    data = L.load(L.DEFAULT_PATHS[name])
    if data is None:
        pytest.skip(f"{L.DEFAULT_PATHS[name]} not generated")
    return data


def _need_path(path):
    data = L.load(path)
    if data is None:
        pytest.skip(f"{path} not generated")
    return data


def _clean(text: str) -> None:
    assert text and "nan" not in text.lower().split() and "None" not in text, text


@pytest.mark.parametrize("name", ["acs", "clear", "gas"])
def test_window_study_loaders_charts_and_text(name):
    study = _need(name)
    summary = L.window_summary(study)
    assert len(summary) == len(study["windows"])
    assert (summary["acc_low"] <= summary["accuracy"]).all() and (summary["accuracy"] <= summary["acc_high"]).all()
    focus = L.focus_window(summary)
    label = summary.set_index("key").loc[focus, "window"]
    det = L.detector_table(study)
    rel = L.reliability_summary(study)
    coefs = L.fusion_coefficients(study)
    figs = [
        C.split_bar(L.split_sizes(study), "reference"), C.window_sizes(summary, "rows"),
        C.accuracy_with_ci(summary, study["reference"]["test_accuracy"]),
        C.estimate_vs_measured(summary, study["reference"]["test_accuracy"]), C.estimate_error(summary),
        C.confidence_hist(L.confidence_histogram(study, focus), label),
        C.detector_small_multiples(det, L.DETECTOR_STATISTIC, study["config"]["psi_threshold"]), C.alarm_matrix(det),
        C.decision_matrix(L.decision_path(study)), C.auroc_dots(rel), C.risk_shares(rel), C.risk_error_rates(rel),
        C.risk_hist(L.reliability_records(study, focus), study["reliability_model"]["fusion"]["label_thresholds"], label),
        C.fusion_bars(coefs),
    ]
    assert all(isinstance(f, go.Figure) for f in figs)
    for text in (N.accuracy(summary, study["config"]["alpha"]), N.proxy(summary), N.detection(det),
                 N.confidence(L.cbpe_test(study, focus), label), N.diagnosis(summary), N.reliability(rel, coefs)):
        _clean(text)


def test_decision_path_matches_the_verdicts():
    for name in ("acs", "clear"):
        study = _need(name)
        dp = L.decision_path(study)
        for key, g in dp.groupby("key"):
            a = dict(zip(g["question"], g["answer"]))
            verdict = g["verdict"].iloc[0]
            label_free = {a["feature tests reject"], a["MMD rejects"], a["predictions shifted"]}
            if verdict in ("covariate", "prior"):
                assert "yes" in label_free
            if verdict in ("concept", "label noise"):
                assert a["audit: accuracy fell"] == "yes"
                assert a["audit: errors depend on a feature"] == ("yes" if verdict == "concept" else "no")


def test_acs_specific_views():
    acs = _need("acs")
    loc = L.acs_localization(acs)
    assert set(loc["test"]) == {"numeric", "categorical"}
    focus = L.focus_window(L.window_summary(acs))
    shap = L.acs_shap(acs, focus)
    figs = [C.feature_heatmaps(loc), C.shap_heatmap(L.acs_shap_all(acs)), C.shap_dumbbell(shap, focus),
            C.prior_lines(L.prior_rates(acs))]
    assert all(isinstance(f, go.Figure) for f in figs)
    for text in (N.acs_localization(loc, focus), N.unseen_overall(loc), N.shap(shap, focus),
                 N.prior(L.prior_rates(acs), focus)):
        _clean(text)


def test_clear_specific_views():
    clear = _need("clear")
    loc, slices = L.clear_localization(clear), L.clear_class_slices(clear)
    focus = L.focus_window(L.window_summary(clear))
    assert isinstance(C.class_slice_heatmap(slices), go.Figure)
    assert isinstance(C.classwise_f1(L.clear_classwise_f1(clear, focus), focus), go.Figure)
    _clean(N.clear_localization(loc, slices, focus))
    proj = L.clear_projection(clear)
    if proj is not None:
        assert len(proj["explained"]) == 3 and 0 < sum(proj["explained"]) < 1
        assert isinstance(C.pca_points(proj, 10), go.Figure) and isinstance(C.pca_centroids(proj, 10), go.Figure)
        _clean(N.pca(proj))


def test_gas_specific_views():
    gas = _need("gas")
    sl = L.gas_sensor_localization(gas)
    assert sl["sensor"].nunique() == gas["config"]["n_sensors"]
    assert isinstance(C.sensor_heatmap(sl), go.Figure)
    comp = L.gas_composition(gas)
    assert comp.groupby("window")["share"].sum().round(6).eq(1).all()
    assert isinstance(C.composition_bars(comp), go.Figure)


def test_year_gap_breaks_the_acs_lines():
    acs = _need("acs")
    gaps = L.gap_before(acs)
    years = [w.get("year") for w in acs["windows"].values() if w["analysis_type"] == "temporal"]
    if 2019 in years and 2021 in years:
        assert gaps["CA 2021"] and not gaps["CA 2019"]
    assert sum(gaps.values()) == sum(1 for a, b in zip(years, years[1:]) if b - a > 1)


def test_concept_views():
    concepts = _need("concepts")
    val = L.concept_validation(concepts)
    b = str(concepts["config"]["figure_bucket"])
    v = concepts["buckets"][b]["metadata_validation"]
    assert isinstance(C.concept_tag_scatter(L.concept_tag_pairs(concepts, b), v["spearman_rho"], v["p_value"], b), go.Figure)
    _clean(N.concepts(val, concepts["config"]["alpha"]))


def test_remediation_views():
    rem = _need("remediation")
    for groups in (("acs_temporal", "acs_geographic"), ("clear10",)):
        table, card = L.remediation_table(rem, groups), L.triage_scorecard(rem, groups)
        assert not table.empty and not card.empty
        assert table.loc[~table["applicable"], "reason"].str.len().gt(0).all()
        assert isinstance(C.remediation_heatmap(table), go.Figure)
        assert isinstance(C.probe_scatter(L.triage_probes(rem, groups)), go.Figure)
        _clean(N.remediation(table, card))


@pytest.mark.parametrize("design", ["A", "B"])
def test_benchmark_views(design):
    bench = _need_path(L.benchmark_path("acs_income", design, "lr"))
    runs = L.benchmark_runs(bench)
    assert len(runs) == bench["summary"]["n_runs"]
    assert L.benchmark_confusion(bench).values.sum() == bench["summary"]["n_runs"]
    rates = L.benchmark_alarm_rates(runs)
    assert rates["rate"].between(0, 1).all()
    bt = L.benchmark_types(bench)
    types = [t for t in L.TYPE_ORDER if t in set(runs["injected"])]
    figs = [C.bench_true_drop(runs, types), C.bench_alarm_heatmap(rates, types), C.bench_estimate_scatter(runs, types),
            C.bench_per_intensity(L.benchmark_per_intensity(bench), types),
            C.stage_share(L.benchmark_stage_share(bench), types), C.bench_reliability_gain(bt),
            C.bench_triage_strip(runs, types), C.triage_confusion(bench["summary"]["triage"]["confusion"])]
    assert all(isinstance(f, go.Figure) for f in figs)
    texts = [N.bench_design(bench), N.bench_ground_truth(runs), N.bench_estimator(bt), N.bench_type_id(bench),
             N.bench_audit(L.benchmark_audit_budget(bench)), N.bench_reliability(bt), N.bench_triage(bench)]
    if design == "A":
        figs.append(C.false_alarms(L.benchmark_false_alarms(bench), 0.05))
        texts.append(N.bench_localization(bench))
    assert all(isinstance(f, go.Figure) for f in figs)
    for text in texts:
        _clean(text)
    assert len(L.injection_recipes(bench)) == (design == "A") + sum(
        len(v) for v in bench["config"]["intensity_ladder"].values())
    est = L.benchmark_estimators(bench)
    assert set(est["key"]) == {"cbpe", "ac", "doc", "atc"}
    assert isinstance(C.estimator_mae(est), go.Figure)
    _clean(N.bench_estimators(est))
    summary = L.load(L.DEFAULT_PATHS["benchmark_summary"])
    if summary is not None:
        meth = L.benchmark_methods(summary, f"acs_income_{design}_lr")
        assert "ours" in set(meth["key"]) and meth["accuracy"].between(0, 1).all()
        assert isinstance(C.method_dots(meth), go.Figure)
        _clean(N.bench_methods(meth))
        if design == "B":
            mech = L.benchmark_mechanisms(summary, "acs_income", "lr")
            assert isinstance(C.mechanism_lines(mech), go.Figure)


def test_app_renders_without_exceptions():
    pytest.importorskip("streamlit")
    for name in ("acs", "clear", "gas", "benchmark", "remediation", "concepts"):
        _need(name)
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("examples/dashboard/app.py", default_timeout=300)
    at.run()
    assert len(at.exception) == 0, [e.value for e in at.exception]
    assert len(at.tabs) == 4
