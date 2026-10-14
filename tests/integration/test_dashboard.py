"""Dashboard smoke tests against the generated study outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("examples/dashboard").resolve()))
import loaders as L  # noqa: E402

pytestmark = pytest.mark.integration


def _need(name):
    data = L.load(L.DEFAULT_PATHS[name])
    if data is None:
        pytest.skip(f"{L.DEFAULT_PATHS[name]} not generated")
    return data


def test_window_tables_for_both_real_studies():
    for name in ("acs", "clear"):
        study = _need(name)
        summary = L.window_summary(study)
        assert len(summary) == len(study["windows"])
        assert not L.detector_table(study).empty
        assert not L.reliability_summary(study).empty


def test_acs_localization_and_shap_tables():
    acs = _need("acs")
    assert not L.acs_localization(acs).empty
    assert not L.acs_shap(acs, next(iter(acs["windows"]))).empty


def test_benchmark_and_remediation_tables():
    bench = _need("benchmark")
    assert L.benchmark_confusion(bench).values.sum() == bench["summary"]["n_runs"]
    assert not L.benchmark_types(bench).empty
    rem = _need("remediation")
    assert not L.remediation_table(rem).empty


def test_app_renders_without_exceptions():
    pytest.importorskip("streamlit")
    for name in ("acs", "clear", "benchmark", "remediation", "concepts"):
        _need(name)
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("examples/dashboard/app.py", default_timeout=120)
    at.run()
    assert len(at.exception) == 0, [e.value for e in at.exception]
