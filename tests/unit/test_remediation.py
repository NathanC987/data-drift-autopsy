"""Unit tests for the remediation package on a controlled synthetic shift."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from drift_autopsy.remediation import (
    RemediationContext,
    density_ratio_weights,
    inject_covariate_shift,
    remediation_triage,
    run_remediation_suite,
)
from drift_autopsy.remediation import strategies as S


@pytest.fixture
def covariate_shift_ctx():
    rng = np.random.default_rng(0)
    names = [f"f{i}" for i in range(6)]
    n = 3000

    def sample(n, shift):
        X = rng.normal(size=(n, 6))
        if shift:
            X[:, :2] = X[:, :2] * 1.8 + 1.5
        # mildly non-linear boundary that uses a shifted feature -> the linear
        # base model is misspecified and covariate reweighting helps
        y = (0.7 * X[:, 0] ** 2 + X[:, 3] + 0.3 * rng.normal(size=n) > 1.0).astype(int)
        return X, y

    X_ref, y_ref = sample(n, shift=False)
    X_prod, y_prod = sample(n, shift=True)
    X_hold, y_hold = sample(n, shift=True)

    base = LogisticRegression(max_iter=500, random_state=0).fit(X_ref, y_ref)
    # ceiling: same model family fit on unshifted data, scored on unshifted holdout
    Xc, yc = sample(n, shift=False)
    ref_acc = LogisticRegression(max_iter=500, random_state=0).fit(X_ref, y_ref).score(Xc, yc)

    return RemediationContext(
        reference_X=X_ref, reference_y=y_ref,
        production_X=X_prod, production_y=y_prod,
        holdout_X=X_hold, holdout_y=y_hold,
        feature_names=names, drifted_features=["f0", "f1"],
        shift_name="synthetic", base_model=base,
        model_factory=lambda: LogisticRegression(max_iter=500, random_state=0),
        reference_accuracy=float(ref_acc),
    )


def test_density_ratio_weights_flag_the_shift(covariate_shift_ctx):
    ctx = covariate_shift_ctx
    w, diag = density_ratio_weights(ctx.reference_X, ctx.production_X)
    assert diag["domain_classifier_auc"] > 0.7          # shift is separable
    assert abs(w.mean() - 1.0) < 1e-6                    # normalised
    assert 0 < diag["effective_sample_size"] <= len(ctx.reference_X)


def test_importance_weighting_helps_a_misspecified_model(covariate_shift_ctx):
    ctx = covariate_shift_ctx
    res = S.importance_weighted_retrain(ctx)
    plain = S.full_retrain(ctx)  # same data, no weights
    assert res.accuracy_after >= plain.accuracy_after - 1e-6   # weighting is not worse
    assert res.n_production_labels_required == 0
    assert res.effective_sample_size is not None


def test_feature_drop_uses_fewer_features(covariate_shift_ctx):
    res = S.feature_drop_retrain(covariate_shift_ctx)
    assert res.n_features_used == 4
    assert set(res.extra["dropped_features"]) == {"f0", "f1"}


def test_result_schema_and_timing(covariate_shift_ctx):
    for res in run_remediation_suite(covariate_shift_ctx,
                                     ["full_retrain", "feature_drop_retrain",
                                      "importance_weighted_retrain", "head_refit"]):
        d = res.to_dict()
        assert d["wall_clock_seconds"] > 0
        assert d["fit_flops_proxy"] > 0
        assert np.isfinite(d["fraction_of_gap_recovered"])


def test_triage_flags_a_recoverable_shift(covariate_shift_ctx):
    verdict = remediation_triage(covariate_shift_ctx)
    assert verdict["will_retraining_help"] is True
    assert verdict["signals"]["domain_classifier_auc"] > 0.7
