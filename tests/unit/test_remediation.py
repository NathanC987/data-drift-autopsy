"""Unit tests for remediation strategies and the label-free triage."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from drift_autopsy.remediation import (
    NotApplicable,
    RemediationContext,
    density_ratio_weights,
    remediation_triage,
    run_remediation_suite,
)
from drift_autopsy.remediation import strategies as S


def _factory():
    return LogisticRegression(max_iter=1000)


def _sample(rng, n, covariate=False, concept=False):
    X = rng.normal(size=(n, 4))
    if covariate:
        X[:, 0] = X[:, 0] * 1.8 + 1.5
    # quadratic boundary: the linear model is misspecified, so reweighting toward
    # the production region of x0 changes the best linear fit
    score = 0.7 * X[:, 0] ** 2 + X[:, 1] + 0.3 * rng.normal(size=n)
    y = (score > 1.0).astype(int)
    if concept:
        y = np.where(X[:, 1] > 0.5, 1 - y, y)
    return X, y


def _ctx(covariate=False, concept=False, seed=0):
    rng = np.random.default_rng(seed)
    X_ref, y_ref = _sample(rng, 4000)
    X_val, y_val = _sample(rng, 2000)
    X_prod, y_prod = _sample(rng, 3000, covariate, concept)
    X_hold, y_hold = _sample(rng, 3000, covariate, concept)
    base = _factory().fit(X_ref, y_ref)
    return RemediationContext(
        reference_X=X_ref, reference_y=y_ref, production_X=X_prod, production_y=y_prod,
        holdout_X=X_hold, holdout_y=y_hold, feature_names=["a", "b", "c", "d"],
        drifted_features=["a"], shift_name="synthetic", base_model=base, model_factory=_factory,
        reference_accuracy=float(base.score(X_val, y_val)), seed=seed,
    )


def test_density_ratio_weights_are_normalised():
    rng = np.random.default_rng(1)
    w, diag = density_ratio_weights(rng.normal(size=(1500, 3)), rng.normal(loc=0.5, size=(1500, 3)))
    assert abs(w.mean() - 1.0) < 1e-9
    assert 0 < diag["effective_sample_fraction"] <= 1
    assert diag["domain_classifier_auc"] > 0.55


def test_triage_escalates_on_correctable_covariate_shift_and_iw_recovers():
    ctx = _ctx(covariate=True)
    triage = remediation_triage(ctx)
    assert triage["will_retraining_help"]
    result = S.importance_weighted_retrain(ctx)
    assert result.recovery_significant and result.accuracy_recovered > 0


def test_triage_holds_on_concept_shift():
    triage = remediation_triage(_ctx(concept=True))
    assert not triage["will_retraining_help"]


def test_strategies_that_need_missing_data_are_skipped_not_faked():
    ctx = _ctx()
    with pytest.raises(NotApplicable):
        S.full_retrain(ctx)
    out = run_remediation_suite(ctx, ["retrain_on_recent", "importance_weighted_retrain"])
    assert "retrain_on_recent" in out["skipped"]
    assert [r.strategy for r in out["results"]] == ["importance_weighted_retrain"]


def test_strategies_are_deterministic():
    ctx = _ctx(covariate=True)
    a = S.importance_weighted_retrain(ctx).accuracy_after
    b = S.importance_weighted_retrain(ctx).accuracy_after
    assert a == b
