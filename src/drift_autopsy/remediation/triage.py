"""Remediation triage: predict, without fresh production labels, whether a cheap
retrain will raise production accuracy.

Two cheap probes are trained and each is validated on the most relevant
labelled data the team already holds:

* importance-weighted retrain. The reference is split into a fit part and a
  validation part. Density-ratio weights w(x) ~ p_prod(x) / p_ref(x) come from a
  cross-fitted domain classifier. The probe is trained on the fit part with
  these weights; its production accuracy is estimated by importance-weighted
  validation (Sugiyama, Krauledat and Mueller, 2007): the w-weighted accuracy
  on the validation part. This estimate is unbiased under covariate shift and
  blind to changes in P(Y | X).
* retrain on the most recent delayed-label window (if one exists). The probe is
  trained on the reference fit part plus 70% of that window and validated on the
  other 30% of it, which assumes production resembles the latest labelled window.

Each probe is compared with a comparator trained the same way without the
remediation (unweighted fit part; fit part only), on the same validation rows,
by a paired bootstrap of the accuracy difference. The triage escalates when at
least one probe's predicted gain is significantly positive, with Holm
correction across probes at level ``alpha``.

The production holdout labels are never used here; demos use them afterwards
only to grade the verdict.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import train_test_split

from drift_autopsy.remediation.base import RemediationContext, fit_model
from drift_autopsy.remediation.importance_weighting import density_ratio_weights
from drift_autopsy.stats import holm_reject, paired_accuracy_difference

VALIDATION_FRACTION = 0.3
PROBE_STRATEGIES = ("importance_weighted_retrain", "retrain_on_recent")   # the strategies the probes stand for


def _correct(model: Any, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (model.predict(X) == y).astype(int)


def remediation_triage(ctx: RemediationContext, alpha: float = 0.05, n_bootstrap: int = 2000) -> Dict[str, Any]:
    """Run the probes and return the verdict with its evidence (see module docstring)."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        ctx.reference_X, ctx.reference_y, test_size=VALIDATION_FRACTION,
        random_state=ctx.seed, stratify=ctx.reference_y,
    )
    weights, diag = density_ratio_weights(np.vstack([X_fit, X_val]), ctx.production_X, random_state=ctx.seed)
    w_fit, w_val = weights[: len(X_fit)], weights[len(X_fit):]

    probes: List[Dict[str, Any]] = []

    comparator, _, _ = fit_model(ctx.model_factory, X_fit, y_fit, repeats=1)
    iw_model, _, _ = fit_model(ctx.model_factory, X_fit, y_fit, sample_weight=w_fit, repeats=1)
    iw = paired_accuracy_difference(
        _correct(comparator, X_val, y_val), _correct(iw_model, X_val, y_val),
        weights=w_val, n_bootstrap=n_bootstrap, alpha=alpha, random_state=ctx.seed,
    )
    probes.append({
        "probe": "importance_weighted_retrain",
        "validation": "importance-weighted reference validation",
        "estimated_comparator_accuracy": float(np.average(_correct(comparator, X_val, y_val), weights=w_val)),
        **iw,
    })

    if ctx.recent_windows:
        X_new, y_new = ctx.recent_windows[-1]
        X_nf, X_nv, y_nf, y_nv = train_test_split(
            X_new, y_new, test_size=VALIDATION_FRACTION, random_state=ctx.seed, stratify=y_new,
        )
        recent_model, _, _ = fit_model(ctx.model_factory, np.vstack([X_fit, X_nf]), np.concatenate([y_fit, y_nf]),
                                       repeats=1)
        rec = paired_accuracy_difference(
            _correct(comparator, X_nv, y_nv), _correct(recent_model, X_nv, y_nv),
            n_bootstrap=n_bootstrap, alpha=alpha, random_state=ctx.seed,
        )
        probes.append({
            "probe": "retrain_on_recent",
            "validation": "held-out part of the latest delayed-label window",
            "estimated_comparator_accuracy": float(_correct(comparator, X_nv, y_nv).mean()),
            **rec,
        })

    rejected = holm_reject([p["p_value_greater"] for p in probes], alpha)
    for p, rej in zip(probes, rejected):
        p["predicted_gain_significant"] = bool(rej)
    escalate = bool(np.any(rejected))

    if escalate:
        best = max((p for p in probes if p["predicted_gain_significant"]), key=lambda p: p["difference"])
        rationale = (f"probe '{best['probe']}' is predicted to raise accuracy by {best['difference']:+.4f} "
                     f"(95% CI {best['ci_low']:+.4f} to {best['ci_high']:+.4f}); escalate to retraining")
    else:
        rationale = ("no cheap probe is predicted to raise accuracy significantly; if accuracy is falling the "
                     "change is likely in P(Y|X), so collect fresh labels before retraining")

    return {
        "will_retraining_help": escalate,
        "rationale": rationale,
        "probes": probes,
        "signals": {
            "domain_classifier_auc": diag["domain_classifier_auc"],
            "effective_sample_fraction": diag["effective_sample_fraction"],
        },
        "alpha": alpha,
    }
