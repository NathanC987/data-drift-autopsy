"""Remediation triage: decide whether retraining is worth it, cheaply, first.

Rather than a heuristic verdict, triage *runs the two cheapest label-free
strategies* -- importance-weighted adaptation and (if a recent window exists)
retrain-on-recent -- as probes, and escalates to the expensive options (full
retrain, collecting production labels) only if a probe already recovered a
meaningful slice of the gap. It also reports the a-priori signals that explain
the verdict:

* ``domain_classifier_auc`` -- reference/production separability. Near 0.5 means
  little covariate movement.
* ``effective_sample_fraction`` -- fraction of the reference that behaves like
  production under the density ratio.
* ``feature_drop_sensitivity`` -- change in held-out reference accuracy when the
  localised features are removed. Near 0 => those features do not carry the
  decision, so feature-scoped surgery on them is a no-op (or, if strongly
  negative, actively harmful).

A separable shift (high AUC) is *not* sufficient: the CA->state ACS shift is
highly separable yet irreducible, because the label rule itself differs.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from drift_autopsy.remediation.base import RemediationContext
from drift_autopsy.remediation.importance_weighting import density_ratio_weights
from drift_autopsy.remediation import strategies as S

ESCALATE_GAP_FRACTION = 0.25


def _feature_drop_sensitivity(ctx: RemediationContext, seed: int = 42) -> float:
    drop_idx = set(ctx.drifted_index.tolist())
    keep = [i for i in range(len(ctx.feature_names)) if i not in drop_idx]
    if not keep or not drop_idx:
        return 0.0
    Xtr, Xte, ytr, yte = train_test_split(
        ctx.reference_X, ctx.reference_y, test_size=0.3, random_state=seed, stratify=ctx.reference_y
    )
    full = ctx.model_factory().fit(Xtr, ytr)
    dropped = ctx.model_factory().fit(Xtr[:, keep], ytr)
    return float(
        accuracy_score(yte, full.predict(Xte)) - accuracy_score(yte, dropped.predict(Xte[:, keep]))
    )


def remediation_triage(
    ctx: RemediationContext,
    estimated_gap: float | None = None,
) -> Dict[str, Any]:
    """Run the cheap label-free probes and return signals + an escalation verdict."""
    _, diag = density_ratio_weights(ctx.reference_X, ctx.production_X)
    sensitivity = _feature_drop_sensitivity(ctx)

    probes: Dict[str, float] = {}
    try:
        probes["importance_weighted_retrain"] = S.importance_weighted_retrain(ctx).fraction_of_gap_recovered
    except Exception:
        pass
    if ctx.recent_windows:
        try:
            probes["retrain_on_recent"] = S.retrain_on_recent(ctx, "previous").fraction_of_gap_recovered
        except Exception:
            pass

    best_probe = max(probes.values(), default=0.0)
    escalate = best_probe >= ESCALATE_GAP_FRACTION

    signals = {
        "domain_classifier_auc": diag["domain_classifier_auc"],
        "effective_sample_fraction": diag["effective_sample_fraction"],
        "feature_drop_sensitivity": sensitivity,
        "estimated_gap": None if estimated_gap is None else float(estimated_gap),
        "observed_gap": float(ctx.reference_accuracy - ctx.baseline_before)
        if ctx.reference_accuracy is not None
        else None,
        "cheap_probe_recovery": probes,
    }

    if escalate:
        best_name = max(probes, key=probes.get)
        rationale = (
            f"the cheap probe '{best_name}' already recovered {best_probe:.0%} of the gap "
            f"(AUC {diag['domain_classifier_auc']:.2f}) -- retraining is worth escalating"
        )
    elif diag["domain_classifier_auc"] < 0.55:
        rationale = (
            f"reference and production are barely separable (AUC {diag['domain_classifier_auc']:.2f}) "
            f"and no cheap probe recovered more than {best_probe:.0%} of the gap"
        )
    else:
        rationale = (
            f"the shift is separable (AUC {diag['domain_classifier_auc']:.2f}) but no cheap probe "
            f"recovered more than {best_probe:.0%} of the gap -- the loss is likely concept-level, "
            f"not covariate; collecting production labels or investigating the label rule comes "
            f"before a full retrain"
        )

    return {"will_retraining_help": bool(escalate), "rationale": rationale, "signals": signals}
