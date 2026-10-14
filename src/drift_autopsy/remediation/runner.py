"""Run a set of remediation strategies over one context."""

from __future__ import annotations

from typing import Dict, List, Sequence

import pandas as pd

from drift_autopsy.remediation.base import RemediationContext, RemediationResult
from drift_autopsy.remediation.strategies import STRATEGIES, NotApplicable


def run_remediation_suite(
    ctx: RemediationContext,
    strategies: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Run each named strategy; strategies that do not apply are listed with the reason."""
    names = list(strategies) if strategies is not None else list(STRATEGIES)
    results: List[RemediationResult] = []
    skipped: Dict[str, str] = {}
    for name in names:
        if name not in STRATEGIES:
            raise KeyError(f"unknown strategy {name!r}; known: {sorted(STRATEGIES)}")
        try:
            results.append(STRATEGIES[name](ctx))
        except NotApplicable as exc:
            skipped[name] = str(exc)
    return {"results": results, "skipped": skipped}


def results_to_frame(results: Sequence[RemediationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in results])
