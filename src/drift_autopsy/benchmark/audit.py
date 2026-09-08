"""A small delayed-label audit.

Concept drift and unstructured label noise are provably invisible to a
label-free monitor when the inputs and the model are frozen: nothing the model
emits has changed. The realistic escape hatch is a handful of freshly labelled
production rows arriving late. This module measures, from a few hundred labels,
how far accuracy fell, whether the class prior moved, and whether the loss is
concentrated in one feature's range (concept) or spread evenly (noise).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def labelled_audit(
    model: Any,
    production_X: pd.DataFrame,
    production_y: np.ndarray,
    feature_names: List[str],
    reference_accuracy: float,
    reference_positive_rate: float = 0.0,
    n_labels: int = 600,
    seed: int = 42,
) -> Dict[str, Any]:
    """Score ``n_labels`` random production rows and characterise the loss."""
    feature_names = list(feature_names)
    production_X = production_X.reset_index(drop=True)[feature_names]
    production_y = np.asarray(production_y).astype(int)
    rng = np.random.default_rng(seed)

    n = min(n_labels, len(production_X))
    idx = np.sort(rng.choice(len(production_X), n, replace=False))
    Xs = production_X.iloc[idx]
    ys = production_y[idx]
    preds = model.predict(Xs.to_numpy(dtype=float))
    correct = (preds == ys).astype(float)

    measured_acc = float(accuracy_score(ys, preds))
    measured_drop = float(reference_accuracy - measured_acc)
    audit_positive_rate = float(np.mean(ys))

    per_feature_gap: Dict[str, float] = {}
    for f in feature_names:
        col = Xs[f].to_numpy(dtype=float)
        med = float(np.median(col))
        hi = col > med
        lo = ~hi
        if hi.sum() < 15 or lo.sum() < 15:
            continue
        per_feature_gap[f] = float(abs(correct[lo].mean() - correct[hi].mean()))

    gaps = np.array(list(per_feature_gap.values()), dtype=float)
    structured_feature = max(per_feature_gap, key=per_feature_gap.get, default=None)
    max_gap = per_feature_gap.get(structured_feature, 0.0)
    # excess concentration over the per-feature noise floor
    structure_score = float(max_gap - np.median(gaps)) if gaps.size else 0.0

    return {
        "n_labels": int(n),
        "measured_accuracy": round(measured_acc, 4),
        "measured_drop": round(measured_drop, 4),
        "audit_positive_rate": round(audit_positive_rate, 4),
        "prior_shift": round(audit_positive_rate - float(reference_positive_rate), 4),
        "structured_feature": structured_feature,
        "max_feature_gap": round(float(max_gap), 4),
        "structure_score": round(structure_score, 4),
        "per_feature_accuracy_gap": {k: round(v, 4) for k, v in per_feature_gap.items()},
    }
