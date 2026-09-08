"""Build the self-contained explanatory payload for the benchmark dashboard.

A reader opening the benchmark view knows nothing about ACS Income, the model,
or the pipeline. This module assembles everything the view needs to explain
itself from first principles: a dataset card, a plain-language card per drift
type with the exact recipe used, and the raw material for the before/after
visuals (feature histograms, class balance, and model accuracy sliced by the
feature the drift targets).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

FEATURE_GLOSSARY: Dict[str, str] = {
    "AGEP": "age, in years",
    "COW": "class of worker (private company, government, self-employed, ...)",
    "SCHL": "educational attainment, ordinal from no schooling up to a doctorate",
    "MAR": "marital status",
    "OCCP": "occupation, a numeric census code",
    "POBP": "place of birth, a country or US-state code",
    "RELP": "relationship to the householder",
    "WKHP": "usual hours worked per week",
    "SEX": "sex",
    "RAC1P": "race, a single-race recode",
}

# focus feature for the before/after visual; None => show the label-flip breakdown
FOCUS_FEATURE: Dict[str, Optional[str]] = {
    "covariate": "AGEP",
    "prior": "SCHL",
    "concept": "SCHL",
    "label_noise": None,
}


def _drift_cards(ref_positive_rate: float) -> Dict[str, Dict[str, str]]:
    pct = f"{ref_positive_rate:.0%}"
    return {
        "covariate": {
            "title": "Covariate shift - the population changes, the rule does not",
            "plain": (
                "The mix of people being scored moves, but the relationship between someone's "
                "attributes and their income is unchanged. A model built on one workforce is now "
                "scoring a different one - an ageing region, a new market segment."
            ),
            "moves": "P(X): the input distribution",
            "keeps": "P(Y|X): the income rule is untouched",
            "recipe": (
                "For a random subset of production rows, two features were linearly rescaled - "
                "x' = mean + scale x (x - mean) + offset x std - applied to AGEP (age) and WKHP "
                "(weekly hours). Every label was left exactly as recorded, so the true income rule "
                "is unchanged; the model simply now sees inputs from a region of feature space it "
                "was under-trained on."
            ),
            "monitor_sees": "input drift detectors fire; the population is measurably different",
        },
        "prior": {
            "title": "Prior probability shift - the mix of outcomes changes",
            "plain": (
                "High earners become a larger share of the population, but each kind of person "
                "still earns what they used to. Cause: an economic upswing, or sampling a "
                "wealthier area than the model was trained on."
            ),
            "moves": "P(Y): the base rate of the positive class (income > $50k)",
            "keeps": "P(X|Y): each class's feature profile",
            "recipe": (
                f"Whole rows were resampled - never modified - so the share earning over $50k "
                f"moves from the reference's {pct} up towards the target rate. Because entire rows "
                f"are kept or dropped, every class-conditional feature distribution is an exact "
                f"sub-sample of the original; only the mixing weight changes."
            ),
            "monitor_sees": "a modest, broad input drift (income-correlated features shift with the mix)",
        },
        "concept": {
            "title": "Concept drift - the rule itself changes, in one region",
            "plain": (
                "For a specific slice of the population the link between attributes and income "
                "flips: what used to mark a high earner now marks a low one. Cause: a policy or "
                "definitional change affecting, say, only the most educated."
            ),
            "moves": "P(Y|X): the target rule, for the most-educated rows",
            "keeps": "P(X): every input distribution is byte-for-byte identical to the reference",
            "recipe": (
                "The rows with the highest SCHL (educational attainment) had their label "
                "deterministically inverted - a high earner is relabelled a low earner and vice "
                "versa. Not one feature value was changed, so a monitor that only watches the "
                "inputs sees absolutely nothing."
            ),
            "monitor_sees": "nothing - the inputs and the model's outputs are unchanged",
        },
        "label_noise": {
            "title": "Label noise - the recorded answers are wrong",
            "plain": (
                "The model's predictions are fine; the ground-truth labels arriving from the "
                "annotation pipeline are corrupted. Cause: a broken labelling job, annotator "
                "disagreement, a data-entry bug."
            ),
            "moves": "the observed labels: a uniformly random fraction are flipped",
            "keeps": "P(X) and the model - nothing the model produces changes",
            "recipe": (
                "A uniformly random fraction of production labels were flipped, with no "
                "dependence on any feature. Measured against these labels the model looks less "
                "accurate, but the model is not wrong - the labels are."
            ),
            "monitor_sees": "nothing - and even a labelled audit cannot blame any feature",
        },
    }


def _intensity_param_summary(drift_type: str, params: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"intensity": params.get("label", "?")}
    if drift_type == "covariate":
        row["scale"] = params.get("scale")
        row["offset"] = params.get("offset")
        row["rows affected"] = f"{params.get('fraction', 0) * 100:.0f}%"
    elif drift_type == "prior":
        row["target positive rate"] = f"{params.get('target_positive_rate', 0):.0%}"
        row["actual positive rate"] = f"{spec.get('p_y_production', 0):.0%}"
    elif drift_type == "concept":
        row["region (top SCHL)"] = f"{spec.get('params', {}).get('region_fraction', 0) * 100:.0f}%"
        row["labels inverted"] = f"{spec.get('params', {}).get('region_fraction', 0) * 100:.0f}%"
    elif drift_type == "label_noise":
        row["flip rate"] = f"{params.get('rate', 0):.0%}"
        row["labels flipped"] = f"{spec.get('params', {}).get('observed_flip_fraction', 0) * 100:.1f}%"
    row["true accuracy drop"] = round(spec.get("true_accuracy_drop", 0.0) * 100, 1)
    row["true positive rate"] = f"{spec.get('p_y_production', 0):.0%}"
    return row


def _hist(values: np.ndarray, edges: np.ndarray) -> List[int]:
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    return [int(c) for c in counts]


def _accuracy_by_bin(
    model: Any, X: pd.DataFrame, y: np.ndarray, feature: str, edges: np.ndarray
) -> List[Dict[str, Any]]:
    col = X[feature].to_numpy(dtype=float)
    binned = np.clip(np.digitize(col, edges[1:-1]), 0, len(edges) - 2)
    correct = (model.predict(X.to_numpy(dtype=float)) == np.asarray(y)).astype(float)
    out = []
    for b in range(len(edges) - 1):
        m = binned == b
        out.append({
            "bin_mid": round(float((edges[b] + edges[b + 1]) / 2), 3),
            "lo": round(float(edges[b]), 3),
            "hi": round(float(edges[b + 1]), 3),
            "n": int(m.sum()),
            "accuracy": round(float(correct[m].mean()), 4) if m.any() else None,
        })
    return out


def _accuracy_by_decile(
    model: Any, X: pd.DataFrame, y: np.ndarray, feature: str, decile_edges: np.ndarray
) -> List[Dict[str, Any]]:
    """Model accuracy in each decile of ``feature`` (equal-count buckets from the
    reference), so every point has enough samples even for a coarse feature."""
    col = X[feature].to_numpy(dtype=float)
    binned = np.clip(np.digitize(col, decile_edges[1:-1]), 0, len(decile_edges) - 2)
    correct = (model.predict(X.to_numpy(dtype=float)) == np.asarray(y)).astype(float)
    out = []
    for b in range(len(decile_edges) - 1):
        m = binned == b
        out.append({
            "decile": b + 1,
            "feature_lo": round(float(decile_edges[b]), 2),
            "feature_hi": round(float(decile_edges[b + 1]), 2),
            "n": int(m.sum()),
            "accuracy": round(float(correct[m].mean()), 4) if m.sum() >= 5 else None,
        })
    return out


def build_presentation(
    data: Any,
    model: Any,
    snapshots: List[Dict[str, Any]],
    runs: List[Dict[str, Any]],
    reference_accuracy: float,
    reference_positive_rate: float,
) -> Dict[str, Any]:
    """Assemble the dataset card + per-drift-type explanatory cards and visuals."""
    feats = list(data.feature_names)
    ref_X = data.reference_X
    pool_X, pool_y = data.prod_pool_X, data.prod_pool_y

    dataset_card = {
        "name": "ACS Income - US Census microdata (folktables)",
        "task": "Binary classification: does this person earn more than $50,000 per year?",
        "reference_window": "California, 2014 one-year American Community Survey",
        "n_reference": int(len(ref_X)),
        "n_production_pool": int(len(pool_X)),
        "model": "Logistic regression on standardised features",
        "reference_accuracy": round(float(reference_accuracy), 4),
        "reference_positive_rate": round(float(reference_positive_rate), 4),
        "features": [{"name": f, "meaning": FEATURE_GLOSSARY.get(f, "")} for f in feats],
        "how_it_works": (
            "For each drift type we take a clean held-out slice of the reference data, corrupt it "
            "in one precisely-defined way, and feed it to the monitoring pipeline as if it were "
            "live production traffic. The pipeline never sees what we did. We then check its "
            "diagnosis against the recipe."
        ),
    }

    cards = _drift_cards(reference_positive_rate)
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for s in snapshots:
        by_type.setdefault(s["drift_type"], []).append(s)

    run_by_key = {(r["drift_type"], r["intensity_label"]): r for r in runs}

    catalog: Dict[str, Any] = {}
    for dt, card in cards.items():
        snaps = sorted(by_type.get(dt, []), key=lambda s: s["intensity_order"])
        if not snaps:
            continue
        focus = FOCUS_FEATURE[dt]
        if focus is not None and focus not in feats:
            focus = None
        entry: Dict[str, Any] = {
            "title": card["title"],
            "plain": card["plain"],
            "moves": card["moves"],
            "keeps": card["keeps"],
            "recipe": card["recipe"],
            "monitor_sees": card["monitor_sees"],
            "focus_feature": focus,
            "focus_feature_meaning": FEATURE_GLOSSARY.get(focus) if focus else None,
            "intensity_table": [],
            "intensities": [],
        }

        edges = None
        if focus is not None:
            ref_fv = ref_X[focus].to_numpy(dtype=float)
            # bin over the union of the reference and every injected window so a
            # feature that gets rescaled (covariate) still fits on one axis
            allvals = [ref_fv]
            for s in snaps:
                allvals.append(s["prod_X"][focus].to_numpy(dtype=float))
            union = np.concatenate(allvals)
            lo, hi = float(np.percentile(union, 0.5)), float(np.percentile(union, 99.5))
            edges = np.linspace(lo, hi, 23)
            entry["focus_hist_bin_edges"] = [round(float(e), 3) for e in edges]
            entry["focus_hist_reference"] = _hist(np.clip(ref_fv, lo, hi), edges)
            if dt == "concept":
                dedges = np.unique(np.quantile(ref_X[focus].to_numpy(dtype=float),
                                               np.linspace(0, 1, 11)))
                entry["decile_edges"] = [round(float(e), 2) for e in dedges]
                entry["accuracy_by_decile_reference"] = _accuracy_by_decile(
                    model, pool_X, pool_y, focus, dedges)

        for s in snaps:
            run = run_by_key.get((dt, s["intensity_label"]))
            spec = run["ground_truth"] if run else {}
            entry["intensity_table"].append(_intensity_param_summary(dt, s["params"], spec))

            item: Dict[str, Any] = {
                "label": s["intensity_label"],
                "order": s["intensity_order"],
                "true_accuracy_drop": round(spec.get("true_accuracy_drop", 0.0), 4),
                "positive_rate": round(spec.get("p_y_production", 0.0), 4),
            }
            prod_X, prod_y = s["prod_X"], s["prod_y"]
            if focus is not None and edges is not None:
                lo, hi = edges[0], edges[-1]
                item["focus_hist"] = _hist(np.clip(prod_X[focus].to_numpy(dtype=float), lo, hi), edges)
                if dt == "concept":
                    dedges = np.array(entry["decile_edges"], dtype=float)
                    item["accuracy_by_decile"] = _accuracy_by_decile(model, prod_X, prod_y, focus, dedges)

            # label-flip breakdown where rows stay aligned to the pool
            if dt in ("concept", "label_noise") and len(prod_y) == len(pool_y):
                flipped = np.asarray(prod_y) != np.asarray(pool_y)
                item["flips_neg_to_pos"] = int(((np.asarray(pool_y) == 0) & flipped).sum())
                item["flips_pos_to_neg"] = int(((np.asarray(pool_y) == 1) & flipped).sum())
                item["unchanged_labels"] = int((~flipped).sum())
                if dt == "concept" and focus is not None:
                    fvals = prod_X[focus].to_numpy(dtype=float)
                    item["flipped_focus_min"] = round(float(fvals[flipped].min()), 3) if flipped.any() else None
                    item["nonflipped_focus_max"] = round(float(fvals[~flipped].max()), 3) if (~flipped).any() else None
            entry["intensities"].append(item)

        catalog[dt] = entry

    return {"dataset_card": dataset_card, "drift_catalog": catalog}
