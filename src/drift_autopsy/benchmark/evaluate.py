"""Orchestrate the drift-injection benchmark.

Trains the project's standard tabular model on a real reference distribution,
then for each injected drift type and intensity runs the full pipeline probe,
the remediation triage, and a small labelled audit, grades every stage against
the answer key, and returns one nested result dict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from drift_autopsy.benchmark import injectors as inj
from drift_autopsy.benchmark.audit import labelled_audit
from drift_autopsy.benchmark.metrics import aggregate, estimator_error, localisation_prf
from drift_autopsy.benchmark.pipeline_probe import grade_reliability, run_pipeline_probe
from drift_autopsy.benchmark.presentation import build_presentation
from drift_autopsy.benchmark.signature import classify_drift_type, drift_signature
from drift_autopsy.benchmark.spec import InjectedDriftSpec
from drift_autopsy.remediation import RemediationContext, remediation_triage

logger = logging.getLogger(__name__)

COVARIATE_FEATURES = ["AGEP", "WKHP"]
CONCEPT_REGION_FEATURE = "SCHL"

# intensity ladders, ordered weak -> strong
INTENSITY_LADDER: Dict[str, List[Dict[str, Any]]] = {
    "covariate": [
        {"label": "mild", "scale": 1.5, "offset": 0.8, "fraction": 0.5},
        {"label": "moderate", "scale": 2.2, "offset": 1.7, "fraction": 0.6},
        {"label": "strong", "scale": 3.2, "offset": 2.8, "fraction": 0.7},
    ],
    "prior": [
        {"label": "mild", "target_positive_rate": 0.49},
        {"label": "moderate", "target_positive_rate": 0.58},
        {"label": "strong", "target_positive_rate": 0.67},
    ],
    "concept": [
        {"label": "mild", "region_fraction": 0.15},
        {"label": "moderate", "region_fraction": 0.30},
        {"label": "strong", "region_fraction": 0.50},
    ],
    "label_noise": [
        {"label": "mild", "rate": 0.08},
        {"label": "moderate", "rate": 0.18},
        {"label": "strong", "rate": 0.32},
    ],
}


@dataclass
class BenchmarkData:
    train_X: pd.DataFrame
    train_y: np.ndarray
    reference_X: pd.DataFrame
    reference_y: np.ndarray
    prod_pool_X: pd.DataFrame
    prod_pool_y: np.ndarray
    eval_pool_X: pd.DataFrame
    eval_pool_y: np.ndarray
    feature_names: List[str]


def default_model_factory() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def load_acs_reference(
    year: int = 2014,
    state: str = "CA",
    n_train: int = 20000,
    n_reference: int = 6000,
    n_pool: int = 8000,
    seed: int = 42,
) -> BenchmarkData:
    """Split one ACS Income (state, year) into train / reference / injection pool."""
    from drift_autopsy.data import FolktablesLoader

    feats = FolktablesLoader.ACS_INCOME_FEATURES
    frame = FolktablesLoader.load_acs_income_cached(year, state)
    X = frame[feats].reset_index(drop=True).astype(float)
    y = frame["target"].to_numpy().astype(int)

    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=min(n_train, len(X) // 2), random_state=seed, stratify=y
    )
    X_ref, X_pool, y_ref, y_pool = train_test_split(
        X_rest, y_rest,
        train_size=min(n_reference, len(X_rest) // 2),
        random_state=seed, stratify=y_rest,
    )
    take = min(n_pool, len(X_pool))
    X_pool, y_pool = X_pool.iloc[:take], y_pool[:take]
    X_prod, X_eval, y_prod, y_eval = train_test_split(
        X_pool, y_pool, test_size=0.4, random_state=seed, stratify=y_pool
    )

    return BenchmarkData(
        train_X=X_tr.reset_index(drop=True), train_y=y_tr,
        reference_X=X_ref.reset_index(drop=True), reference_y=y_ref,
        prod_pool_X=X_prod.reset_index(drop=True), prod_pool_y=y_prod,
        eval_pool_X=X_eval.reset_index(drop=True), eval_pool_y=y_eval,
        feature_names=list(feats),
    )


def _inject(drift_type: str, params: Dict[str, Any], X: pd.DataFrame, y: np.ndarray,
            seed: int) -> inj.InjectionResult:
    label = params["label"]
    if drift_type == "none":
        return inj.inject_none(X, y)
    if drift_type == "covariate":
        return inj.inject_covariate(
            X, y, COVARIATE_FEATURES, params["scale"], params["offset"], label,
            fraction=params.get("fraction", 0.6), seed=seed,
        )
    if drift_type == "prior":
        return inj.inject_prior(X, y, params["target_positive_rate"], label, seed=seed)
    if drift_type == "concept":
        return inj.inject_concept(
            X, y, CONCEPT_REGION_FEATURE, params["region_fraction"], label, seed=seed
        )
    if drift_type == "label_noise":
        return inj.inject_label_noise(X, y, params["rate"], label, seed=seed)
    raise ValueError(drift_type)


def _finalise_spec(spec: InjectedDriftSpec, model, ref_X, ref_y, prod_X, prod_y) -> InjectedDriftSpec:
    from sklearn.metrics import accuracy_score

    spec.reference_accuracy = float(accuracy_score(ref_y, model.predict(ref_X.to_numpy(dtype=float))))
    spec.production_accuracy = float(
        accuracy_score(prod_y, model.predict(prod_X.to_numpy(dtype=float)))
    )
    return spec


def run_benchmark(
    data: Optional[BenchmarkData] = None,
    drift_types: Optional[List[str]] = None,
    reliability_sample: int = 150,
    audit_labels: int = 600,
    seed: int = 42,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the full sweep and return the graded result dict."""
    if data is None:
        data = load_acs_reference(seed=seed)
    drift_types = drift_types or ["none", "covariate", "prior", "concept", "label_noise"]
    say = progress or (lambda m: logger.info(m))

    model = default_model_factory().fit(data.train_X.to_numpy(dtype=float), data.train_y)
    ref_np = data.reference_X.to_numpy(dtype=float)
    ref_proba = model.predict_proba(ref_np)
    ref_pred = model.predict(ref_np)
    reference_cache = {
        "confidence": ref_proba.max(axis=1),
        "correct": (ref_pred == data.reference_y).astype(float),
    }
    from sklearn.metrics import accuracy_score

    reference_accuracy = float(accuracy_score(data.reference_y, ref_pred))
    reference_positive_rate = float(np.mean(data.train_y))
    _probe_cache: Dict[bytes, Dict[str, Any]] = {}

    runs: List[Dict[str, Any]] = []
    snapshots: List[Dict[str, Any]] = []
    for drift_type in drift_types:
        ladder = [{"label": "control"}] if drift_type == "none" else INTENSITY_LADDER[drift_type]
        for order, params in enumerate(ladder):
            tag = f"{drift_type}/{params['label']}"
            say(f"running {tag}")

            prod = _inject(drift_type, params, data.prod_pool_X, data.prod_pool_y, seed)
            holdout = _inject(drift_type, params, data.eval_pool_X, data.eval_pool_y, seed + 1)
            spec = _finalise_spec(prod.spec, model, data.reference_X, data.reference_y, prod.X, prod.y)
            if drift_type != "none":
                snapshots.append({
                    "drift_type": drift_type, "intensity_label": params["label"],
                    "intensity_order": order, "params": dict(params),
                    "prod_X": prod.X, "prod_y": prod.y,
                })

            # covariate/prior change X; concept/label-noise/none leave X byte
            # identical, so the label-free probe is reused across them. Only the
            # per-window reliability grading (which needs labels) is redone.
            x_key = prod.X.to_numpy(dtype=float).tobytes()
            if x_key in _probe_cache:
                say("  (reusing label-free probe - inputs identical to an earlier window)")
            else:
                _probe_cache[x_key] = run_pipeline_probe(
                    reference_X=data.reference_X,
                    production_X=prod.X,
                    model=model,
                    feature_names=data.feature_names,
                    reference_predictions_cache=reference_cache,
                    reliability_reference=data.reference_X,
                    reliability_sample=reliability_sample,
                    seed=seed,
                )
            probe = dict(_probe_cache[x_key])
            probe["reliability"] = grade_reliability(
                probe["reliability"], model, prod.X, prod.y, data.feature_names
            )

            localised = probe["localisation"].get("KS Test", {}).get("drifted_features", [])
            ctx = RemediationContext(
                reference_X=data.train_X.to_numpy(dtype=float),
                reference_y=data.train_y,
                production_X=prod.X.to_numpy(dtype=float),
                production_y=prod.y,
                holdout_X=holdout.X.to_numpy(dtype=float),
                holdout_y=holdout.y,
                feature_names=data.feature_names,
                drifted_features=list(localised),
                shift_name=tag,
                base_model=model,
                recent_windows=[],
                reference_accuracy=reference_accuracy,
            )
            estimated_gap = reference_accuracy - probe["label_free_accuracy_estimate"]
            try:
                triage = remediation_triage(ctx, estimated_gap=estimated_gap)
            except Exception as exc:
                logger.warning("triage failed for %s: %s", tag, exc)
                triage = {"will_retraining_help": False, "rationale": f"triage error: {exc}",
                          "signals": {}}

            audit = labelled_audit(
                model=model, production_X=prod.X, production_y=prod.y,
                feature_names=data.feature_names, reference_accuracy=reference_accuracy,
                reference_positive_rate=reference_positive_rate,
                n_labels=audit_labels, seed=seed,
            )

            signature = drift_signature(
                probe=probe, triage=triage, reference_accuracy=reference_accuracy,
                production_X=prod.X, feature_names=data.feature_names,
            )
            verdict = classify_drift_type(signature, audit=audit)
            verdict["correct"] = bool(verdict["predicted_type"] == drift_type)
            verdict["expected_diagnosis"] = spec.expected_diagnosis
            verdict["triage_verdict"] = bool(triage["will_retraining_help"])
            verdict["triage_rationale"] = triage["rationale"]

            audit_feature = (
                audit.get("structured_feature")
                if audit.get("structure_score", 0.0) >= 0.10
                else None
            )
            grading = {
                "localisation": localisation_prf(localised, spec.affected_features, audit_feature),
                "estimator": estimator_error(probe["label_free_accuracy_estimate"], spec),
            }

            runs.append({
                "drift_type": drift_type,
                "intensity_label": params["label"],
                "intensity_order": order,
                "ground_truth": spec.to_dict(),
                "pipeline": probe,
                "triage": {
                    "will_retraining_help": bool(triage["will_retraining_help"]),
                    "rationale": triage["rationale"],
                    "signals": triage.get("signals", {}),
                },
                "labelled_audit": audit,
                "signature": signature,
                "verdict": verdict,
                "grading": grading,
            })

    presentation = build_presentation(
        data=data, model=model, snapshots=snapshots, runs=runs,
        reference_accuracy=reference_accuracy, reference_positive_rate=reference_positive_rate,
    )

    return {
        "config": {
            "seed": seed,
            "reference": "ACS Income CA 2014",
            "model": "StandardScaler + LogisticRegression (binary: income > $50k)",
            "n_train": int(len(data.train_X)),
            "n_reference": int(len(data.reference_X)),
            "n_production_pool": int(len(data.prod_pool_X)),
            "reference_accuracy": round(reference_accuracy, 4),
            "reference_positive_rate": round(reference_positive_rate, 4),
            "covariate_features": COVARIATE_FEATURES,
            "concept_region_feature": CONCEPT_REGION_FEATURE,
            "reliability_sample": reliability_sample,
            "audit_labels": audit_labels,
            "intensity_ladder": INTENSITY_LADDER,
        },
        "presentation": presentation,
        "runs": runs,
        "summary": aggregate(runs),
    }
