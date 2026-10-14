"""Orchestrate the drift-injection benchmark across states and seeds.

A replicate is one (state, seed) pair. Within a replicate the state's ACS
Income 2014 data are split into train / calibration / test (see
:mod:`drift_autopsy.experiments.acs`), the monitored model is trained, and two
disjoint samples are drawn from the test split: a production window and a
grading holdout. Each drift type is injected at each intensity into copies of
both samples (the holdout with an independent random stream), plus one
no-drift control, and every window is evaluated by
:func:`drift_autopsy.benchmark.window.evaluate_window`.

Nothing in the monitor is tuned on these windows: every decision is a test at
the configured significance level.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from drift_autopsy.benchmark import injectors as inj
from drift_autopsy.benchmark.metrics import aggregate
from drift_autopsy.benchmark.spec import DRIFT_TYPES
from drift_autopsy.benchmark.window import ReplicateContext, evaluate_window
from drift_autopsy.experiments.acs import (
    ACS_CATEGORICAL,
    ACS_FEATURES,
    load_acs,
    make_acs_model,
    split_reference,
    to_dataset,
)
from drift_autopsy.reliability import ReliabilityAnalyzer

logger = logging.getLogger(__name__)

COVARIATE_FEATURES = ["AGEP", "WKHP"]
CONCEPT_REGION_FEATURE = "SCHL"

# Intensity ladders, weak -> strong. Prior targets are offsets from each
# state's own base rate so the ladder means the same thing in every state.
INTENSITY_LADDER: Dict[str, List[Dict[str, Any]]] = {
    "covariate": [
        {"label": "mild", "scale": 1.5, "offset": 0.8, "fraction": 0.5},
        {"label": "moderate", "scale": 2.2, "offset": 1.7, "fraction": 0.6},
        {"label": "strong", "scale": 3.2, "offset": 2.8, "fraction": 0.7},
    ],
    "prior": [
        {"label": "mild", "positive_rate_increase": 0.14},
        {"label": "moderate", "positive_rate_increase": 0.23},
        {"label": "strong", "positive_rate_increase": 0.32},
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
class BenchmarkConfig:
    states: Sequence[str] = ("CA", "TX", "NY", "FL", "WA")
    seeds: Sequence[int] = tuple(range(10))
    production_size: int = 4000
    holdout_size: int = 2000
    audit_sizes: Sequence[int] = (200, 600, 1000)
    primary_audit_size: int = 600
    alpha: float = 0.05
    psi_threshold: float = 0.2
    mmd_permutations: int = 200
    mmd_max_samples: int = 3000
    year: int = 2014
    n_jobs: int = -1
    drift_types: Sequence[str] = field(default_factory=lambda: list(DRIFT_TYPES))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["intensity_ladder"] = INTENSITY_LADDER
        d["covariate_features"] = COVARIATE_FEATURES
        d["concept_region_feature"] = CONCEPT_REGION_FEATURE
        return d


def _inject(drift_type: str, params: Dict[str, Any], X, y, base_rate: float, seed: int) -> inj.InjectionResult:
    label = params["label"]
    if drift_type == "none":
        return inj.inject_none(X, y)
    if drift_type == "covariate":
        return inj.inject_covariate(X, y, COVARIATE_FEATURES, params["scale"], params["offset"], label,
                                    fraction=params["fraction"], seed=seed)
    if drift_type == "prior":
        return inj.inject_prior(X, y, base_rate + params["positive_rate_increase"], label, seed=seed)
    if drift_type == "concept":
        return inj.inject_concept(X, y, CONCEPT_REGION_FEATURE, params["region_fraction"], label, seed=seed)
    if drift_type == "label_noise":
        return inj.inject_label_noise(X, y, params["rate"], label, seed=seed)
    raise ValueError(drift_type)


def run_replicate(state: str, seed: int, cfg: BenchmarkConfig) -> List[Dict[str, Any]]:
    """All windows of one (state, seed) replicate."""
    with threadpool_limits(limits=1):
        X, y = load_acs(cfg.year, state)
        split = split_reference(X, y, seed=seed)
        factory = lambda keep_columns=None: make_acs_model(random_state=seed, keep_columns=keep_columns)
        model = factory().fit(split.train_X.to_numpy(dtype=float), split.train_y)
        reference = to_dataset(split.calibration_X, split.calibration_y, model=model)
        reference_accuracy = float(np.mean(reference.predictions == split.calibration_y))
        reliability = ReliabilityAnalyzer(
            model, split.train_X.to_numpy(dtype=float), split.calibration_X.to_numpy(dtype=float),
            split.calibration_y, categorical_indices=[ACS_FEATURES.index(f) for f in ACS_CATEGORICAL],
            random_state=seed,
        )
        rc = ReplicateContext(
            state=state, seed=seed, model=model, model_factory=factory, feature_names=list(ACS_FEATURES),
            categorical_features=list(ACS_CATEGORICAL), train_X=split.train_X, train_y=split.train_y,
            reference=reference, reference_accuracy=reference_accuracy, reliability=reliability,
            alpha=cfg.alpha, psi_threshold=cfg.psi_threshold, mmd_permutations=cfg.mmd_permutations,
            mmd_max_samples=cfg.mmd_max_samples, audit_sizes=list(cfg.audit_sizes),
            primary_audit_size=cfg.primary_audit_size,
        )

        need = cfg.production_size + cfg.holdout_size
        if len(split.test_X) < need:
            raise ValueError(f"{state}: test split has {len(split.test_X)} rows, need {need}")
        rng = np.random.default_rng([seed, 1])
        order = rng.permutation(len(split.test_X))
        p_idx, h_idx = order[: cfg.production_size], order[cfg.production_size: need]
        pX, py = split.test_X.iloc[p_idx].reset_index(drop=True), split.test_y[p_idx]
        hX, hy = split.test_X.iloc[h_idx].reset_index(drop=True), split.test_y[h_idx]
        base_rate = float(np.mean(split.train_y))

        records = []
        for t_idx, drift_type in enumerate(cfg.drift_types):
            ladder = [{"label": "control"}] if drift_type == "none" else INTENSITY_LADDER[drift_type]
            for order_i, params in enumerate(ladder):
                prod = _inject(drift_type, params, pX, py, base_rate, seed=int(np.random.SeedSequence([seed, t_idx, order_i, 0]).generate_state(1)[0]))
                hold = _inject(drift_type, params, hX, hy, base_rate, seed=int(np.random.SeedSequence([seed, t_idx, order_i, 1]).generate_state(1)[0]))
                spec = prod.spec
                spec.reference_accuracy = reference_accuracy
                spec.production_accuracy = float(np.mean(model.predict(prod.X.to_numpy(dtype=float)) == prod.y))
                records.append(evaluate_window(
                    rc, drift_type, params["label"], order_i, dict(params),
                    prod.X, prod.y, hold.X, hold.y, spec.to_dict(),
                ))
        return records


def run_benchmark(cfg: Optional[BenchmarkConfig] = None,
                  progress: Optional[Callable[[str], None]] = None,
                  on_runs: Optional[Callable[[List[Dict[str, Any]]], None]] = None) -> Dict[str, Any]:
    """Run every replicate in parallel and aggregate. ``on_runs`` receives the raw
    window records before aggregation (so they can be saved first)."""
    cfg = cfg or BenchmarkConfig()
    jobs = [(s, k) for s in cfg.states for k in cfg.seeds]
    say = progress or (lambda m: logger.info(m))
    say(f"running {len(jobs)} replicates x {sum(1 if t == 'none' else 3 for t in cfg.drift_types)} windows")
    per_job = Parallel(n_jobs=cfg.n_jobs, verbose=10)(delayed(run_replicate)(s, k, cfg) for s, k in jobs)
    runs = [r for job in per_job for r in job]
    if on_runs is not None:
        on_runs(runs)
    return {"config": cfg.to_dict(), "runs": runs, "summary": aggregate(runs, cfg)}
