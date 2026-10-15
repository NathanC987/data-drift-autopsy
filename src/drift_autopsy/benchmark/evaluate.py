"""Orchestrate the drift-injection benchmark across datasets, designs, models, states and seeds.

A replicate is one (state, seed) pair. Within a replicate the state's 2014
data for the task are split into train / calibration / test (see
:mod:`drift_autopsy.experiments.acs`), the monitored model is trained, and two
disjoint samples are drawn from the test split: a production window and a
grading holdout. Each mechanism is injected at each intensity into copies of
both samples (the holdout with an independent random stream), plus one
no-drift control in design A, and every window is evaluated by
:func:`drift_autopsy.benchmark.window.evaluate_window`.

Two designs. Design A is the original benchmark; its mechanisms partly match
what the tests look for. Design B uses held-out mechanisms (see
:mod:`drift_autopsy.benchmark.injectors`). All parameters below were fixed
before design B was first run, and the diagnosis was not changed after.

Nothing in the monitor is tuned on these windows: every decision is a test at
the configured significance level.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from drift_autopsy.benchmark import injectors as inj
from drift_autopsy.benchmark.metrics import aggregate
from drift_autopsy.benchmark.spec import DRIFT_TYPES
from drift_autopsy.benchmark.window import MECHANISMS_B, ReplicateContext, evaluate_window
from drift_autopsy.experiments.acs import ACS_TASKS, load_acs_task, make_acs_model, split_reference, to_dataset
from drift_autopsy.reliability import ReliabilityAnalyzer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkDataset:
    """A reference population for the benchmark and its design A injection targets."""

    name: str
    task: str
    covariate_features: Tuple[str, ...]
    concept_region_feature: str
    income_column: Optional[str] = None      # enables the threshold mechanism of design B


DATASETS: Dict[str, BenchmarkDataset] = {
    "acs_income": BenchmarkDataset("acs_income", "income", ("AGEP", "WKHP"), "SCHL", income_column="PINCP"),
    # ACS Employment has two numeric inputs; the concept region is set by age
    "acs_employment": BenchmarkDataset("acs_employment", "employment", ("AGEP", "SCHL"), "AGEP"),
}

# ACS Income design A names, kept for the existing outputs and tests.
COVARIATE_FEATURES = list(DATASETS["acs_income"].covariate_features)
CONCEPT_REGION_FEATURE = DATASETS["acs_income"].concept_region_feature

# Design A intensity ladders, weak -> strong. Prior targets are offsets from each
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

# Design B: held-out mechanisms, the drift type each one produces, and ladders.
MECHANISM_TYPE: Dict[str, str] = {
    "selection": "covariate",
    "prior_decrease": "prior",
    "concept_oblique": "concept",
    "concept_threshold": "concept",
    "class_noise": "label_noise",
}
INTENSITY_LADDER_B: Dict[str, List[Dict[str, Any]]] = {
    "selection": [{"label": "mild", "beta": 0.5}, {"label": "moderate", "beta": 1.0},
                  {"label": "strong", "beta": 1.5}],
    "prior_decrease": [{"label": "mild", "positive_rate_decrease": 0.08},
                       {"label": "moderate", "positive_rate_decrease": 0.14},
                       {"label": "strong", "positive_rate_decrease": 0.20}],
    "concept_oblique": [{"label": "mild", "region_fraction": 0.15}, {"label": "moderate", "region_fraction": 0.30},
                        {"label": "strong", "region_fraction": 0.50}],
    "concept_threshold": [{"label": "mild", "threshold": 55000.0}, {"label": "moderate", "threshold": 60000.0},
                          {"label": "strong", "threshold": 70000.0}],
    "class_noise": [{"label": "mild", "rate": 0.10}, {"label": "moderate", "rate": 0.20},
                    {"label": "strong", "rate": 0.35}],
}


@dataclass
class BenchmarkConfig:
    dataset: str = "acs_income"
    design: str = "A"
    model: str = "lr"
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

    def mechanisms(self) -> List[Tuple[str, str, List[Dict[str, Any]]]]:
        """(mechanism, drift type, ladder) for every window family of this design."""
        ds = DATASETS[self.dataset]
        if self.design == "A":
            return [(t, t, [{"label": "control"}] if t == "none" else INTENSITY_LADDER[t]) for t in self.drift_types]
        out = []
        for m in MECHANISMS_B:
            if m == "concept_threshold" and ds.income_column is None:
                continue
            if MECHANISM_TYPE[m] in self.drift_types:
                out.append((m, MECHANISM_TYPE[m], INTENSITY_LADDER_B[m]))
        return out

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        ds = DATASETS[self.dataset]
        d["task"] = ds.task
        d["intensity_ladder"] = INTENSITY_LADDER if self.design == "A" else {
            m: INTENSITY_LADDER_B[m] for m, _, _ in self.mechanisms()}
        d["mechanism_type"] = {m: t for m, t, _ in self.mechanisms()}
        d["covariate_features"] = list(ds.covariate_features)
        d["concept_region_feature"] = ds.concept_region_feature
        return d


def _inject(design: str, mechanism: str, params: Dict[str, Any], X, y, base_rate: float, seed: int,
            ds: BenchmarkDataset, directions: Dict[str, inj.RandomDirection], income=None) -> inj.InjectionResult:
    label = params["label"]
    if design == "A":
        if mechanism == "none":
            return inj.inject_none(X, y)
        if mechanism == "covariate":
            return inj.inject_covariate(X, y, list(ds.covariate_features), params["scale"], params["offset"], label,
                                        fraction=params["fraction"], seed=seed)
        if mechanism == "prior":
            return inj.inject_prior(X, y, base_rate + params["positive_rate_increase"], label, seed=seed)
        if mechanism == "concept":
            return inj.inject_concept(X, y, ds.concept_region_feature, params["region_fraction"], label, seed=seed)
        if mechanism == "label_noise":
            return inj.inject_label_noise(X, y, params["rate"], label, seed=seed)
    else:
        if mechanism == "selection":
            return inj.inject_selection(X, y, directions["selection"], params["beta"], label, seed=seed)
        if mechanism == "prior_decrease":
            return inj.inject_prior(X, y, base_rate - params["positive_rate_decrease"], label, seed=seed)
        if mechanism == "concept_oblique":
            return inj.inject_concept_oblique(X, y, directions["concept"], params["region_fraction"], label, seed=seed)
        if mechanism == "concept_threshold":
            return inj.inject_threshold(X, y, income, params["threshold"], label)
        if mechanism == "class_noise":
            return inj.inject_class_noise(X, y, params["rate"], label, seed=seed)
    raise ValueError(f"{design}/{mechanism}")


def _seed(*key: int) -> int:
    return int(np.random.SeedSequence(list(key)).generate_state(1)[0])


def run_replicate(state: str, seed: int, cfg: BenchmarkConfig) -> List[Dict[str, Any]]:
    """All windows of one (state, seed) replicate."""
    ds = DATASETS[cfg.dataset]
    spec = ACS_TASKS[ds.task]
    with threadpool_limits(limits=1):
        X, y, extra = load_acs_task(ds.task, cfg.year, state, with_extra=True)
        income_col = ds.income_column
        if income_col is not None:
            X = X.assign(**{income_col: extra[income_col].to_numpy()})
        split = split_reference(X, y, seed=seed)
        income_test = split.test_X.pop(income_col).to_numpy() if income_col else None
        if income_col:
            split.train_X.pop(income_col)
            split.calibration_X.pop(income_col)

        def factory(keep_columns=None):
            return make_acs_model(random_state=seed, keep_columns=keep_columns, kind=cfg.model, task=ds.task)

        model = factory().fit(split.train_X.to_numpy(dtype=float), split.train_y)
        reference = to_dataset(split.calibration_X, split.calibration_y, model=model, task=ds.task)
        reference_accuracy = float(np.mean(reference.predictions == split.calibration_y))
        categorical_idx = [list(spec.features).index(f) for f in spec.categorical]
        reliability = ReliabilityAnalyzer(
            model, split.train_X.to_numpy(dtype=float), split.calibration_X.to_numpy(dtype=float),
            split.calibration_y, categorical_indices=categorical_idx, random_state=seed,
        )
        rc = ReplicateContext(
            state=state, seed=seed, dataset=cfg.dataset, model_kind=cfg.model, model=model, model_factory=factory,
            feature_names=list(spec.features), categorical_features=spec.categorical,
            train_X=split.train_X, train_y=split.train_y,
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
        p_inc = income_test[p_idx] if income_test is not None else None
        h_inc = income_test[h_idx] if income_test is not None else None
        base_rate = float(np.mean(split.train_y))
        directions = {}
        if cfg.design == "B":
            directions = {
                "selection": inj.RandomDirection(split.train_X, spec.numeric, spec.categorical, _seed(seed, 7001)),
                "concept": inj.RandomDirection(split.train_X, spec.numeric, spec.categorical, _seed(seed, 7002)),
            }

        records = []
        for m_idx, (mechanism, drift_type, ladder) in enumerate(cfg.mechanisms()):
            # design A keeps the original seeding (type index); design B offsets by 100
            family = DRIFT_TYPES.index(mechanism) if cfg.design == "A" else 100 + MECHANISMS_B.index(mechanism)
            for order_i, params in enumerate(ladder):
                prod = _inject(cfg.design, mechanism, params, pX, py, base_rate, _seed(seed, family, order_i, 0),
                               ds, directions, income=p_inc)
                hold = _inject(cfg.design, mechanism, params, hX, hy, base_rate, _seed(seed, family, order_i, 1),
                               ds, directions, income=h_inc)
                gt = prod.spec
                gt.reference_accuracy = reference_accuracy
                gt.production_accuracy = float(np.mean(model.predict(prod.X.to_numpy(dtype=float)) == prod.y))
                records.append(evaluate_window(
                    rc, cfg.design, mechanism, drift_type, params["label"], order_i, dict(params),
                    prod.X, prod.y, hold.X, hold.y, gt.to_dict(),
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
    say(f"{cfg.dataset} design {cfg.design} model {cfg.model}: running {len(jobs)} replicates x "
        f"{sum(len(ladder) for _, _, ladder in cfg.mechanisms())} windows")
    per_job = Parallel(n_jobs=cfg.n_jobs, verbose=10)(delayed(run_replicate)(s, k, cfg) for s, k in jobs)
    runs = [r for job in per_job for r in job]
    if on_runs is not None:
        on_runs(runs)
    return {"config": cfg.to_dict(), "runs": runs, "summary": aggregate(runs, cfg)}
