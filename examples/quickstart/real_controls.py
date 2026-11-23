"""More control windows and the published methods on the real datasets (BNRC revision).

For one real dataset and one model family:

* extra controls - the reference is split again with each of CONTROL_SEEDS,
  the model is refitted on the new training part and the new test part is
  the control window; every detector, our diagnosis (label-free stage and
  audit) and the published methods of drift_autopsy.published_methods run on
  it, so each control counts how often a method raises an alarm on data
  without drift;
* real windows - the published methods run on every window of the study
  (seed 42 setup), with the same audit labels as our audit; the regenerated
  audit sample is checked against the stored study output.

Run:  DDA_OUTPUTS=outputs/bnrc DDA_PERMUTATIONS=1000 \
      python examples/quickstart/real_controls.py --dataset income --model lr
Output: {DDA_OUTPUTS}/real_controls/{dataset}_{model}.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from drift_autopsy import published_methods as pm
from drift_autopsy.detectors import CBPE, KSTest, MMD, PSI
from drift_autopsy.diagnosis import audit_diagnosis, combine, label_free_diagnosis
from drift_autopsy.experiments.common import baseline_detectors, json_default, window_rng
from drift_autopsy.utils import setup_logging

ALPHA = 0.05
PSI_THRESHOLD = 0.2
N_PERMUTATIONS = int(os.environ.get("DDA_PERMUTATIONS", 200))
MMD_MAX_SAMPLES = 3000
STUDY_SEED = 42
CONTROL_SEEDS = tuple(range(20))
OUTPUTS = Path(os.environ.get("DDA_OUTPUTS", "outputs"))


@dataclass
class Setup:
    """A fitted reference for one split seed."""

    model: Any
    reference: Any                                    # calibration Dataset with target
    control_X: pd.DataFrame
    control_y: np.ndarray
    window_ds: Callable[[pd.DataFrame], Any]          # frame -> production Dataset
    features: List[str]
    categorical: List[str]
    audit_n: Callable[[int], int]


def setup(dataset: str, model_kind: str, seed: int) -> Setup:
    if dataset in ("income", "employment"):
        from drift_autopsy.experiments.acs import ACS_TASKS, load_acs_task, make_acs_model, split_reference, to_dataset
        X, y = load_acs_task(dataset, 2014, "CA")
        split = split_reference(X, y, seed=seed)
        model = make_acs_model(random_state=seed, kind=model_kind, task=dataset).fit(
            split.train_X.to_numpy(dtype=float), split.train_y)
        spec = ACS_TASKS[dataset]
        return Setup(model, to_dataset(split.calibration_X, split.calibration_y, model=model, task=dataset),
                     split.test_X.reset_index(drop=True), np.asarray(split.test_y),
                     lambda F: to_dataset(F, None, model=model, task=dataset), list(spec.features),
                     list(spec.categorical), lambda n: min(600, n))
    if dataset == "gas":
        from drift_autopsy.experiments import gas
        data = gas.load_batches()
        reference = data[data["batch"].isin(gas.REFERENCE_BATCHES)].reset_index(drop=True)
        split = gas.split_reference(reference, seed=seed)
        with threadpool_limits(limits=1):
            model = gas.make_gas_model(random_state=seed, kind=model_kind).fit(gas.features(split.train),
                                                                               split.train["gas"])
        return Setup(model, gas.to_dataset(split.calibration, model=model),
                     split.test[gas.FEATURES].reset_index(drop=True), split.test["gas"].to_numpy(),
                     lambda F: gas.to_dataset(F, model=model, with_target=False),
                     list(gas.FEATURES), [], gas.audit_size)
    from drift_autopsy.experiments import clear10 as c
    bucket1 = c.load_bucket(c.REFERENCE_BUCKET)
    split = c.split_reference(bucket1, seed=seed)
    with threadpool_limits(limits=1):
        model = c.make_clear_model(random_state=seed, kind=model_kind).fit(c.embeddings(split.train),
                                                                           split.train["y_true"])
    return Setup(model, c.to_dataset(split.calibration, model=model),
                 split.test[c.EMBEDDING_COLUMNS].reset_index(drop=True), split.test["y_true"].to_numpy(),
                 lambda F: c.to_dataset(F, model=model, with_target=False), list(c.EMBEDDING_COLUMNS), [],
                 lambda n: min(600, n))


def study_windows(dataset: str) -> Dict[str, tuple]:
    """The study's windows other than the control: key -> (X, y)."""
    if dataset in ("income", "employment"):
        from drift_autopsy.experiments.acs import load_acs_task
        out = {f"CA {yr}": load_acs_task(dataset, yr, "CA") for yr in (2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023)}
        out.update({f"{s} 2014": load_acs_task(dataset, 2014, s) for s in ("TX", "NY", "FL", "WA")})
        return {k: (X.reset_index(drop=True), np.asarray(y)) for k, (X, y) in out.items()}
    if dataset == "gas":
        from drift_autopsy.experiments import gas
        data = gas.load_batches()
        return {f"batch {b}": (data[data["batch"] == b][gas.FEATURES].reset_index(drop=True),
                               data[data["batch"] == b]["gas"].to_numpy()) for b in gas.ANALYSIS_BATCHES}
    from drift_autopsy.experiments import clear10 as c
    out = {}
    for b in c.ANALYSIS_BUCKETS:
        f = c.load_bucket(b)
        out[str(b)] = (f[c.EMBEDDING_COLUMNS].reset_index(drop=True), f["y_true"].to_numpy())
    return out


def control_key(dataset: str) -> str:
    return {"gas": "reference held-out", "clear10": "1 held-out"}.get(dataset, "CA 2014 held-out")


def detectors(seed: int) -> Dict[str, Any]:
    return {
        "KS Test": KSTest(threshold=ALPHA, correction="bonferroni"),
        "PSI": PSI(threshold=PSI_THRESHOLD, n_bins=10),
        "MMD": MMD(threshold=ALPHA, n_permutations=N_PERMUTATIONS, max_samples=MMD_MAX_SAMPLES,
                   standardize=True, random_state=seed),
        "CBPE": CBPE(threshold=ALPHA, n_bins=10),
        **baseline_detectors(ALPHA, seed, max_samples=MMD_MAX_SAMPLES),
    }


def published(s: Setup, prod, X: pd.DataFrame, y: np.ndarray, idx: np.ndarray, lf_verdict: str,
              mmd_p: Optional[float], seed: int) -> Dict[str, Any]:
    aX, ay = X.iloc[idx][s.features], y[idx]
    ros = pm.roschewitz_identification(s.reference, prod, alpha=ALPHA, n_permutations=N_PERMUTATIONS,
                                       max_samples=MMD_MAX_SAMPLES, seed=seed, mmd_p=mmd_p)
    ds = pm.detectshift_diagnosis(s.reference.to_pandas()[s.features], np.asarray(s.reference.target), aX, ay,
                                  categorical=s.categorical, alpha=ALPHA, n_permutations=N_PERMUTATIONS, seed=seed)
    cl = pm.confident_learning_audit(s.model, s.reference, aX, ay, alpha=ALPHA)
    return {"verdicts": {"roschewitz": ros["verdict"], "detectshift": ds["verdict"],
                         "confident_learning": pm.label_free_then_cl(lf_verdict, cl)},
            "roschewitz": ros, "detectshift": ds, "confident_learning": cl}


def run_control(dataset: str, model_kind: str, seed: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with threadpool_limits(limits=1):
        s = setup(dataset, model_kind, seed)
        prod = s.window_ds(s.control_X)
        detection = {}
        for name, det in detectors(seed).items():
            res = det.fit_detect(s.reference, prod)
            detection[name] = {"drift_detected": bool(res.drift_detected),
                               "p_value": None if res.p_value is None else float(res.p_value)}
        mmd = {"statistic": 0.0, "p_value": detection["MMD"]["p_value"]}
        lf = label_free_diagnosis(s.reference, prod, alpha=ALPHA, mmd_permutations=N_PERMUTATIONS,
                                  mmd_max_samples=MMD_MAX_SAMPLES, random_state=seed, precomputed_mmd=mmd)
        rng = window_rng(seed, "control")
        idx = rng.choice(len(s.control_X), s.audit_n(len(s.control_X)), replace=False)
        au = audit_diagnosis(s.model, s.reference, s.control_X.iloc[idx], s.control_y[idx], alpha=ALPHA)
        diag = combine(lf, au)
        pub = published(s, prod, s.control_X, s.control_y, idx, lf["verdict"], mmd["p_value"], seed)
    return {"seed": seed, "n_rows": int(len(s.control_X)), "detection": detection,
            "label_free_verdict": lf["verdict"], "diagnosis": diag.drift_type, "stage": diag.stage,
            "audit": {"accuracy_drop_significant": au["accuracy_drop_significant"],
                      "structured_features": au["structured_features"]},
            "published": pub, "seconds": round(time.perf_counter() - t0, 1)}


def run_study_window(dataset: str, model_kind: str, key: str, stored: Dict[str, Any]) -> Dict[str, Any]:
    with threadpool_limits(limits=1):
        s = setup(dataset, model_kind, STUDY_SEED)
        if key == control_key(dataset):
            X, y = s.control_X, s.control_y
        else:
            X, y = study_windows(dataset)[key]
        rng = window_rng(STUDY_SEED, f"bucket {key}" if dataset == "clear10" else key)   # the study's stream
        rng.choice(len(X), min(1000, len(X)), replace=False)      # the study's reliability records come first
        idx = rng.choice(len(X), s.audit_n(len(X)), replace=False)
        acc = float(np.mean(s.model.predict(X.iloc[idx][s.features].to_numpy(float)) == y[idx]))
        d = stored["diagnosis"]
        if not np.isclose(acc, d["audit"]["audit_accuracy"] if "audit" in d else d["audit_accuracy"], atol=1e-12):
            raise AssertionError(f"{dataset} {model_kind} {key}: audit sample differs from the study")
        prod = s.window_ds(X)
        mmd_p = stored.get("pipelines", {}).get("MMD", {}).get("detection", {}).get("p_value") \
            or stored.get("detection", {}).get("MMD", {}).get("p_value")
        pub = published(s, prod, X, y, idx, stored["diagnosis"]["label_free_verdict"], mmd_p, STUDY_SEED)
    return {"key": key, "ours": stored["diagnosis"]["final"]["drift_type"], "published": pub}


def study_output(dataset: str, model_kind: str) -> Path:
    if dataset in ("income", "employment"):
        if dataset == "income" and model_kind == "lr":
            return OUTPUTS / "folktables_drift_results.json"
        return OUTPUTS / f"folktables_{dataset}_drift_results_{model_kind}.json"
    if dataset == "gas":
        return OUTPUTS / f"gas_drift_results_{model_kind}.json"
    return OUTPUTS / ("clear10_drift_results.json" if model_kind == "lr" else f"clear10_drift_results_{model_kind}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["income", "employment", "gas", "clear10"], default="income")
    ap.add_argument("--model", choices=["lr", "hgb"], default="lr")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    setup_logging(level="WARNING")
    start = time.time()
    study = json.loads(study_output(args.dataset, args.model).read_text())["windows"]
    seeds = CONTROL_SEEDS[:1] if args.smoke else CONTROL_SEEDS
    keys = list(study)[:2] if args.smoke else list(study)
    controls = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_control)(args.dataset, args.model, k) for k in seeds)
    windows = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_study_window)(args.dataset, args.model, k, study[k]) for k in keys)
    out_dir = OUTPUTS / "real_controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.dataset}_{args.model}{'_smoke' if args.smoke else ''}.json"
    out.write_text(json.dumps({"config": {"dataset": args.dataset, "model": args.model, "alpha": ALPHA,
                                          "permutations": N_PERMUTATIONS, "control_seeds": list(seeds)},
                               "controls": controls, "windows": windows,
                               "wall_clock_seconds": round(time.time() - start, 1)}, default=json_default))
    alarms = sum(c["diagnosis"] != "none" for c in controls)
    print(f"our diagnosis: drift verdict on {alarms} of {len(controls)} extra controls")
    for n in ("roschewitz", "detectshift", "confident_learning"):
        print(f"  {n}: {sum(c['published']['verdicts'][n] != 'none' for c in controls)} controls with a verdict; "
              f"windows {[w['published']['verdicts'][n] for w in windows]}")
    print(f"wrote {out} ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
