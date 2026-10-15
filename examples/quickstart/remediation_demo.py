"""Remediation strategies and the label-free triage on the real drift settings.

Settings (every available window, none hand-picked):
* ACS temporal   - production = CA 2015 to 2019 and 2021 to 2023; the earlier
                   years from 2015 on are delayed-label windows (for both ACS
                   tasks, Income and Employment);
* ACS geographic - production = TX, NY, FL, WA 2014; no delayed windows;
* CLEAR-10       - production = bucket 2 ... 10; buckets 2 .. k-1 are
                   delayed-label windows;
* gas sensor     - production = batch 3 ... 10; batches 3 .. k-1 are
                   delayed-label windows.

In each setting the deployed model is the study's monitored model (trained on
the reference train split). The production window is split in half: an
unlabelled production sample (density ratio; the label-buying strategies draw
their labels from it) and a labelled holdout used only for grading. Every
strategy's recovery is tested against the deployed model with a paired
bootstrap; the triage verdict is graded against whether either label-free
probe strategy (importance weighting, retrain on recent) truly helps.
Run:    python examples/quickstart/remediation_demo.py --model lr
Output: outputs/remediation_results.json (lr) or outputs/remediation_results_{model}.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from drift_autopsy.experiments import acs, clear10, gas
from drift_autopsy.experiments.common import json_default
from drift_autopsy.experiments.models import MODEL_DESCRIPTIONS
from drift_autopsy.remediation import RemediationContext, remediation_triage, run_remediation_suite
from drift_autopsy.remediation import triage as triage_module
from drift_autopsy.utils import setup_logging

SEED = 42
ALPHA = 0.05
N_LABELS = 2000          # fresh labels bought by label_retrain / recalibrate
STRATEGIES = ["importance_weighted_retrain", "retrain_on_recent", "full_retrain", "feature_drop_retrain",
              "label_retrain", "recalibrate"]
PROBE_STRATEGIES = set(triage_module.PROBE_STRATEGIES)
ACS_YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023]
GEO_STATES = ["TX", "NY", "FL", "WA"]
ACS_GROUPS = {"income": "acs", "employment": "acs_employment"}


def results_path(study: str, model: str) -> Path:
    suffix = "" if model == "lr" else f"_{model}"
    return {
        "income": Path("outputs/folktables_drift_results.json" if model == "lr"
                       else f"outputs/folktables_income_drift_results_{model}.json"),
        "employment": Path(f"outputs/folktables_employment_drift_results_{model}.json"),
        "clear10": Path(f"outputs/clear10_drift_results{suffix}.json"),
        "gas": Path(f"outputs/gas_drift_results_{model}.json"),
    }[study]


def _half(X, y, seed):
    return train_test_split(X, y, test_size=0.5, random_state=seed, stratify=y)


def run_setting(ctx: RemediationContext, group: str) -> Dict[str, Any]:
    triage = remediation_triage(ctx, alpha=ALPHA)
    suite = run_remediation_suite(ctx, STRATEGIES)
    results = [r.to_dict() for r in suite["results"]]
    probes = [r for r in results if r["strategy"] in PROBE_STRATEGIES]
    truth = any(r["recovery_significant"] and r["accuracy_recovered"] > 0 for r in probes)
    print(f"\n=== {ctx.shift_name}: before {ctx.baseline_before:.4f} (reference {ctx.reference_accuracy:.4f}); "
          f"triage {'escalate' if triage['will_retraining_help'] else 'hold'}; truth {'helps' if truth else 'no'}")
    for r in results:
        t = r["recovery_test"]
        print(f"    {r['strategy']:28s} {r['accuracy_recovered']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] "
              f"{'*' if r['recovery_significant'] else ' '} {r['wall_clock_seconds']:.3f}s labels={r['n_production_labels_required']}")
    return {
        "group": group,
        "shift_name": ctx.shift_name,
        "reference_accuracy": ctx.reference_accuracy,
        "accuracy_before": ctx.baseline_before,
        "gap": ctx.reference_accuracy - ctx.baseline_before,
        "n_holdout": int(len(ctx.holdout_y)),
        "triage": triage,
        "results": results,
        "skipped": suite["skipped"],
        "probe_truth_helps": bool(truth),
        "triage_correct": bool(triage["will_retraining_help"] == truth),
    }


def acs_settings(task: str, kind: str) -> List[Dict[str, Any]]:
    drift = json.loads(results_path(task, kind).read_text())
    features = list(acs.ACS_TASKS[task].features)
    X14, y14 = acs.load_acs_task(task, 2014, "CA")
    split = acs.split_reference(X14, y14, seed=SEED)
    def factory(keep_columns=None):
        return acs.make_acs_model(random_state=SEED, keep_columns=keep_columns, kind=kind, task=task)
    with threadpool_limits(limits=1):
        model = factory().fit(split.train_X.to_numpy(float), split.train_y)
    ref_acc = float(drift["reference"]["test_accuracy"])
    ref_X, ref_y = split.train_X.to_numpy(float), split.train_y
    title, group = acs.ACS_TASKS[task].title, ACS_GROUPS[task]

    def drifted(key):
        return list(drift["windows"][key]["pipelines"]["KS Test"]["localization"]["drifted_features"])

    out = []
    windows = {y: acs.load_acs_task(task, y, "CA") for y in ACS_YEARS}
    for i, year in enumerate(ACS_YEARS):
        X, y = windows[year]
        Xp, Xh, yp, yh = _half(X.to_numpy(float), y, SEED)
        recent = [(windows[p][0].to_numpy(float), windows[p][1]) for p in ACS_YEARS[:i]]
        ctx = RemediationContext(ref_X, ref_y, Xp, Xh, yh, features, drifted(f"CA {year}"),
                                 f"{title} temporal CA {year}", model, factory, ref_acc, recent_windows=recent,
                                 production_y=yp, seed=SEED, n_labels=N_LABELS)
        out.append(run_setting(ctx, f"{group}_temporal"))
    for state in GEO_STATES:
        X, y = acs.load_acs_task(task, 2014, state)
        Xp, Xh, yp, yh = _half(X.to_numpy(float), y, SEED)
        ctx = RemediationContext(ref_X, ref_y, Xp, Xh, yh, features, drifted(f"{state} 2014"),
                                 f"{title} geographic {state} 2014", model, factory, ref_acc, production_y=yp,
                                 seed=SEED, n_labels=N_LABELS)
        out.append(run_setting(ctx, f"{group}_geographic"))
    return out


def clear_settings(kind: str) -> List[Dict[str, Any]]:
    drift = json.loads(results_path("clear10", kind).read_text())
    b1 = clear10.load_bucket(clear10.REFERENCE_BUCKET)
    split = clear10.split_reference(b1, seed=SEED)
    def factory():
        return clear10.make_clear_model(random_state=SEED, kind=kind)
    with threadpool_limits(limits=1):
        model = factory().fit(clear10.embeddings(split.train), split.train["y_true"])
    ref_acc = float(drift["reference"]["test_accuracy"])
    ref_X, ref_y = clear10.embeddings(split.train), split.train["y_true"].to_numpy()
    frames = {b: clear10.load_bucket(b) for b in clear10.ANALYSIS_BUCKETS}
    out = []
    for i, b in enumerate(clear10.ANALYSIS_BUCKETS):
        f = frames[b]
        Xp, Xh, yp, yh = _half(clear10.embeddings(f), f["y_true"].to_numpy(), SEED)
        recent = [(clear10.embeddings(frames[p]), frames[p]["y_true"].to_numpy()) for p in clear10.ANALYSIS_BUCKETS[:i]]
        ctx = RemediationContext(ref_X, ref_y, Xp, Xh, yh, clear10.EMBEDDING_COLUMNS, [],
                                 f"CLEAR-10 bucket {b}", model, factory, ref_acc, recent_windows=recent,
                                 production_y=yp, seed=SEED, n_labels=N_LABELS)
        out.append(run_setting(ctx, "clear10"))
    return out


def gas_settings(kind: str) -> List[Dict[str, Any]]:
    drift = json.loads(results_path("gas", kind).read_text())
    data = gas.load_batches()
    split = gas.split_reference(data[data["batch"].isin(gas.REFERENCE_BATCHES)].reset_index(drop=True), seed=SEED)
    def factory(keep_columns=None):
        return gas.make_gas_model(random_state=SEED, kind=kind)
    with threadpool_limits(limits=1):
        model = factory().fit(gas.features(split.train), split.train["gas"])
    ref_acc = float(drift["reference"]["test_accuracy"])
    ref_X, ref_y = gas.features(split.train), split.train["gas"].to_numpy()
    frames = {b: data[data["batch"] == b].reset_index(drop=True) for b in gas.ANALYSIS_BATCHES}
    out = []
    for i, b in enumerate(gas.ANALYSIS_BATCHES):
        f = frames[b]
        Xp, Xh, yp, yh = _half(gas.features(f), f["gas"].to_numpy(), SEED)
        recent = [(gas.features(frames[p]), frames[p]["gas"].to_numpy()) for p in gas.ANALYSIS_BATCHES[:i]]
        drifted = drift["windows"][f"batch {b}"]["localization"]["drifted_features"]
        ctx = RemediationContext(ref_X, ref_y, Xp, Xh, yh, gas.FEATURES, drifted, f"Gas sensor batch {b}",
                                 model, factory, ref_acc, recent_windows=recent, production_y=yp, seed=SEED, n_labels=N_LABELS)
        out.append(run_setting(ctx, "gas"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODEL_DESCRIPTIONS), default="lr")
    args = ap.parse_args()
    setup_logging(level="WARNING")
    settings = (acs_settings("income", args.model) + acs_settings("employment", args.model)
                + clear_settings(args.model) + gas_settings(args.model))
    n_correct = sum(s["triage_correct"] for s in settings)
    payload = {
        "config": {"model_kind": args.model, "model": MODEL_DESCRIPTIONS[args.model], "seed": SEED, "alpha": ALPHA, "n_labels_bought": N_LABELS, "strategies": STRATEGIES,
                   "probe_strategies": sorted(PROBE_STRATEGIES),
                   "grading": "paired bootstrap on a labelled half of each production window"},
        "settings": settings,
        "summary": {"n_settings": len(settings), "triage_correct": n_correct},
    }
    out = Path("outputs/remediation_results.json" if args.model == "lr" else f"outputs/remediation_results_{args.model}.json")
    out.write_text(json.dumps(payload, indent=1, allow_nan=False, default=json_default))
    print(f"\ntriage correct in {n_correct}/{len(settings)} settings; wrote {out}")


if __name__ == "__main__":
    main()
