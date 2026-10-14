"""Remediation strategies and the label-free triage on the real drift settings.

Settings (every available window, none hand-picked):
* ACS temporal   - production = CA 2015, 2016, 2017, 2018; the earlier years
                   from 2015 on are delayed-label windows;
* ACS geographic - production = TX, NY, FL, WA 2014; no delayed windows;
* CLEAR-10       - production = bucket 2 ... 10; buckets 2 .. k-1 are
                   delayed-label windows.

In each setting the deployed model is the study's monitored model (trained on
the reference train split). The production window is split in half: an
unlabelled production sample (density ratio; the label-buying strategies draw
their labels from it) and a labelled holdout used only for grading. Every
strategy's recovery is tested against the deployed model with a paired
bootstrap; the triage verdict is graded against whether either label-free
probe strategy (importance weighting, retrain on recent) truly helps.
Output: outputs/remediation_results.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from drift_autopsy.experiments import acs, clear10
from drift_autopsy.remediation import RemediationContext, remediation_triage, run_remediation_suite
from drift_autopsy.utils import setup_logging

SEED = 42
ALPHA = 0.05
N_LABELS = 2000          # fresh labels bought by label_retrain / recalibrate
STRATEGIES = ["importance_weighted_retrain", "retrain_on_recent", "full_retrain", "feature_drop_retrain",
              "label_retrain", "recalibrate"]
PROBE_STRATEGIES = {"importance_weighted_retrain", "retrain_on_recent"}
ACS_RESULTS = Path("outputs/folktables_drift_results.json")
CLEAR_RESULTS = Path("outputs/clear10_drift_results.json")
OUTPUT = Path("outputs/remediation_results.json")


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


def acs_settings() -> List[Dict[str, Any]]:
    drift = json.loads(ACS_RESULTS.read_text())
    X14, y14 = acs.load_acs(2014, "CA")
    split = acs.split_reference(X14, y14, seed=SEED)
    factory = lambda keep_columns=None: acs.make_acs_model(random_state=SEED, keep_columns=keep_columns)
    with threadpool_limits(limits=1):
        model = factory().fit(split.train_X.to_numpy(float), split.train_y)
    ref_acc = float(drift["reference"]["test_accuracy"])
    ref_X, ref_y = split.train_X.to_numpy(float), split.train_y

    def drifted(key):
        return list(drift["windows"][key]["pipelines"]["KS Test"]["localization"]["drifted_features"])

    out = []
    years = [2015, 2016, 2017, 2018]
    windows = {y: acs.load_acs(y, "CA") for y in years}
    for i, year in enumerate(years):
        X, y = windows[year]
        Xp, Xh, yp, yh = _half(X.to_numpy(float), y, SEED)
        recent = [(windows[p][0].to_numpy(float), windows[p][1]) for p in years[:i]]
        ctx = RemediationContext(ref_X, ref_y, Xp, Xh, yh, acs.ACS_FEATURES, drifted(f"CA {year}"),
                                 f"ACS temporal CA {year}", model, factory, ref_acc, recent_windows=recent,
                                 production_y=yp, seed=SEED)
        out.append(run_setting(ctx, "acs_temporal"))
    for state in ["TX", "NY", "FL", "WA"]:
        X, y = acs.load_acs(2014, state)
        Xp, Xh, yp, yh = _half(X.to_numpy(float), y, SEED)
        ctx = RemediationContext(ref_X, ref_y, Xp, Xh, yh, acs.ACS_FEATURES, drifted(f"{state} 2014"),
                                 f"ACS geographic {state} 2014", model, factory, ref_acc, production_y=yp, seed=SEED)
        out.append(run_setting(ctx, "acs_geographic"))
    return out


def clear_settings() -> List[Dict[str, Any]]:
    drift = json.loads(CLEAR_RESULTS.read_text())
    b1 = clear10.load_bucket(clear10.REFERENCE_BUCKET)
    split = clear10.split_reference(b1, seed=SEED)
    factory = lambda: clear10.make_clear_model(random_state=SEED)
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
                                 production_y=yp, seed=SEED)
        out.append(run_setting(ctx, "clear10"))
    return out


def main() -> None:
    setup_logging(level="WARNING")
    settings = acs_settings() + clear_settings()
    n_correct = sum(s["triage_correct"] for s in settings)
    payload = {
        "config": {"seed": SEED, "alpha": ALPHA, "n_labels_bought": N_LABELS, "strategies": STRATEGIES,
                   "probe_strategies": sorted(PROBE_STRATEGIES),
                   "grading": "paired bootstrap on a labelled half of each production window"},
        "settings": settings,
        "summary": {"n_settings": len(settings), "triage_correct": n_correct},
    }
    OUTPUT.write_text(json.dumps(payload, indent=1, allow_nan=False, default=_json_default))
    print(f"\ntriage correct in {n_correct}/{len(settings)} settings; wrote {OUTPUT}")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


if __name__ == "__main__":
    main()
