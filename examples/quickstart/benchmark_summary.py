"""Cross-configuration analyses of the injection benchmark.

Reads every outputs/benchmark/{dataset}_{design}_{model}.json (and the raw
window records next to it) and writes outputs/benchmark/summary.json with:

* ``methods``   - per configuration, the scores of ours, the rule baselines,
  the ablations and the learned meta-classifier, with paired sign-flip tests
  against ours (Holm across the comparisons of a configuration). In design A
  the meta-classifier is validated leave one state out. In design B it is
  trained on the design A windows of the same dataset and model, again
  leaving out the state it predicts, so it never sees a held-out mechanism.
* ``mechanisms`` - design B correct-type rate of ours per mechanism.

Run after the benchmark:  python examples/quickstart/benchmark_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from threadpoolctl import threadpool_limits

from drift_autopsy.benchmark.metrics import (
    meta_classifier_predictions,
    method_scores,
    paired_method_tests,
    rule_predictions,
)
from drift_autopsy.experiments.common import json_default
from drift_autopsy.stats import wilson_interval

DIR = Path("outputs/benchmark")
DATASETS = ("acs_income", "acs_employment")
MODELS = ("lr", "hgb")
ALPHA = 0.05


def runs_of(dataset: str, design: str, model: str) -> List[Dict[str, Any]]:
    path = DIR / f"{dataset}_{design}_{model}_runs.json"
    return json.loads(path.read_text()) if path.exists() else []


def meta_transfer(train: List[Dict[str, Any]], test: List[Dict[str, Any]]) -> List[str]:
    """Meta-classifier trained on design A, applied to design B, state held out."""
    pred = [None] * len(test)
    for s in sorted({r["state"] for r in test}):
        tr = [r for r in train if r["state"] != s]
        idx = [i for i, r in enumerate(test) if r["state"] == s]
        for i, p in zip(idx, meta_classifier_predictions(tr, [test[i] for i in idx])):
            pred[i] = p
    return pred


def main() -> None:
    out: Dict[str, Any] = {"alpha": ALPHA, "methods": {}, "mechanisms": {}}
    for ds in DATASETS:
        for model in MODELS:
            runs_a = runs_of(ds, "A", model)
            runs_b = runs_of(ds, "B", model)
            for design, runs in (("A", runs_a), ("B", runs_b)):
                if not runs:
                    continue
                preds = rule_predictions(runs)
                if design == "A":
                    preds["meta_classifier"] = meta_classifier_predictions(runs)
                elif runs_a:
                    preds["meta_classifier"] = meta_transfer(runs_a, runs)
                key = f"{ds}_{design}_{model}"
                out["methods"][key] = {
                    "n_windows": len(runs),
                    "n_replicates": len({(r["state"], r["seed"]) for r in runs}),
                    "scores": method_scores(runs, preds),
                    "paired_tests": paired_method_tests(runs, preds, alpha=ALPHA),
                    "meta_validation": ("leave one state out" if design == "A"
                                        else "trained on design A, state held out"),
                }
                if design == "B":
                    mech: Dict[str, Any] = {}
                    for m in sorted({r["mechanism"] for r in runs}):
                        sub = [r for r in runs if r["mechanism"] == m]
                        k = sum(r["verdicts"]["ours"] == r["drift_type"] for r in sub)
                        lo, hi = wilson_interval(k, len(sub))
                        diagnosed = {}
                        for r in sub:
                            diagnosed[r["verdicts"]["ours"]] = diagnosed.get(r["verdicts"]["ours"], 0) + 1
                        mech[m] = {
                            "drift_type": sub[0]["drift_type"], "k": k, "n": len(sub), "rate": k / len(sub),
                            "ci": [lo, hi], "diagnosed": diagnosed,
                            "mean_true_drop": float(np.mean([r["true_accuracy_drop"] for r in sub])),
                            "per_intensity": {
                                lab: float(np.mean([r["verdicts"]["ours"] == r["drift_type"]
                                                    for r in sub if r["intensity_label"] == lab]))
                                for lab in ("mild", "moderate", "strong")},
                            "mean_true_drop_per_intensity": {
                                lab: float(np.mean([r["true_accuracy_drop"] for r in sub
                                                    if r["intensity_label"] == lab]))
                                for lab in ("mild", "moderate", "strong")},
                        }
                    out["mechanisms"][f"{ds}_{model}"] = mech
    path = DIR / "summary.json"
    path.write_text(json.dumps(out, indent=1, allow_nan=False, default=json_default))
    for key, v in out["methods"].items():
        s = v["scores"]
        line = "  ".join(f"{m} {s[m]['accuracy']['rate']:.3f}" for m in s)
        print(f"{key:28s} {line}")
    print(f"wrote {path}")


if __name__ == "__main__":
    with threadpool_limits(limits=1):     # many small boosted fits; threads only add contention
        main()
