"""Controlled drift-injection benchmark.

Injects drift with a known ground truth into 2014 ACS data (five states, ten
seeds), runs the monitor on every window and grades it against the answer key.

* ``--dataset``  acs_income or acs_employment
* ``--design``   A (the original mechanisms: covariate, prior, concept and label
  noise at three intensities plus a no-drift control) or B (held-out
  mechanisms that do not match the tests; see drift_autopsy.benchmark.injectors)
* ``--model``    lr or hgb (drift_autopsy.experiments.models)

Writes outputs/benchmark/{dataset}_{design}_{model}.json (config, summary) and
the raw window records next to it (``*_runs.json``). The paper's tables and
figures are rendered from these files by examples/paper/build_artifacts.py.

Run:  python examples/quickstart/drift_injection_benchmark.py --dataset acs_income --design A --model lr
      add --smoke for one state and one seed (pipeline check only)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from drift_autopsy.benchmark import BenchmarkConfig, run_benchmark
from drift_autopsy.utils import setup_logging

OUTPUT_DIR = Path("outputs/benchmark")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one state, one seed (pipeline check only)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--dataset", choices=["acs_income", "acs_employment"], default="acs_income")
    ap.add_argument("--design", choices=["A", "B"], default="A")
    ap.add_argument("--model", choices=["lr", "hgb"], default="lr")
    args = ap.parse_args()
    setup_logging(level="WARNING")

    cfg = BenchmarkConfig(dataset=args.dataset, design=args.design, model=args.model, n_jobs=args.n_jobs)
    if args.smoke:
        cfg.states, cfg.seeds = ("CA",), (0,)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{args.dataset}_{args.design}_{args.model}{'_smoke' if args.smoke else ''}.json"
    raw = out.with_name(out.stem + "_runs.json")

    def save_raw(runs):
        raw.write_text(json.dumps(runs, default=_json_default))
        print(f"saved {len(runs)} raw window records to {raw}")

    start = time.time()
    result = run_benchmark(cfg, progress=print, on_runs=save_raw)
    result["wall_clock_seconds"] = round(time.time() - start, 1)

    out.write_text(json.dumps(result, indent=1, allow_nan=False, default=_json_default))

    s = result["summary"]
    ti = s["type_identification"]
    print(f"\n{s['n_runs']} windows over {s['n_replicates']} replicates in {result['wall_clock_seconds']} s")
    print(f"type identification: {ti['overall']['rate']:.3f} {ti['overall']['ci']}")
    for t, v in ti["per_type"].items():
        if v["n"]:
            print(f"  {t:12s} {v['k']}/{v['n']} = {v['rate']:.3f}  stages {ti['stage_counts'][t]}")
    for m, v in s["methods"].get("scores", {}).items():
        print(f"  method {m:24s} accuracy {v['accuracy']['rate']:.3f}  macro F1 {v['macro_f1']:.3f}")
    print("detector false alarms on controls:", {k: v["rate"] for k, v in s["detector_false_alarms"].items()})
    print("triage agreement:", s["triage"]["agreement"], s["triage"]["confusion"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
