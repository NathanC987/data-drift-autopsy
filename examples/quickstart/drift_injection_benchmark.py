"""Controlled drift-injection benchmark.

Injects covariate, prior, concept and label-noise drift (three intensities
each, plus a no-drift control) into ACS Income 2014 for five states and ten
seeds, runs the monitor on every window, and grades it against the known
ground truth. Writes outputs/drift_injection_benchmark.json; the paper's
tables and figures are rendered from that file by examples/paper/build_artifacts.py.

Run:  python examples/quickstart/drift_injection_benchmark.py            # full study
      python examples/quickstart/drift_injection_benchmark.py --smoke    # 1 state, 1 seed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from drift_autopsy.benchmark import BenchmarkConfig, run_benchmark
from drift_autopsy.utils import setup_logging

OUTPUT = Path("outputs/drift_injection_benchmark.json")


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
    args = ap.parse_args()
    setup_logging(level="WARNING")

    cfg = BenchmarkConfig(n_jobs=args.n_jobs)
    if args.smoke:
        cfg.states, cfg.seeds = ("CA",), (0,)
    out = OUTPUT if not args.smoke else OUTPUT.with_name("drift_injection_benchmark_smoke.json")
    raw = out.with_name(out.stem + "_runs.json")

    def save_raw(runs):
        raw.write_text(json.dumps(runs, default=_json_default))
        print(f"saved {len(runs)} raw window records to {raw}")

    start = time.time()
    result = run_benchmark(cfg, progress=print, on_runs=save_raw)
    result["wall_clock_seconds"] = round(time.time() - start, 1)

    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1, allow_nan=False, default=_json_default))

    s = result["summary"]
    ti = s["type_identification"]
    print(f"\n{s['n_runs']} windows over {s['n_replicates']} replicates in {result['wall_clock_seconds']} s")
    print(f"type identification: {ti['overall']['rate']:.3f} {ti['overall']['ci']}")
    for t, v in ti["per_type"].items():
        print(f"  {t:12s} {v['k']}/{v['n']} = {v['rate']:.3f}  stages {ti['stage_counts'][t]}")
    print("detector false alarms on controls:", {k: v["rate"] for k, v in s["detector_false_alarms"].items()})
    print("triage agreement:", s["triage"]["agreement"], s["triage"]["confusion"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
