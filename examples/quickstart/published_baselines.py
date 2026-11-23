"""Published diagnosis methods on the windows of an existing benchmark run (BNRC revision).

Rebuilds every replicate of one benchmark configuration exactly as
drift_injection_benchmark.py does (same splits, models, injections and audit
samples), checks each regenerated window against the stored record (true
accuracy and audit accuracy must match), and runs the published methods of
drift_autopsy.published_methods on it: Roschewitz et al. (2025), DetectShift
(Maia Polo et al., 2023) and confident learning (Northcutt et al., 2021) in
place of our audit stage. The label-free verdict and the MMD p-value are read
from the stored record, so nothing of ours is recomputed.

Run:  DDA_OUTPUTS=outputs/bnrc DDA_PERMUTATIONS=1000 \
      python examples/quickstart/published_baselines.py --dataset acs_income --design A --model lr
Output: {DDA_OUTPUTS}/benchmark/published/{dataset}_{design}_{model}.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from drift_autopsy.benchmark import BenchmarkConfig
from drift_autopsy.benchmark.evaluate import replicate_windows
from drift_autopsy.benchmark.window import audit_order, published_verdicts
from drift_autopsy.utils import setup_logging

OUTPUT_DIR = Path(os.environ.get("DDA_OUTPUTS", "outputs")) / "benchmark"


def _key(state, seed, mechanism, order) -> str:
    return f"{state}|{seed}|{mechanism}|{order}"


def run_replicate(state: str, seed: int, cfg: BenchmarkConfig, stored: Dict[str, Dict[str, Any]]) -> List[Dict]:
    out = []
    with threadpool_limits(limits=1):
        for rc, mechanism, drift_type, params, order_i, prod, hold in replicate_windows(
                state, seed, cfg, with_reliability=False):
            run = stored[_key(state, seed, mechanism, order_i)]
            if not np.isclose(run["true_accuracy"], float(np.mean(
                    rc.model.predict(prod.X[rc.feature_names].to_numpy(float)) == prod.y)), atol=1e-12):
                raise AssertionError(f"window {state} {seed} {mechanism} {order_i} differs from the stored run")
            idx = audit_order(rc, cfg.design, mechanism, drift_type, order_i, len(prod.X))[:rc.primary_audit_size]
            aud_acc = float(np.mean(rc.model.predict(prod.X.iloc[idx][rc.feature_names].to_numpy(float))
                                    == np.asarray(prod.y)[idx]))
            if not np.isclose(aud_acc, run["audit"]["audit_accuracy"], atol=1e-12):
                raise AssertionError(f"audit sample {state} {seed} {mechanism} {order_i} differs from the stored run")
            t0 = time.perf_counter()
            pub = published_verdicts(rc, prod.X, prod.y, idx, run["label_free"]["verdict"],
                                     run["detection"]["MMD"]["p_value"])
            out.append({"state": state, "seed": seed, "mechanism": mechanism, "intensity_order": order_i,
                        "intensity_label": params["label"], "drift_type": drift_type,
                        "ours": run["diagnosis"]["drift_type"], "published": pub,
                        "seconds": round(time.perf_counter() - t0, 2)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["acs_income", "acs_employment"], default="acs_income")
    ap.add_argument("--design", choices=["A", "B"], default="A")
    ap.add_argument("--model", choices=["lr", "hgb"], default="lr")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="one replicate only")
    args = ap.parse_args()
    setup_logging(level="WARNING")

    cfg = BenchmarkConfig(dataset=args.dataset, design=args.design, model=args.model,
                          mmd_permutations=int(os.environ.get("DDA_PERMUTATIONS", 200)))
    runs = json.loads((OUTPUT_DIR / f"{args.dataset}_{args.design}_{args.model}_runs.json").read_text())
    stored = {_key(r["state"], r["seed"], r["mechanism"], r["intensity_order"]): r for r in runs}
    jobs = [(s, k) for s in cfg.states for k in cfg.seeds]
    if args.smoke:
        jobs = jobs[:1]
    start = time.time()
    per_job = Parallel(n_jobs=args.n_jobs, verbose=10)(delayed(run_replicate)(s, k, cfg, stored) for s, k in jobs)
    records = [r for job in per_job for r in job]
    out_dir = OUTPUT_DIR / "published"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.dataset}_{args.design}_{args.model}{'_smoke' if args.smoke else ''}.json"
    out.write_text(json.dumps({"config": cfg.to_dict(), "records": records,
                               "wall_clock_seconds": round(time.time() - start, 1)}))
    for name in ("roschewitz", "detectshift", "confident_learning"):
        ok = np.mean([r["published"]["verdicts"][name] == r["drift_type"] for r in records])
        print(f"{name:20s} correct {ok:.3f}")
    print(f"ours                 correct {np.mean([r['ours'] == r['drift_type'] for r in records]):.3f}")
    print(f"wrote {out} ({len(records)} windows, {time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
