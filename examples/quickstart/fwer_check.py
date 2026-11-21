"""Family-wise error of the label-free decisions, by simulation.

Scenario: one feature truly shifted (p = 1e-12); the other features, the
prediction test and the MMD test are true nulls with independent uniform
p-values. An error is any rejection of a true null. Two procedures:

* ``weighted_holm`` - the method (drift_autopsy.diagnosis.label_free_decisions);
* ``gatekeeping``   - the rule used before the correction: Holm over the three
  block tests, then Holm within the feature block at the level the block was
  rejected at. It spends alpha twice once one feature is rejected.

Run:  python examples/quickstart/fwer_check.py
Output: outputs/fwer_check.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from drift_autopsy.diagnosis import label_free_decisions
from drift_autopsy.stats import holm_reject

ALPHA = 0.05
N_FEATURES = (10, 16)          # ACS Income and ACS Employment
REPLICATES = 200_000
SEED = 0


def gatekeeping(fp: np.ndarray, pp: float, mp: float, alpha: float):
    block_p = [min(1.0, len(fp) * fp.min()), pp, mp]
    block = holm_reject(block_p, alpha)
    ranks = np.empty(3, dtype=int)
    ranks[np.argsort(block_p, kind="stable")] = np.arange(3)
    features = holm_reject(fp, alpha / (3 - ranks[0])) if block[0] else np.zeros(len(fp), dtype=bool)
    return features, block[1], block[2]


def weighted_holm(fp: np.ndarray, pp: float, mp: float, alpha: float):
    d = label_free_decisions(fp, pp, mp, alpha)
    return d["feature_reject"], d["prediction"], d["multivariate"]


def main() -> None:
    out = {"alpha": ALPHA, "replicates": REPLICATES, "seed": SEED,
           "scenario": "one shifted feature (p = 1e-12); every other test a true null", "results": {}}
    for m in N_FEATURES:
        res = {}
        for name, proc in (("weighted_holm", weighted_holm), ("gatekeeping", gatekeeping)):
            rng = np.random.default_rng([SEED, m])
            errors = 0
            for _ in range(REPLICATES):
                fp = np.r_[1e-12, rng.random(m - 1)]
                features, pred, mmd = proc(fp, rng.random(), rng.random(), ALPHA)
                errors += bool(features[1:].any() or pred or mmd)
            rate = errors / REPLICATES
            res[name] = {"fwer": rate, "mc_se": float(np.sqrt(rate * (1 - rate) / REPLICATES))}
        out["results"][str(m)] = res
        print(f"m={m}: weighted Holm {res['weighted_holm']['fwer']:.4f}, gatekeeping {res['gatekeeping']['fwer']:.4f}")
    path = Path("outputs/fwer_check.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
