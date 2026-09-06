"""Backwards-compatible entry point for the dashboard's "retrain" button.

The remediation logic now lives in ``drift_autopsy.remediation`` and is exercised
by ``examples/quickstart/remediation_demo.py``. This shim keeps the old path
(``examples/quickstart/retrain_demo.py``) working: the dashboard calls it by
path and may pass ``--strategy`` / ``--drop-features``.
"""

import argparse
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="full_retrain")
    ap.add_argument("--drop-features", default="")
    ap.add_argument("--shift", default="temporal")
    args, _ = ap.parse_known_args()

    forwarded = ["--shift", args.shift]
    if args.drop_features:
        forwarded += ["--drop-features", args.drop_features]

    sys.argv = [str(Path(__file__).with_name("remediation_demo.py"))] + forwarded
    runpy.run_path(sys.argv[0], run_name="__main__")
