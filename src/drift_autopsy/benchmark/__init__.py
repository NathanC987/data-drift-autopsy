"""Controlled drift-injection benchmark.

Injects drift with a known ground truth into ACS Income reference
distributions (several states, several seeds) and grades the label-free
diagnosis, the delayed-label audit, the CBPE accuracy estimate, per-prediction
reliability and the remediation triage against the answer key.
"""

from drift_autopsy.benchmark.evaluate import INTENSITY_LADDER, BenchmarkConfig, run_benchmark, run_replicate
from drift_autopsy.benchmark.injectors import (
    InjectionResult,
    inject_concept,
    inject_covariate,
    inject_label_noise,
    inject_none,
    inject_prior,
)
from drift_autopsy.benchmark.metrics import aggregate
from drift_autopsy.benchmark.spec import DRIFT_TYPES, EXPECTED_DIAGNOSIS, InjectedDriftSpec

__all__ = [
    "BenchmarkConfig",
    "INTENSITY_LADDER",
    "run_benchmark",
    "run_replicate",
    "aggregate",
    "DRIFT_TYPES",
    "EXPECTED_DIAGNOSIS",
    "InjectedDriftSpec",
    "InjectionResult",
    "inject_none",
    "inject_covariate",
    "inject_prior",
    "inject_concept",
    "inject_label_noise",
]
