"""Controlled drift-injection benchmark.

Injects drift with a known ground truth into a real reference distribution and
grades every pipeline stage -- detection, localisation, root cause, the
label-free accuracy estimate, per-prediction reliability, and remediation
triage -- against that answer key. It is the evaluator-facing verification that
the diagnosis the pipeline produces is the correct one.
"""

from drift_autopsy.benchmark.audit import labelled_audit
from drift_autopsy.benchmark.evaluate import (
    BenchmarkData,
    load_acs_reference,
    run_benchmark,
)
from drift_autopsy.benchmark.injectors import (
    InjectionResult,
    inject_concept,
    inject_covariate,
    inject_label_noise,
    inject_none,
    inject_prior,
)
from drift_autopsy.benchmark.metrics import (
    estimator_error,
    localisation_prf,
    type_identification_summary,
)
from drift_autopsy.benchmark.pipeline_probe import (
    label_free_accuracy_estimate,
    run_pipeline_probe,
)
from drift_autopsy.benchmark.presentation import build_presentation
from drift_autopsy.benchmark.signature import classify_drift_type, drift_signature
from drift_autopsy.benchmark.spec import DRIFT_TYPES, EXPECTED_DIAGNOSIS, InjectedDriftSpec

__all__ = [
    "BenchmarkData",
    "DRIFT_TYPES",
    "EXPECTED_DIAGNOSIS",
    "InjectedDriftSpec",
    "InjectionResult",
    "inject_none",
    "inject_covariate",
    "inject_prior",
    "inject_concept",
    "inject_label_noise",
    "labelled_audit",
    "label_free_accuracy_estimate",
    "run_pipeline_probe",
    "build_presentation",
    "drift_signature",
    "classify_drift_type",
    "localisation_prf",
    "estimator_error",
    "type_identification_summary",
    "load_acs_reference",
    "run_benchmark",
]
