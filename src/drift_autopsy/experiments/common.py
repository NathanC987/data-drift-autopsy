"""Pieces shared by the natural drift studies (ACS, gas sensor, CLEAR-10)."""

from __future__ import annotations

import zlib
from typing import Any, Dict

import numpy as np

from drift_autopsy.detectors import BBSD, DomainClassifier
from drift_autopsy.diagnosis import audit_diagnosis
from drift_autopsy.diagnosis_baselines import all_rule_verdicts, window_tests
from drift_autopsy.estimation import label_free_estimates

AUDIT_ABLATIONS = {"no_confidence": "none", "class_confidence": "class_confidence"}


def window_rng(seed: int, key: str) -> np.random.Generator:
    """Independent random stream for one window, so adding or removing a window
    never changes another window's audit sample or stored records."""
    return np.random.default_rng([seed, zlib.crc32(key.encode("utf-8"))])


def baseline_detectors(alpha: float, seed: int, max_samples: int = 3000) -> Dict[str, Any]:
    """The detection baselines of Rabanser et al. (2019) added to every study."""
    return {
        "BBSD": BBSD(threshold=alpha),
        "Domain classifier": DomainClassifier(threshold=alpha, max_samples=max_samples, random_state=seed),
    }


def comparison_block(*, lf: Dict[str, Any], audit: Dict[str, Any], model: Any, reference_ds: Any,
                     audit_X: Any, audit_y: np.ndarray, alpha: float, cbpe_estimate: float,
                     reference_accuracy: float, production_probabilities: np.ndarray,
                     actual_accuracy: float) -> Dict[str, Any]:
    """Baseline and ablation verdicts, and the label-free accuracy estimates, for one window.

    The audit ablations reuse the same audit rows as the method."""
    variants = {k: audit_diagnosis(model, reference_ds, audit_X, audit_y, alpha=alpha, null=null)
                for k, null in AUDIT_ABLATIONS.items()}
    tests = window_tests(lf, audit, cbpe_estimate - reference_accuracy, audit_variants=variants)
    cal_correct = (np.asarray(reference_ds.predictions) == np.asarray(reference_ds.target)).astype(float)
    estimates = label_free_estimates(reference_ds.prediction_probabilities, cal_correct, production_probabilities)
    return {
        "tests": tests,
        "verdicts": all_rule_verdicts(tests, alpha),
        "estimates": estimates,
        "estimate_errors": {k: v - actual_accuracy for k, v in estimates.items()},
    }


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)}")
