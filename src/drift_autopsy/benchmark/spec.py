"""Ground-truth description of one injected drift.

The benchmark injects drift we designed into a real reference distribution, so
for every production window we know exactly what changed: which of P(X) and
P(Y|X) moved, which features carry the change, the class prior before and
after, and the accuracy the deployed model actually loses. The pipeline never
sees any of this -- it is the answer key the diagnosis is graded against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

DRIFT_TYPES = ("none", "covariate", "prior", "concept", "label_noise")

# What a correct label-free diagnosis should conclude for each injected type.
EXPECTED_DIAGNOSIS: Dict[str, str] = {
    "none": "no action - distributions match the reference",
    "covariate": "importance-weighted retrain (no production labels needed)",
    "prior": "re-weight or recalibrate the decision threshold to the new base rate",
    "concept": "collect fresh labels - retraining on the old target rule will not help",
    "label_noise": "audit the labelling pipeline - the model itself is not at fault",
}


@dataclass
class InjectedDriftSpec:
    """The answer key for one production window."""

    drift_type: str
    intensity_label: str
    params: Dict[str, float] = field(default_factory=dict)

    affected_features: List[str] = field(default_factory=list)
    px_changed: bool = False
    pygivenx_changed: bool = False

    p_y_reference: float = 0.0
    p_y_production: float = 0.0

    reference_accuracy: float = 0.0
    production_accuracy: float = 0.0

    def __post_init__(self) -> None:
        if self.drift_type not in DRIFT_TYPES:
            raise ValueError(f"unknown drift_type {self.drift_type!r}")

    @property
    def true_accuracy_drop(self) -> float:
        return float(self.reference_accuracy - self.production_accuracy)

    @property
    def prior_shift(self) -> float:
        return float(self.p_y_production - self.p_y_reference)

    @property
    def expected_diagnosis(self) -> str:
        return EXPECTED_DIAGNOSIS[self.drift_type]

    @property
    def is_feature_localised(self) -> bool:
        return bool(self.affected_features)

    def to_dict(self) -> Dict[str, object]:
        return {
            "drift_type": self.drift_type,
            "intensity_label": self.intensity_label,
            "params": {k: _round(v) for k, v in self.params.items()},
            "affected_features": list(self.affected_features),
            "px_changed": bool(self.px_changed),
            "pygivenx_changed": bool(self.pygivenx_changed),
            "p_y_reference": _round(self.p_y_reference),
            "p_y_production": _round(self.p_y_production),
            "prior_shift": _round(self.prior_shift),
            "reference_accuracy": _round(self.reference_accuracy),
            "production_accuracy": _round(self.production_accuracy),
            "true_accuracy_drop": _round(self.true_accuracy_drop),
            "expected_diagnosis": self.expected_diagnosis,
        }


def _round(value: float, ndigits: int = 5) -> float:
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value
