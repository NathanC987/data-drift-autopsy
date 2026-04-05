"""Risk scoring engine for reliability and hallucination detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RiskWeights:
    """Weights for weighted risk scoring."""

    confidence_risk: float = 0.20
    ood_score: float = 0.25
    stability_score: float = 0.20
    calibration_risk: float = 0.20
    explanation_score: float = 0.15

    def normalized(self) -> "RiskWeights":
        total = (
            self.confidence_risk
            + self.ood_score
            + self.stability_score
            + self.calibration_risk
            + self.explanation_score
        )
        if total <= 0:
            return RiskWeights()
        return RiskWeights(
            confidence_risk=self.confidence_risk / total,
            ood_score=self.ood_score / total,
            stability_score=self.stability_score / total,
            calibration_risk=self.calibration_risk / total,
            explanation_score=self.explanation_score / total,
        )


class RiskScoringEngine:
    """Combine reliability signals into weighted risk score and risk label."""

    def __init__(
        self,
        weights: RiskWeights | None = None,
        low_threshold: float = 0.33,
        high_threshold: float = 0.66,
    ):
        self.weights = (weights or RiskWeights()).normalized()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    @staticmethod
    def _clip(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, 1.0))

    def weighted_score(
        self,
        confidence_score: Optional[float],
        ood_score: Optional[float],
        stability_score: Optional[float],
        calibration_risk: Optional[float],
        explanation_score: Optional[float],
    ) -> tuple[Optional[float], Dict[str, Any]]:
        """Compute weighted risk score in [0, 1] by renormalizing over available signals."""

        signal_values: Dict[str, Optional[float]] = {
            "confidence_risk": None
            if confidence_score is None
            else (1.0 - self._clip(confidence_score)),
            "ood_score": None if ood_score is None else self._clip(ood_score),
            "stability_score": None if stability_score is None else self._clip(stability_score),
            "calibration_risk": None if calibration_risk is None else self._clip(calibration_risk),
            "explanation_score": None if explanation_score is None else self._clip(explanation_score),
        }

        weight_map = {
            "confidence_risk": self.weights.confidence_risk,
            "ood_score": self.weights.ood_score,
            "stability_score": self.weights.stability_score,
            "calibration_risk": self.weights.calibration_risk,
            "explanation_score": self.weights.explanation_score,
        }

        used_signals = [name for name, value in signal_values.items() if value is not None]
        missing_signals = [name for name, value in signal_values.items() if value is None]
        raw_weight_sum = float(sum(weight_map[name] for name in used_signals))

        if not used_signals or raw_weight_sum <= 0.0:
            return None, {
                "used_signals": used_signals,
                "missing_signals": missing_signals,
                "renormalized_weights": {},
                "components": signal_values,
            }

        renormalized_weights = {
            name: float(weight_map[name] / raw_weight_sum) for name in used_signals
        }
        score = float(
            sum(renormalized_weights[name] * float(signal_values[name]) for name in used_signals)
        )

        return self._clip(score), {
            "used_signals": used_signals,
            "missing_signals": missing_signals,
            "renormalized_weights": renormalized_weights,
            "components": signal_values,
        }

    @staticmethod
    def rule_based_label(
        confidence_score: Optional[float],
        ood_score: Optional[float],
        stability_score: Optional[float],
        calibration_flag: str,
        explanation_score: Optional[float],
    ) -> str:
        """Fallback rule-based risk label."""
        if (
            ood_score is not None
            and confidence_score is not None
            and ood_score > 0.7
            and confidence_score > 0.9
        ):
            return "HIGH"
        if (
            stability_score is not None
            and confidence_score is not None
            and stability_score > 0.7
            and confidence_score > 0.85
        ):
            return "HIGH"
        if calibration_flag == "suspicious" and explanation_score is not None and explanation_score > 0.6:
            return "HIGH"
        if (
            (ood_score is not None and ood_score > 0.5)
            or (stability_score is not None and stability_score > 0.5)
            or (explanation_score is not None and explanation_score > 0.5)
        ):
            return "MEDIUM"
        if (
            ood_score is None
            and stability_score is None
            and explanation_score is None
            and confidence_score is None
        ):
            return "UNKNOWN"
        return "LOW"

    def label_from_score(self, risk_score: Optional[float]) -> str:
        """Map weighted score to label."""
        if risk_score is None:
            return "UNKNOWN"
        if risk_score >= self.high_threshold:
            return "HIGH"
        if risk_score >= self.low_threshold:
            return "MEDIUM"
        return "LOW"

    def combine(
        self,
        confidence_score: Optional[float],
        ood_score: Optional[float],
        stability_score: Optional[float],
        calibration_flag: str,
        calibration_risk: Optional[float],
        explanation_score: Optional[float],
    ) -> Dict[str, Any]:
        """Return weighted score, weighted label, rule label, and final label."""
        weighted, weighted_meta = self.weighted_score(
            confidence_score=confidence_score,
            ood_score=ood_score,
            stability_score=stability_score,
            calibration_risk=calibration_risk,
            explanation_score=explanation_score,
        )
        weighted_label = self.label_from_score(weighted)
        rule_label = self.rule_based_label(
            confidence_score=confidence_score,
            ood_score=ood_score,
            stability_score=stability_score,
            calibration_flag=calibration_flag,
            explanation_score=explanation_score,
        )

        final_label = weighted_label
        if rule_label == "HIGH":
            final_label = "HIGH"

        return {
            "risk_score": weighted,
            "risk_label": final_label,
            "weighted_label": weighted_label,
            "rule_label": rule_label,
            "weighting": weighted_meta,
        }
