"""Risk scoring engine for reliability and hallucination detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
        confidence_score: float,
        ood_score: float,
        stability_score: float,
        calibration_risk: float,
        explanation_score: float,
    ) -> float:
        """Compute weighted risk score in [0, 1]."""
        confidence_risk = 1.0 - self._clip(confidence_score)
        score = (
            self.weights.confidence_risk * confidence_risk
            + self.weights.ood_score * self._clip(ood_score)
            + self.weights.stability_score * self._clip(stability_score)
            + self.weights.calibration_risk * self._clip(calibration_risk)
            + self.weights.explanation_score * self._clip(explanation_score)
        )
        return self._clip(score)

    @staticmethod
    def rule_based_label(
        confidence_score: float,
        ood_score: float,
        stability_score: float,
        calibration_flag: str,
        explanation_score: float,
    ) -> str:
        """Fallback rule-based risk label."""
        if ood_score > 0.7 and confidence_score > 0.9:
            return "HIGH"
        if stability_score > 0.7 and confidence_score > 0.85:
            return "HIGH"
        if calibration_flag == "suspicious" and explanation_score > 0.6:
            return "HIGH"
        if ood_score > 0.5 or stability_score > 0.5 or explanation_score > 0.5:
            return "MEDIUM"
        return "LOW"

    def label_from_score(self, risk_score: float) -> str:
        """Map weighted score to label."""
        if risk_score >= self.high_threshold:
            return "HIGH"
        if risk_score >= self.low_threshold:
            return "MEDIUM"
        return "LOW"

    def combine(
        self,
        confidence_score: float,
        ood_score: float,
        stability_score: float,
        calibration_flag: str,
        calibration_risk: float,
        explanation_score: float,
    ) -> Dict[str, Any]:
        """Return weighted score, weighted label, rule label, and final label."""
        weighted = self.weighted_score(
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
        }
