"""Indirect calibration checks without labels using confidence and CBPE."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class CalibrationChecker:
    """
    Evaluate confidence calibration risk without labels.

    Uses confidence level, confidence shift, and external CBPE score.
    """

    def __init__(
        self,
        overconfidence_gap: float = 0.25,
        suspicious_threshold: float = 0.6,
    ):
        self.overconfidence_gap = overconfidence_gap
        self.suspicious_threshold = suspicious_threshold

    @staticmethod
    def confidence_shift(reference_confidences: np.ndarray, current_confidences: np.ndarray) -> float:
        """Compute normalized confidence distribution shift in [0, 1]."""
        ref = np.asarray(reference_confidences, dtype=float).reshape(-1)
        cur = np.asarray(current_confidences, dtype=float).reshape(-1)
        if len(ref) == 0 or len(cur) == 0:
            return 0.0

        q_ref = np.percentile(ref, [10, 25, 50, 75, 90])
        q_cur = np.percentile(cur, [10, 25, 50, 75, 90])
        shift = float(np.mean(np.abs(q_ref - q_cur)))
        return float(np.clip(shift, 0.0, 1.0))

    def evaluate(
        self,
        confidence_score: float,
        cbpe_score: Optional[float],
        confidence_distribution_shift: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Evaluate calibration quality with external CBPE signal.

        Returns:
            Dictionary with calibration_flag and calibration_risk in [0, 1].
        """
        conf = float(np.clip(confidence_score, 0.0, 1.0))
        cbpe = float(np.clip(cbpe_score, 0.0, 1.0)) if cbpe_score is not None else 0.5
        shift = float(np.clip(confidence_distribution_shift, 0.0, 1.0))

        overconfidence = max(0.0, conf - cbpe)
        gap_risk = float(np.clip(overconfidence / max(self.overconfidence_gap, 1e-8), 0.0, 1.0))

        calibration_risk = float(np.clip(0.7 * gap_risk + 0.3 * shift, 0.0, 1.0))
        calibration_flag = "suspicious" if calibration_risk >= self.suspicious_threshold else "good"

        return {
            "calibration_flag": calibration_flag,
            "calibration_risk": calibration_risk,
            "metadata": {
                "confidence": conf,
                "cbpe_score": cbpe,
                "confidence_distribution_shift": shift,
                "overconfidence_gap": overconfidence,
            },
        }
