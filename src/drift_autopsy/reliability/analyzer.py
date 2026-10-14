"""Per-prediction reliability for classifiers on feature vectors.

For every production prediction the analyzer computes three label-free
signals, each in [0, 1]:

* ``confidence_risk``  - 1 - c, where c is the top-class confidence.
* ``ood``              - out-of-distribution score against the model's
  training data (see :mod:`drift_autopsy.reliability.ood`).
* ``instability``      - mean change in the predicted probabilities under
  small input noise (see :mod:`drift_autopsy.reliability.stability`).

It also reports ``calibrated_error`` = 1 - a(c), the error probability read off
the histogram-binning map from confidence to accuracy fitted on the labelled
calibration set (see :mod:`drift_autopsy.estimation`). That map is the CBPE
building block; it is a step function of c, so it is reported but not fused
(the fusion below recalibrates the continuous confidence itself).

With calibration data the three signals are fused by a logistic regression
fitted on the calibration set (see :mod:`drift_autopsy.reliability.fusion`),
giving an estimated error probability and a LOW / MEDIUM / HIGH label. Without
calibration data the risk score is the unweighted mean of the signals and the
label is ``UNCALIBRATED``.

Population-level evidence (the CBPE accuracy estimate, the confidence shift,
the explanation shift) describes a window, not a row, and is reported per
window by the pipeline rather than folded into each prediction's score.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from drift_autopsy.estimation import BinnedAccuracyCalibration, top_class_confidence
from drift_autopsy.reliability.fusion import RISK_LABELS, RiskFusion
from drift_autopsy.reliability.ood import OODDetector
from drift_autopsy.reliability.stability import StabilityChecker

SIGNALS = ("confidence_risk", "ood", "instability")


class ReliabilityAnalyzer:
    """
    Args:
        model: fitted classifier with ``predict_proba`` that takes the raw feature matrix
        reference_X: the model's training features (fits the OOD and noise scales)
        calibration_X, calibration_y: labelled data the model was not trained on
            (fits the confidence-to-accuracy map and the fusion)
        categorical_indices: columns holding nominal codes (scored by rarity, never perturbed)
        n_bins: confidence bins of the accuracy map
        perturbation_strength: noise SD as a fraction of each feature's reference SD
        n_perturbations: noisy copies per row
        random_state: seed for every random draw
    """

    def __init__(
        self,
        model: Any,
        reference_X: np.ndarray,
        calibration_X: Optional[np.ndarray] = None,
        calibration_y: Optional[np.ndarray] = None,
        categorical_indices: Optional[Sequence[int]] = None,
        n_bins: int = 10,
        perturbation_strength: float = 0.1,
        n_perturbations: int = 5,
        random_state: int = 42,
    ):
        self.model = model
        reference_X = np.asarray(reference_X, dtype=float)
        categorical = list(categorical_indices or [])
        numeric = [j for j in range(reference_X.shape[1]) if j not in set(categorical)]

        self.ood = OODDetector(categorical_indices=categorical, random_state=random_state).fit(reference_X)
        self.stability = StabilityChecker(
            model, perturb_indices=numeric, strength=perturbation_strength,
            n_perturbations=n_perturbations, random_state=random_state,
        ).fit(reference_X)

        self.calibration: Optional[BinnedAccuracyCalibration] = None
        self.fusion = RiskFusion(SIGNALS, random_state=random_state)
        if calibration_X is not None and calibration_y is not None:
            calibration_X = np.asarray(calibration_X, dtype=float)
            proba = np.asarray(model.predict_proba(calibration_X), dtype=float)
            correct = (model.classes_[proba.argmax(axis=1)] == np.asarray(calibration_y)).astype(float)
            self.calibration = BinnedAccuracyCalibration.fit(top_class_confidence(proba), correct, n_bins)
            self.fusion.fit(self.signals(calibration_X), 1 - correct.astype(int))

    def signals(self, X: np.ndarray) -> pd.DataFrame:
        """Per-row prediction, confidence and the three reliability signals."""
        X = np.asarray(X, dtype=float)
        proba = np.asarray(self.model.predict_proba(X), dtype=float)
        confidence = top_class_confidence(proba)
        if self.calibration is not None:
            calibrated_error = 1.0 - self.calibration.predict_correct(confidence)
        else:
            calibrated_error = 1.0 - confidence
        return pd.DataFrame({
            "prediction": self.model.classes_[proba.argmax(axis=1)],
            "confidence": confidence,
            "confidence_risk": 1.0 - confidence,
            "calibrated_error": calibrated_error,
            "ood": self.ood.score(X),
            "instability": self.stability.score(X),
        })

    def analyze(self, X: np.ndarray) -> pd.DataFrame:
        """Signals plus ``risk_score`` and ``risk_label`` per row."""
        frame = self.signals(X)
        if self.fusion.fitted:
            frame["risk_score"] = self.fusion.predict(frame)
            frame["risk_label"] = self.fusion.label(frame["risk_score"].to_numpy())
        else:
            frame["risk_score"] = frame[list(SIGNALS)].mean(axis=1)
            frame["risk_label"] = "UNCALIBRATED"
        return frame

    def describe(self) -> Dict[str, Any]:
        return {
            "signals": list(SIGNALS),
            "fusion": self.fusion.to_dict() if self.fusion.fitted else None,
            "calibration_map": self.calibration.to_dict() if self.calibration is not None else None,
            "ood": {"zero_percentile": self.ood.zero_percentile, "one_percentile": self.ood.one_percentile},
            "instability": {"strength": self.stability.strength, "n_perturbations": self.stability.n_perturbations},
        }

    @staticmethod
    def summarize(frame: pd.DataFrame, y_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Window summary; with ``y_true`` it also scores the risk as an error detector.

        Labels are used here only to evaluate the layer, never to compute it.
        """
        out: Dict[str, Any] = {"n": int(len(frame))}
        for col in ("confidence", "calibrated_error") + SIGNALS + ("risk_score",):
            out[f"mean_{col}"] = float(frame[col].mean())
        labels = frame["risk_label"].astype(str)
        for lab in RISK_LABELS:
            out[f"share_{lab.lower()}"] = float((labels == lab).mean())
        if y_true is None:
            return out
        error = (frame["prediction"].to_numpy() != np.asarray(y_true)).astype(int)
        out["accuracy"] = float(1 - error.mean())
        if len(np.unique(error)) == 2:
            out["auroc_risk"] = float(roc_auc_score(error, frame["risk_score"]))
            out["auroc_confidence_only"] = float(roc_auc_score(error, 1 - frame["confidence"]))
            for col in SIGNALS + ("calibrated_error",):
                out[f"auroc_{col}"] = float(roc_auc_score(error, frame[col]))
            out["brier_risk"] = float(brier_score_loss(error, np.clip(frame["risk_score"], 0, 1)))
        for lab in RISK_LABELS:
            mask = (labels == lab).to_numpy()
            out[f"error_rate_{lab.lower()}"] = float(error[mask].mean()) if mask.any() else None
        return out
