"""Fusion of per-prediction reliability signals into an estimated error probability.

The weights are learned, not chosen: a logistic regression is fitted on a
labelled calibration set the monitored model was not trained on, with the
standardised signals as inputs and "the prediction was wrong" as the target.
The fused risk is therefore an estimate of P(error | signals) under the
reference distribution, and its coefficients show how much each signal
contributes.

A logistic model needs enough events to estimate its weights: the usual rule
is at least ten events per input (Peduzzi et al., 1996). With fewer wrong
calibration predictions than ``MIN_EVENTS_PER_SIGNAL`` times the number of
signals, no weights are learned and the fused risk is the confidence risk
1 - c alone (``fallback_signal``), which keeps the ranking of the confidence
and is reported as such.

Risk labels use two thresholds that have a direct meaning:

* LOW    - estimated error probability below the reference error rate
           (safer than an average reference prediction);
* MEDIUM - at or above the reference error rate but below 0.5;
* HIGH   - 0.5 or more (the prediction is more likely wrong than right).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RISK_LABELS = ("LOW", "MEDIUM", "HIGH")
MIN_EVENTS_PER_SIGNAL = 10          # Peduzzi et al. (1996)


class RiskFusion:
    """Logistic stacking of reliability signals (see module docstring)."""

    def __init__(self, signals: Sequence[str], random_state: int = 42, fallback_signal: str = "confidence_risk"):
        self.signals = list(signals)
        self.random_state = random_state
        self.fallback_signal = fallback_signal
        self._model: Pipeline | None = None
        self.reference_error_rate: float | None = None
        self.n_errors: int | None = None
        self.learned: bool = False

    def fit(self, signal_frame: pd.DataFrame, error: np.ndarray) -> "RiskFusion":
        error = np.asarray(error, dtype=int)
        if len(np.unique(error)) < 2:
            raise ValueError("fusion needs both correct and incorrect calibration predictions")
        self.reference_error_rate = float(error.mean())
        self.n_errors = int(error.sum())
        events = min(self.n_errors, len(error) - self.n_errors)
        self.learned = events >= MIN_EVENTS_PER_SIGNAL * len(self.signals)
        self._model = None
        if self.learned:
            self._model = Pipeline([
                ("scale", StandardScaler()),
                ("logit", LogisticRegression(random_state=self.random_state)),
            ]).fit(signal_frame[self.signals].to_numpy(dtype=float), error)
        self._fitted = True
        return self

    @property
    def fitted(self) -> bool:
        return getattr(self, "_fitted", False)

    def predict(self, signal_frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("RiskFusion must be fitted first")
        if self._model is None:
            return signal_frame[self.fallback_signal].to_numpy(dtype=float)
        return self._model.predict_proba(signal_frame[self.signals].to_numpy(dtype=float))[:, 1]

    def label(self, risk: np.ndarray) -> np.ndarray:
        risk = np.asarray(risk, dtype=float)
        out = np.full(risk.shape, "MEDIUM", dtype=object)
        out[risk < self.reference_error_rate] = "LOW"
        out[risk >= 0.5] = "HIGH"
        return out

    def coefficients(self) -> Dict[str, float]:
        """Logit coefficients per standardised signal (log-odds per reference SD)."""
        if self._model is None:
            return {}
        coef = self._model.named_steps["logit"].coef_[0]
        return {name: float(c) for name, c in zip(self.signals, coef)}

    def to_dict(self) -> Dict[str, object]:
        return {
            "learned": self.learned,
            "n_calibration_errors": self.n_errors,
            "min_events": MIN_EVENTS_PER_SIGNAL * len(self.signals),
            "risk_source": "learned fusion" if self.learned else self.fallback_signal,
            "signals": self.signals,
            "coefficients": self.coefficients(),
            "intercept": float(self._model.named_steps["logit"].intercept_[0]) if self._model else None,
            "reference_error_rate": self.reference_error_rate,
            "label_thresholds": {"LOW_below": self.reference_error_rate, "HIGH_from": 0.5},
        }
