"""Shared experimental setup for the ACS Income (Folktables) studies.

One place defines the feature typing, the monitored model and the reference
split, so the drift study, the remediation study and the injection benchmark
all monitor the same model trained the same way.

Reference split. The 2014 California data are split, stratified on the label,
into three disjoint parts:

* ``train``       - fits the monitored model;
* ``calibration`` - never seen in training; it is the reference window every
  detector compares production against, and it fits the CBPE accuracy map;
* ``test``        - never seen in training or calibration; its accuracy is the
  reference accuracy that production accuracy drops are measured against.

Feature typing. ACS codes such as occupation (OCCP) or place of birth (POBP)
are nominal: their numeric order means nothing. They are one-hot encoded for
the model and tested with count-based statistics by the detectors. AGEP and
WKHP are numeric; SCHL (educational attainment, 24 ordered levels) is ordinal
and treated as numeric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.data.loaders import FolktablesLoader

ACS_FEATURES: List[str] = list(FolktablesLoader.ACS_INCOME_FEATURES)
ACS_NUMERIC: List[str] = ["AGEP", "SCHL", "WKHP"]
ACS_CATEGORICAL: List[str] = [f for f in ACS_FEATURES if f not in ACS_NUMERIC]

ACS_FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "AGEP": "age",
    "COW": "class of worker",
    "SCHL": "educational attainment",
    "MAR": "marital status",
    "OCCP": "occupation",
    "POBP": "place of birth",
    "RELP": "relationship",
    "WKHP": "usual hours worked per week",
    "SEX": "sex",
    "RAC1P": "race",
}

SPLIT_FRACTIONS: Tuple[float, float, float] = (0.6, 0.2, 0.2)


def make_acs_model(random_state: int = 42, keep_columns: Optional[Sequence[int]] = None) -> Pipeline:
    """The monitored model: one-hot nominal codes, standardised numeric features,
    L2-regularised logistic regression (C = 1). Input is the raw 10-column
    feature matrix in ``ACS_FEATURES`` order, or, with ``keep_columns``, the
    matrix restricted to those column positions (used by feature dropping)."""
    names = ACS_FEATURES if keep_columns is None else [ACS_FEATURES[i] for i in keep_columns]
    numeric_idx = [names.index(f) for f in ACS_NUMERIC if f in names]
    categorical_idx = [names.index(f) for f in ACS_CATEGORICAL if f in names]
    preprocess = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_idx),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_idx),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("classifier", LogisticRegression(max_iter=5000, random_state=random_state)),
        ]
    )


def load_acs(year: int, state: str, data_root: str = "data") -> Tuple[pd.DataFrame, np.ndarray]:
    """Feature frame (ACS_FEATURES order, float codes) and binary label for one state-year."""
    frame = FolktablesLoader.load_acs_income_cached(year, state, data_root=data_root)
    X = frame[ACS_FEATURES].astype(float).reset_index(drop=True)
    y = frame["target"].to_numpy().astype(int)
    return X, y


@dataclass
class ACSReferenceSplit:
    train_X: pd.DataFrame
    train_y: np.ndarray
    calibration_X: pd.DataFrame
    calibration_y: np.ndarray
    test_X: pd.DataFrame
    test_y: np.ndarray
    seed: int

    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train_X), "calibration": len(self.calibration_X), "test": len(self.test_X)}


def split_reference(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 42,
    fractions: Tuple[float, float, float] = SPLIT_FRACTIONS,
) -> ACSReferenceSplit:
    """Stratified train / calibration / test split (see module docstring)."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to 1")
    train_frac, cal_frac, test_frac = fractions
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=train_frac, random_state=seed, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, test_size=test_frac / (cal_frac + test_frac), random_state=seed, stratify=y_rest
    )
    return ACSReferenceSplit(
        train_X=X_train.reset_index(drop=True), train_y=y_train,
        calibration_X=X_cal.reset_index(drop=True), calibration_y=y_cal,
        test_X=X_test.reset_index(drop=True), test_y=y_test,
        seed=seed,
    )


def to_dataset(
    X: pd.DataFrame,
    y: Optional[np.ndarray] = None,
    model=None,
    metadata: Optional[pd.DataFrame] = None,
) -> Dataset:
    """Wrap an ACS frame as a Dataset with the nominal features declared.

    When ``model`` is given, its predictions and class probabilities are attached
    (needed by the CBPE detector and the reliability layer).
    """
    X = X[ACS_FEATURES].reset_index(drop=True)
    predictions = probabilities = None
    if model is not None:
        values = X.to_numpy(dtype=float)
        probabilities = model.predict_proba(values)
        predictions = model.classes_[probabilities.argmax(axis=1)]
    return Dataset(
        data=X,
        feature_names=list(ACS_FEATURES),
        target=None if y is None else np.asarray(y),
        target_name=None if y is None else "target",
        predictions=predictions,
        prediction_probabilities=probabilities,
        metadata=metadata,
        categorical_features=list(ACS_CATEGORICAL),
    )
