"""Shared experimental setup for the ACS (Folktables) studies.

One place defines the tasks, the feature typing, the monitored models and the
reference split, so the drift studies, the remediation study and the injection
benchmark all monitor the same models trained the same way.

Tasks.
* ``income``     - ACS Income (Ding et al., 2021): is the personal income above
  $50,000? Adults (age > 16) with income > $100 who worked in the past year.
* ``employment`` - ACS Employment (Ding et al., 2021): is the person employed
  (ESR = 1)? Restricted to ages 16 to 90 as in the task description; the
  Folktables code applies no filter, so children (who have no employment
  status) would otherwise make the task trivial.

Years. Folktables reads the 2014 to 2018 files directly. From 2019 the Census
replaced the relationship variable RELP with RELSHIPP (new codes that split
spouses and partners by sex). :func:`harmonize_relationship` maps RELSHIPP back
to RELP using the code labels of the 2018 and 2019 PUMS data dictionaries.
RELP 11 ("Roomer or boarder") has no successor and so never occurs from 2019.
There is no standard 2020 one-year release (COVID-19 data collection).

Reference split. The reference state-year is split, stratified on the label,
into three disjoint parts:

* ``train``       - fits the monitored model;
* ``calibration`` - never seen in training; it is the reference window every
  detector compares production against, and it fits the CBPE accuracy map;
* ``test``        - never seen in training or calibration; its accuracy is the
  reference accuracy that production accuracy drops are measured against.

Feature typing. ACS codes such as occupation (OCCP) or place of birth (POBP)
are nominal: their numeric order means nothing. They are one-hot encoded for
the logistic model, split natively by the boosted trees and tested with
count-based statistics by the detectors. AGEP and WKHP are numeric; SCHL
(educational attainment, 24 ordered levels) is ordinal and treated as numeric.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.experiments.models import make_tabular_model

# RELSHIPP (2019 on) -> RELP (2014 to 2018), matched on the code labels of the
# 2018 and 2019 PUMS data dictionaries. Same-sex and opposite-sex spouses map to
# "Husband/wife" and both partner codes to "Unmarried partner".
RELSHIPP_TO_RELP: Dict[int, int] = {
    20: 0, 21: 1, 23: 1, 22: 13, 24: 13, 25: 2, 26: 3, 27: 4, 28: 5, 29: 6,
    30: 7, 31: 8, 32: 9, 33: 10, 34: 12, 35: 14, 36: 15, 37: 16, 38: 17,
}

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
    "DIS": "disability",
    "ESP": "employment status of parents",
    "CIT": "citizenship",
    "MIG": "mobility status (lived here a year ago)",
    "MIL": "military service",
    "ANC": "ancestry recode",
    "NATIVITY": "nativity",
    "DEAR": "hearing difficulty",
    "DEYE": "vision difficulty",
    "DREM": "cognitive difficulty",
}


@dataclass(frozen=True)
class AcsTask:
    """One Folktables prediction task."""

    name: str
    title: str
    features: Tuple[str, ...]
    numeric: Tuple[str, ...]
    label_description: str
    extra_columns: Tuple[str, ...] = ()   # loaded alongside, never model inputs

    @property
    def categorical(self) -> List[str]:
        return [f for f in self.features if f not in self.numeric]

    def select(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Rows of the task population with features, label and extra columns."""
        if self.name == "income":
            df = raw[(raw["AGEP"] > 16) & (raw["PINCP"] > 100) & (raw["WKHP"] > 0) & (raw["PWGTP"] >= 1)]
            target = df["PINCP"] > 50000
        elif self.name == "employment":
            df = raw[(raw["AGEP"] >= 16) & (raw["AGEP"] <= 90)]
            target = df["ESR"] == 1
        else:
            raise ValueError(self.name)
        out = pd.DataFrame(np.nan_to_num(df[list(self.features)].to_numpy(dtype=float), nan=-1.0),
                           columns=list(self.features))
        out["target"] = target.to_numpy()
        for c in self.extra_columns:
            out[c] = df[c].to_numpy(dtype=float)
        return out


ACS_TASKS: Dict[str, AcsTask] = {
    "income": AcsTask(
        name="income", title="ACS Income",
        features=("AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"),
        numeric=("AGEP", "SCHL", "WKHP"),
        label_description="personal income above $50,000",
        extra_columns=("PINCP",),
    ),
    "employment": AcsTask(
        name="employment", title="ACS Employment",
        features=("AGEP", "SCHL", "MAR", "RELP", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC",
                  "NATIVITY", "DEAR", "DEYE", "DREM", "SEX", "RAC1P"),
        numeric=("AGEP", "SCHL"),
        label_description="employed (ESR = 1)",
    ),
}

# ACS Income names kept for the existing studies.
ACS_FEATURES: List[str] = list(ACS_TASKS["income"].features)
ACS_NUMERIC: List[str] = list(ACS_TASKS["income"].numeric)
ACS_CATEGORICAL: List[str] = ACS_TASKS["income"].categorical

SPLIT_FRACTIONS: Tuple[float, float, float] = (0.6, 0.2, 0.2)
CACHE_DIR = Path("folktables_us_census") / "tasks"


def harmonize_relationship(raw: pd.DataFrame) -> pd.DataFrame:
    """Add RELP from RELSHIPP for 2019 and later files (see module docstring)."""
    if "RELP" in raw.columns or "RELSHIPP" not in raw.columns:
        return raw
    codes = raw["RELSHIPP"]
    unknown = sorted(set(codes.dropna().astype(int)) - set(RELSHIPP_TO_RELP))
    if unknown:
        raise ValueError(f"RELSHIPP codes without a RELP equivalent: {unknown}")
    raw = raw.copy()
    raw["RELP"] = codes.map(lambda c: RELSHIPP_TO_RELP.get(int(c)) if pd.notna(c) else np.nan)
    return raw


def load_acs_task_frame(task: str, year: int, state: str, data_root: str = "data",
                        download: bool = True) -> pd.DataFrame:
    """Task frame for one state-year (features, ``target`` and extra columns), cached as parquet."""
    spec = ACS_TASKS[task]
    cache = Path(data_root) / CACHE_DIR / f"{task}_{state}_{year}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from folktables import ACSDataSource

    raw = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person",
                        root_dir=data_root).get_data(states=[state], download=download)
    frame = spec.select(harmonize_relationship(raw))
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def load_acs_task(task: str, year: int, state: str, data_root: str = "data",
                  with_extra: bool = False):
    """Feature frame (task feature order, float codes) and binary label for one state-year.
    With ``with_extra`` the extra columns (for example PINCP) are returned third."""
    spec = ACS_TASKS[task]
    frame = load_acs_task_frame(task, year, state, data_root=data_root)
    X = frame[list(spec.features)].astype(float).reset_index(drop=True)
    y = frame["target"].to_numpy().astype(int)
    if with_extra:
        return X, y, frame[list(spec.extra_columns)].reset_index(drop=True)
    return X, y


def load_acs(year: int, state: str, data_root: str = "data") -> Tuple[pd.DataFrame, np.ndarray]:
    """ACS Income features and label for one state-year."""
    return load_acs_task("income", year, state, data_root=data_root)


def make_acs_model(random_state: int = 42, keep_columns: Optional[Sequence[int]] = None,
                   kind: str = "lr", task: str = "income") -> Pipeline:
    """The monitored model for an ACS task (see :mod:`drift_autopsy.experiments.models`).
    Input is the raw feature matrix in task feature order, or, with
    ``keep_columns``, the matrix restricted to those column positions (used by
    feature dropping)."""
    spec = ACS_TASKS[task]
    names = list(spec.features) if keep_columns is None else [spec.features[i] for i in keep_columns]
    numeric_idx = [names.index(f) for f in spec.numeric if f in names]
    categorical_idx = [names.index(f) for f in spec.categorical if f in names]
    return make_tabular_model(kind, numeric_idx, categorical_idx, random_state=random_state)


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
    task: str = "income",
) -> Dataset:
    """Wrap an ACS frame as a Dataset with the nominal features declared.

    When ``model`` is given, its predictions and class probabilities are attached
    (needed by the CBPE detector and the reliability layer).
    """
    spec = ACS_TASKS[task]
    X = X[list(spec.features)].reset_index(drop=True)
    predictions = probabilities = None
    if model is not None:
        values = X.to_numpy(dtype=float)
        probabilities = model.predict_proba(values)
        predictions = model.classes_[probabilities.argmax(axis=1)]
    return Dataset(
        data=X,
        feature_names=list(spec.features),
        target=None if y is None else np.asarray(y),
        target_name=None if y is None else "target",
        predictions=predictions,
        prediction_probabilities=probabilities,
        metadata=metadata,
        categorical_features=spec.categorical,
    )
