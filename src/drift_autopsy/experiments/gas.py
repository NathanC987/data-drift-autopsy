"""Shared experimental setup for the gas sensor drift study.

Data. The UCI Gas Sensor Array Drift dataset (Vergara et al., 2012;
Rodriguez-Lujan et al., 2014; UCI id 224, CC BY 4.0): 13,910 measurements of
six gases (ethanol, ethylene, ammonia, acetaldehyde, acetone, toluene) by an
array of 16 metal-oxide sensors, recorded over 36 months in ten batches. Each
measurement has 128 features, eight per sensor (the steady-state resistance
change, its normalised version and six exponential moving averages of the
rising and decaying transients). The sensors age and are poisoned over time,
so the response to the same gas drifts: a documented, physical change in
P(X | Y) while the gas labels stay exact.

Reference split. Batches 1 and 2 (months 1 to 10, with gaps) form the reference; batch 1
alone (445 rows) is too small for a three-way split. They are split, stratified
on the gas, into train / calibration / test (60 / 20 / 20) exactly as for ACS.
Batches 3 to 10 are production windows compared with that fixed reference.

Audit size. Some batches are small (batch 4 has 161 rows), so a window's audit
takes min(600, floor(n / 2)) labels.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.experiments.models import make_dense_model

URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
SHA256 = "91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4"
DATA_DIR = Path("data/gas_sensor_drift")

N_SENSORS, FEATURES_PER_SENSOR = 16, 8
FEATURES: List[str] = [f"S{s:02d}_F{f}" for s in range(1, N_SENSORS + 1) for f in range(1, FEATURES_PER_SENSOR + 1)]
GASES: Dict[int, str] = {1: "ethanol", 2: "ethylene", 3: "ammonia", 4: "acetaldehyde", 5: "acetone", 6: "toluene"}
BATCH_MONTHS: Dict[int, str] = {1: "1-2", 2: "3-4, 8-10", 3: "11-13", 4: "14-15", 5: "16", 6: "17-20",
                                7: "21", 8: "22-23", 9: "24 and 30", 10: "36"}
REFERENCE_BATCHES: Tuple[int, ...] = (1, 2)
ANALYSIS_BATCHES: List[int] = list(range(3, 11))
SPLIT_FRACTIONS: Tuple[float, float, float] = (0.6, 0.2, 0.2)
AUDIT_CAP = 600


def sensor_of(feature: str) -> int:
    return int(feature[1:3])


def audit_size(n_rows: int, cap: int = AUDIT_CAP) -> int:
    return int(min(cap, n_rows // 2))


def _archive(data_dir: Path, download: bool) -> Path:
    path = Path(data_dir) / "gas_sensor_array_drift.zip"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} missing; call with download=True")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=300) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"{path} has SHA-256 {digest}, expected {SHA256}")
    return path


def _parse(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        values = np.full(len(FEATURES), np.nan)
        for item in parts[1:]:
            k, v = item.split(":")
            values[int(k) - 1] = float(v)
        rows.append((int(parts[0].split(";")[0]), values))
    frame = pd.DataFrame(np.vstack([v for _, v in rows]), columns=FEATURES)
    frame.insert(0, "gas", [g for g, _ in rows])
    return frame


def load_batches(data_dir: Path = DATA_DIR, download: bool = True) -> pd.DataFrame:
    """All ten batches with columns ``batch``, ``gas`` and the 128 features."""
    cache = Path(data_dir) / "batches.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frames = []
    with zipfile.ZipFile(_archive(data_dir, download)) as z:
        for b in range(1, 11):
            frame = _parse(io.TextIOWrapper(z.open(f"Dataset/batch{b}.dat"), encoding="ascii").read())
            frame.insert(0, "batch", b)
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    if out[FEATURES].isna().any().any():
        raise ValueError("missing feature values in the gas sensor files")
    out.to_parquet(cache, index=False)
    return out


def make_gas_model(random_state: int = 42, kind: str = "lr", keep_columns: Optional[List[int]] = None):
    """Multinomial logistic regression on standardised features, or boosted trees."""
    return make_dense_model(kind, random_state=random_state)


@dataclass
class GasReferenceSplit:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    seed: int

    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train), "calibration": len(self.calibration), "test": len(self.test)}


def split_reference(frame: pd.DataFrame, seed: int = 42,
                    fractions: Tuple[float, float, float] = SPLIT_FRACTIONS) -> GasReferenceSplit:
    train_frac, cal_frac, test_frac = fractions
    train, rest = train_test_split(frame, train_size=train_frac, random_state=seed, stratify=frame["gas"])
    cal, test = train_test_split(rest, test_size=test_frac / (cal_frac + test_frac), random_state=seed,
                                 stratify=rest["gas"])
    return GasReferenceSplit(train.reset_index(drop=True), cal.reset_index(drop=True),
                             test.reset_index(drop=True), seed)


def features(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURES].to_numpy(dtype=float)


def to_dataset(frame: pd.DataFrame, model=None, with_target: bool = True) -> Dataset:
    X = frame[FEATURES].reset_index(drop=True)
    predictions = probabilities = None
    if model is not None:
        probabilities = model.predict_proba(X.to_numpy(dtype=float))
        predictions = model.classes_[probabilities.argmax(axis=1)]
    return Dataset(
        data=X,
        feature_names=list(FEATURES),
        target=frame["gas"].to_numpy() if with_target else None,
        target_name="gas" if with_target else None,
        predictions=predictions,
        prediction_probabilities=probabilities,
        categorical_features=[],
    )
