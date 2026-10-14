"""Shared experimental setup for the CLEAR-10 image study.

Representation. Every image is mapped to the 512-dimensional global average
pool output of an ImageNet-pretrained ResNet-18 (frozen; torchvision weights
IMAGENET1K_V1, 224 x 224 input). The embeddings are extracted once by
``examples/quickstart/clear10_tabularize_demo.py`` into per-bucket parquet
files, one row per image with the image metadata.

Monitored model. A standardised multinomial logistic regression on the
embeddings, i.e. a linear head on the frozen backbone.

Reference split. Bucket 1 is split, stratified on the class, into train /
calibration / test exactly as for ACS (60 / 20 / 20). The calibration split is
the reference window for every detector and fits the CBPE accuracy map; the
test split gives the reference accuracy and serves as a no-drift control.
Buckets 2 to 10 are production windows compared with that fixed reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from drift_autopsy.core.dataset import Dataset

EMBEDDING_DIM = 512
EMBEDDING_COLUMNS: List[str] = [f"feature_{i}" for i in range(EMBEDDING_DIM)]
EMBEDDING_DIR = Path("outputs/clear10_tabularized_demo")
REFERENCE_BUCKET = 1
ANALYSIS_BUCKETS: List[int] = list(range(2, 11))
SPLIT_FRACTIONS: Tuple[float, float, float] = (0.6, 0.2, 0.2)


def load_bucket(bucket: int, embedding_dir: Path = EMBEDDING_DIR) -> pd.DataFrame:
    """Embeddings plus metadata for one bucket (stored model predictions are dropped)."""
    path = Path(embedding_dir) / f"bucket_{bucket}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run examples/quickstart/clear10_tabularize_demo.py first")
    frame = pd.read_parquet(path)
    drop = [c for c in frame.columns if c.startswith("pred_proba_") or c == "y_pred"]
    return frame.drop(columns=drop).reset_index(drop=True)


def class_names(frame: pd.DataFrame) -> Dict[int, str]:
    pairs = frame[["y_true", "class_name"]].drop_duplicates().sort_values("y_true")
    return {int(k): str(v) for k, v in zip(pairs["y_true"], pairs["class_name"])}


def make_clear_model(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=random_state)),
    ])


@dataclass
class ClearReferenceSplit:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame
    seed: int

    def sizes(self) -> Dict[str, int]:
        return {"train": len(self.train), "calibration": len(self.calibration), "test": len(self.test)}


def split_reference(frame: pd.DataFrame, seed: int = 42,
                    fractions: Tuple[float, float, float] = SPLIT_FRACTIONS) -> ClearReferenceSplit:
    train_frac, cal_frac, test_frac = fractions
    train, rest = train_test_split(frame, train_size=train_frac, random_state=seed, stratify=frame["y_true"])
    cal, test = train_test_split(rest, test_size=test_frac / (cal_frac + test_frac), random_state=seed,
                                 stratify=rest["y_true"])
    return ClearReferenceSplit(train.reset_index(drop=True), cal.reset_index(drop=True),
                               test.reset_index(drop=True), seed)


def embeddings(frame: pd.DataFrame) -> np.ndarray:
    return frame[EMBEDDING_COLUMNS].to_numpy(dtype=float)


def to_dataset(frame: pd.DataFrame, model=None, with_target: bool = True) -> Dataset:
    """Embedding Dataset. Metadata carries the predicted class (label free) for slicing."""
    X = frame[EMBEDDING_COLUMNS].reset_index(drop=True)
    predictions = probabilities = None
    metadata = None
    if model is not None:
        probabilities = model.predict_proba(X.to_numpy(dtype=float))
        predictions = model.classes_[probabilities.argmax(axis=1)]
        metadata = pd.DataFrame({"predicted_class": predictions})
    return Dataset(
        data=X,
        feature_names=list(EMBEDDING_COLUMNS),
        target=frame["y_true"].to_numpy() if with_target and "y_true" in frame else None,
        target_name="y_true" if with_target else None,
        predictions=predictions,
        prediction_probabilities=probabilities,
        metadata=metadata,
        categorical_features=[],
    )
