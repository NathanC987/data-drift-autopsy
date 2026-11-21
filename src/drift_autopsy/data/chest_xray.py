"""Chest X-ray dataset loader and splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
NORMAL_DIR_NAMES = {"normal", "no_finding", "healthy"}
DISEASE_DIR_NAMES = {"disease", "abnormal", "pneumonia", "opacity", "positive", "finding"}


@dataclass
class ChestXraySubset:
    """Container for a loaded Chest X-ray image subset."""

    images: torch.Tensor
    labels: np.ndarray
    paths: List[Path]
    metadata: pd.DataFrame


def _resolve_label_from_path(path: Path) -> int:
    parent = path.parent.name.strip().lower()
    if parent in NORMAL_DIR_NAMES:
        return 0
    if parent in DISEASE_DIR_NAMES:
        return 1
    return -1


def build_subset(
    dataset_dir: str | Path,
    metadata_csv: Optional[str | Path] = None,
    max_samples: int = 200,
    image_size: int = 64,
    seed: int = 42,
) -> ChestXraySubset:
    """Load a deterministic Chest X-ray subset resized to grayscale tensors in [0, 1]."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    all_paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not all_paths:
        raise ValueError(f"No images found under {root}")

    rng = np.random.default_rng(seed)
    if len(all_paths) > max_samples:
        chosen_idx = np.sort(rng.choice(len(all_paths), size=max_samples, replace=False))
        sampled_paths = [all_paths[int(i)] for i in chosen_idx]
    else:
        sampled_paths = all_paths

    csv_df: Optional[pd.DataFrame] = None
    csv_lookup: dict[str, dict] = {}
    if metadata_csv is not None:
        csv_path = Path(metadata_csv)
        if csv_path.exists():
            csv_df = pd.read_csv(csv_path)
            if "Image Index" in csv_df.columns:
                deduped = csv_df.drop_duplicates(subset=["Image Index"])
                csv_lookup = deduped.set_index("Image Index").to_dict(orient="index")

    tensors: List[torch.Tensor] = []
    labels: List[int] = []
    meta_rows: List[dict] = []

    for path in sampled_paths:
        with Image.open(path) as img:
            gray = img.convert("L")
            resized = TF.resize(gray, [image_size, image_size])
            tensor = TF.to_tensor(resized)  # (1, H, W) in [0.0, 1.0]
        tensors.append(tensor)

        row_info = csv_lookup.get(path.name)
        if row_info is not None and "Finding Labels" in row_info:
            finding = str(row_info["Finding Labels"]).strip()
            label = 0 if finding == "No Finding" else 1
            meta_row = {"Image Index": path.name, "image_path": str(path), **row_info}
        else:
            label = _resolve_label_from_path(path)
            meta_row = {"Image Index": path.name, "image_path": str(path)}
        labels.append(label)
        meta_rows.append(meta_row)

    images_tensor = torch.stack(tensors, dim=0)
    labels_arr = np.asarray(labels, dtype=np.int64)
    metadata_df = pd.DataFrame(meta_rows)

    return ChestXraySubset(
        images=images_tensor,
        labels=labels_arr,
        paths=sampled_paths,
        metadata=metadata_df,
    )


def split_indices(
    labels: np.ndarray,
    metadata_df: Optional[pd.DataFrame] = None,
    test_ratio: float = 0.35,
    seed: int = 42,
    use_metadata_split: bool = False,
    split_column: str = "View Position",
    baseline_value: str = "PA",
) -> Tuple[np.ndarray, np.ndarray]:
    """Split sample indices into baseline and drifted sets."""
    n = len(labels)
    indices = np.arange(n, dtype=np.int64)

    if use_metadata_split and metadata_df is not None and split_column in metadata_df.columns:
        col_vals = metadata_df[split_column].astype(str).str.strip()
        mask = (col_vals == str(baseline_value).strip()).to_numpy()
        base_idx = indices[mask]
        drift_idx = indices[~mask]
        if len(base_idx) >= 20 and len(drift_idx) >= 20:
            return base_idx, drift_idx

    valid_mask = labels >= 0
    stratify_labels: Optional[np.ndarray] = None
    if valid_mask.sum() == n and len(np.unique(labels)) > 1:
        _, counts = np.unique(labels, return_counts=True)
        if np.all(counts >= 2):
            stratify_labels = labels
    elif valid_mask.sum() > 0 and len(np.unique(labels[valid_mask])) > 1:
        _, counts = np.unique(labels, return_counts=True)
        if np.all(counts >= 2):
            stratify_labels = labels

    base_idx, drift_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )
    return np.sort(base_idx), np.sort(drift_idx)
