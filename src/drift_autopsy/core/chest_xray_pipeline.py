"""End-to-end pipeline orchestration for Chest X-ray drift detection and RCA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import DetectionResult, LocalizationResult
from drift_autopsy.data.chest_xray import ChestXraySubset, build_subset, split_indices
from drift_autopsy.detectors.distribution.mmd import MMD
from drift_autopsy.detectors.model_based.chest_xray_cnn import (
    ChestXrayClassifier,
    train_classifier,
)
from drift_autopsy.detectors.proxy.cbpe import CBPE
from drift_autopsy.detectors.reconstruction.autoencoder import (
    ChestXrayAutoencoder,
    extract_embeddings,
    reconstruction_errors,
    simulate_drift,
    train_autoencoder,
)
from drift_autopsy.detectors.statistical.ks_test import KSTest
from drift_autopsy.detectors.statistical.psi import PSI
from drift_autopsy.localizers.univariate import UnivariateLocalizer
from drift_autopsy.reliability.ood import OODDetector


@dataclass
class XrayAnalysisResult:
    """Container for Chest X-ray drift analysis outputs."""

    subset: ChestXraySubset
    baseline_indices: np.ndarray
    drifted_indices: np.ndarray
    baseline_images: torch.Tensor
    drifted_images: torch.Tensor
    baseline_errors: np.ndarray
    drifted_errors: np.ndarray
    baseline_embeddings: np.ndarray
    drifted_embeddings: np.ndarray
    mmd_result: DetectionResult
    ks_result: DetectionResult
    psi_result: DetectionResult
    localization_result: LocalizationResult
    embedding_df: pd.DataFrame
    pca_explained_variance: np.ndarray
    classifier: Optional[ChestXrayClassifier]
    classifier_val_acc: Optional[float]
    cbpe_result: Optional[DetectionResult]
    drifted_actual_acc: Optional[float]
    baseline_confidences: Optional[np.ndarray]
    drifted_confidences: Optional[np.ndarray]
    baseline_ood_scores: np.ndarray
    drifted_ood_scores: np.ndarray
    drifted_ood_unclipped: np.ndarray
    device: str


def run_xray_analysis(
    dataset_dir: str | Path,
    metadata_csv: Optional[str | Path] = None,
    max_samples: int = 200,
    image_size: int = 64,
    baseline_ratio: float = 0.65,
    batch_size: int = 32,
    seed: int = 42,
    brightness: float = 1.25,
    contrast: float = 1.15,
    noise_std: float = 0.04,
    ae_epochs: int = 8,
    cls_epochs: int = 8,
    mmd_threshold: float = 0.05,
    ks_threshold: float = 0.05,
    psi_threshold: float = 0.2,
    use_metadata_split: bool = False,
    split_column: str = "View Position",
    baseline_value: str = "PA",
    device: Optional[str] = None,
) -> XrayAnalysisResult:
    """Run the full Chest X-ray drift detection and RCA preparation pipeline."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    subset = build_subset(
        dataset_dir=dataset_dir,
        metadata_csv=metadata_csv,
        max_samples=max_samples,
        image_size=image_size,
        seed=seed,
    )
    if len(subset.images) < 40:
        raise ValueError(
            f"At least 40 images are required for Chest X-ray drift analysis, found {len(subset.images)}"
        )

    test_ratio = float(np.clip(1.0 - baseline_ratio, 0.1, 0.9))
    base_idx, drift_idx = split_indices(
        labels=subset.labels,
        metadata_df=subset.metadata,
        test_ratio=test_ratio,
        seed=seed,
        use_metadata_split=use_metadata_split,
        split_column=split_column,
        baseline_value=baseline_value,
    )

    baseline_images = subset.images[base_idx]
    drifted_raw = subset.images[drift_idx]
    drifted_images = simulate_drift(
        drifted_raw,
        brightness=brightness,
        contrast=contrast,
        noise_std=noise_std,
        seed=seed,
    )

    ae = ChestXrayAutoencoder()
    ae_loader = DataLoader(TensorDataset(baseline_images), batch_size=batch_size, shuffle=True)
    train_autoencoder(ae, ae_loader, epochs=ae_epochs, device=resolved_device)

    baseline_errors = reconstruction_errors(ae, baseline_images, batch_size=batch_size, device=resolved_device)
    drifted_errors = reconstruction_errors(ae, drifted_images, batch_size=batch_size, device=resolved_device)
    baseline_embeddings = extract_embeddings(ae, baseline_images, batch_size=batch_size, device=resolved_device)
    drifted_embeddings = extract_embeddings(ae, drifted_images, batch_size=batch_size, device=resolved_device)

    feature_names = [f"z{i}" for i in range(baseline_embeddings.shape[1])]
    ref_ds = Dataset.from_numpy(baseline_embeddings, feature_names=feature_names)
    test_ds = Dataset.from_numpy(drifted_embeddings, feature_names=feature_names)

    mmd_detector = MMD(threshold=mmd_threshold, random_state=seed)
    mmd_detector.fit(ref_ds)
    mmd_result = mmd_detector.detect(test_ds)

    ks_detector = KSTest(threshold=ks_threshold)
    ks_detector.fit(ref_ds)
    ks_result = ks_detector.detect(test_ds)

    psi_detector = PSI(threshold=psi_threshold)
    psi_detector.fit(ref_ds)
    psi_result = psi_detector.detect(test_ds)

    localizer = UnivariateLocalizer(threshold=ks_threshold, correction="holm")
    localization_result = localizer.localize(ref_ds, test_ds)

    ood_detector = OODDetector(random_state=seed)
    ood_detector.fit(baseline_embeddings)
    baseline_ood_scores = ood_detector.score(baseline_embeddings)
    drifted_ood_comp = ood_detector.score_components(drifted_embeddings)
    drifted_ood_scores = drifted_ood_comp["ood"]
    drifted_ood_unclipped = drifted_ood_comp["ood_unclipped"]

    stacked = np.vstack([baseline_embeddings, drifted_embeddings])
    n_components = min(3, stacked.shape[0], stacked.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    pcs = pca.fit_transform(stacked)
    if pcs.shape[1] < 3:
        pcs = np.pad(pcs, ((0, 0), (0, 3 - pcs.shape[1])), mode="constant")

    set_labels = ["Baseline"] * len(baseline_embeddings) + ["Drifted"] * len(drifted_embeddings)
    embedding_df = pd.DataFrame({
        "pc1": pcs[:, 0],
        "pc2": pcs[:, 1],
        "pc3": pcs[:, 2],
        "set": set_labels,
    })

    classifier: Optional[ChestXrayClassifier] = None
    classifier_val_acc: Optional[float] = None
    cbpe_result: Optional[DetectionResult] = None
    drifted_actual_acc: Optional[float] = None
    baseline_confidences: Optional[np.ndarray] = None
    drifted_confidences: Optional[np.ndarray] = None

    base_labels = subset.labels[base_idx]
    valid_mask = base_labels >= 0
    if int(valid_mask.sum()) >= 30 and len(np.unique(base_labels[valid_mask])) > 1:
        valid_imgs = baseline_images[valid_mask]
        valid_y = base_labels[valid_mask]
        _, counts = np.unique(valid_y, return_counts=True)
        strat = valid_y if np.all(counts >= 2) else None
        tr_idx, val_idx = train_test_split(
            np.arange(len(valid_y)),
            test_size=0.2,
            random_state=seed,
            stratify=strat,
        )
        tr_loader = DataLoader(
            TensorDataset(valid_imgs[tr_idx], torch.from_numpy(valid_y[tr_idx])),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(valid_imgs[val_idx], torch.from_numpy(valid_y[val_idx])),
            batch_size=batch_size,
            shuffle=False,
        )
        classifier = ChestXrayClassifier()
        classifier_val_acc = train_classifier(
            classifier,
            tr_loader,
            val_loader,
            epochs=cls_epochs,
            device=resolved_device,
        )

        # Predict on held-out baseline validation split and drifted split for CBPE & confidence
        classifier.eval()
        with torch.no_grad():
            val_logits = []
            for vx, _ in val_loader:
                val_logits.append(torch.sigmoid(classifier(vx.to(resolved_device))).squeeze(1).cpu().numpy())
            val_probs = np.concatenate(val_logits, axis=0)
            val_preds = (val_probs >= 0.5).astype(np.int64)

            drift_loader = DataLoader(TensorDataset(drifted_images), batch_size=batch_size, shuffle=False)
            drift_logits = []
            for (dx,) in drift_loader:
                drift_logits.append(torch.sigmoid(classifier(dx.to(resolved_device))).squeeze(1).cpu().numpy())
            drift_probs = np.concatenate(drift_logits, axis=0)
            drift_preds = (drift_probs >= 0.5).astype(np.int64)

        baseline_confidences = np.maximum(val_probs, 1.0 - val_probs)
        drifted_confidences = np.maximum(drift_probs, 1.0 - drift_probs)

        val_ds = Dataset.from_numpy(
            baseline_embeddings[valid_mask][val_idx],
            feature_names=feature_names,
            y=valid_y[val_idx],
            predictions=val_preds,
            prediction_probabilities=val_probs,
        )
        drift_pred_ds = Dataset.from_numpy(
            drifted_embeddings,
            feature_names=feature_names,
            predictions=drift_preds,
            prediction_probabilities=drift_probs,
        )
        cbpe = CBPE(threshold=ks_threshold)
        cbpe.fit(val_ds)
        cbpe_result = cbpe.detect(drift_pred_ds)

        drift_labels = subset.labels[drift_idx]
        d_valid = drift_labels >= 0
        if int(d_valid.sum()) > 0:
            drifted_actual_acc = float(np.mean(drift_preds[d_valid] == drift_labels[d_valid]))

    return XrayAnalysisResult(
        subset=subset,
        baseline_indices=base_idx,
        drifted_indices=drift_idx,
        baseline_images=baseline_images,
        drifted_images=drifted_images,
        baseline_errors=baseline_errors,
        drifted_errors=drifted_errors,
        baseline_embeddings=baseline_embeddings,
        drifted_embeddings=drifted_embeddings,
        mmd_result=mmd_result,
        ks_result=ks_result,
        psi_result=psi_result,
        localization_result=localization_result,
        embedding_df=embedding_df,
        pca_explained_variance=pca.explained_variance_ratio_,
        classifier=classifier,
        classifier_val_acc=classifier_val_acc,
        cbpe_result=cbpe_result,
        drifted_actual_acc=drifted_actual_acc,
        baseline_confidences=baseline_confidences,
        drifted_confidences=drifted_confidences,
        baseline_ood_scores=baseline_ood_scores,
        drifted_ood_scores=drifted_ood_scores,
        drifted_ood_unclipped=drifted_ood_unclipped,
        device=resolved_device,
    )
