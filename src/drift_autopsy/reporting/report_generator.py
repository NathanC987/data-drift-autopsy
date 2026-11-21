"""Structured drift reporting for Chest X-ray drift analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from drift_autopsy.core.result import DetectionResult


@dataclass
class DriftReport:
    """Structured summary of drift detection, localization, and RCA findings."""

    drift_detected: bool
    drift_type: str
    severity: str
    confidence: float
    affected_samples_percent: float
    baseline_recon_mean: float
    baseline_recon_std: float
    drifted_recon_mean: float
    recon_threshold: float
    focus_shift: Optional[float]
    detector_summary: Dict[str, Dict[str, Any]]
    root_causes: List[str]
    key_findings: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def generate_drift_report(
    baseline_errors: np.ndarray,
    drifted_errors: np.ndarray,
    mmd_result: DetectionResult,
    ks_result: DetectionResult,
    psi_result: DetectionResult,
    baseline_focus_score: Optional[float] = None,
    drifted_focus_score: Optional[float] = None,
    brightness: float = 1.0,
    contrast: float = 1.0,
    noise_std: float = 0.0,
    classifier_val_acc: Optional[float] = None,
) -> DriftReport:
    """Generate a structured DriftReport from reconstruction, detector, and Grad-CAM metrics."""
    base_mean = float(np.mean(baseline_errors))
    base_std = float(np.std(baseline_errors))
    drift_mean = float(np.mean(drifted_errors))
    recon_threshold = base_mean + 2.5 * base_std
    affected_pct = float(np.mean(drifted_errors > recon_threshold) * 100.0)

    focus_shift: Optional[float] = None
    if baseline_focus_score is not None and drifted_focus_score is not None:
        focus_shift = float(drifted_focus_score - baseline_focus_score)

    detectors = [mmd_result, ks_result, psi_result]
    n_triggered = sum(1 for d in detectors if d.drift_detected)
    drift_detected = n_triggered > 0 or affected_pct >= 10.0

    if n_triggered == 3 or affected_pct >= 50.0:
        severity = "high"
    elif n_triggered >= 2 or affected_pct >= 20.0:
        severity = "medium"
    elif drift_detected:
        severity = "low"
    else:
        severity = "none"

    confidence = float(min(1.0, 0.25 * n_triggered + 0.25 * min(1.0, affected_pct / 50.0)))

    has_photometric = (abs(brightness - 1.0) > 1e-3) or (abs(contrast - 1.0) > 1e-3) or (noise_std > 1e-4)
    if drift_detected and has_photometric:
        drift_type = "Covariate Shift (Acquisition / Photometric)"
    elif drift_detected:
        drift_type = "Covariate Shift (Latent Feature Distribution)"
    else:
        drift_type = "No Significant Drift"

    root_causes: List[str] = []
    if abs(brightness - 1.0) > 1e-3:
        root_causes.append(f"Brightness shift applied (factor={brightness:.2f}).")
    if abs(contrast - 1.0) > 1e-3:
        root_causes.append(f"Contrast shift applied (factor={contrast:.2f}).")
    if noise_std > 1e-4:
        root_causes.append(f"Additive Gaussian sensor noise (std={noise_std:.3f}).")
    if focus_shift is not None and abs(focus_shift) > 0.02:
        direction = "increased" if focus_shift > 0 else "decreased"
        root_causes.append(f"CNN Grad-CAM top-10% focus {direction} by {focus_shift:+.3f}.")
    if not root_causes:
        root_causes.append("No strong acquisition or saliency shift identified.")

    key_findings = [
        f"Reconstruction MSE shifted from {base_mean:.5f} ± {base_std:.5f} to {drift_mean:.5f} "
        f"({affected_pct:.1f}% above 2.5σ threshold {recon_threshold:.5f}).",
        f"MMD: score={mmd_result.score:.4f}, p={mmd_result.p_value:.4f} "
        f"({'DRIFT' if mmd_result.drift_detected else 'OK'}).",
        f"KS Test: score={ks_result.score:.4f}, p={ks_result.p_value:.4f} "
        f"({'DRIFT' if ks_result.drift_detected else 'OK'}).",
        f"PSI: score={psi_result.score:.4f} vs threshold={psi_result.threshold:.2f} "
        f"({'DRIFT' if psi_result.drift_detected else 'OK'}).",
    ]
    if classifier_val_acc is not None:
        key_findings.append(f"Baseline CNN classifier validation accuracy: {classifier_val_acc:.1%}.")
    if focus_shift is not None:
        key_findings.append(f"Average Grad-CAM top-10% focus shift: {focus_shift:+.4f}.")

    recommendations: List[str] = []
    if drift_detected:
        recommendations.append("Inspect DICOM/scanner exposure, contrast normalization, and CLAHE/histogram preprocessing.")
        recommendations.append("Apply photometric augmentation during model retraining to improve invariance.")
        if affected_pct >= 20.0:
            recommendations.append("Flag high-reconstruction-error scans for clinical/radiologist review before automated inference.")
    else:
        recommendations.append("Continue routine monitoring; no immediate recalibration or retraining required.")

    detector_summary = {
        "mmd": {
            "score": float(mmd_result.score),
            "threshold": float(mmd_result.threshold),
            "p_value": float(mmd_result.p_value) if mmd_result.p_value is not None else None,
            "drift_detected": bool(mmd_result.drift_detected),
        },
        "ks_test": {
            "score": float(ks_result.score),
            "threshold": float(ks_result.threshold),
            "p_value": float(ks_result.p_value) if ks_result.p_value is not None else None,
            "drift_detected": bool(ks_result.drift_detected),
        },
        "psi": {
            "score": float(psi_result.score),
            "threshold": float(psi_result.threshold),
            "drift_detected": bool(psi_result.drift_detected),
        },
    }

    return DriftReport(
        drift_detected=drift_detected,
        drift_type=drift_type,
        severity=severity,
        confidence=confidence,
        affected_samples_percent=affected_pct,
        baseline_recon_mean=base_mean,
        baseline_recon_std=base_std,
        drifted_recon_mean=drift_mean,
        recon_threshold=recon_threshold,
        focus_shift=focus_shift,
        detector_summary=detector_summary,
        root_causes=root_causes,
        key_findings=key_findings,
        recommendations=recommendations,
    )
