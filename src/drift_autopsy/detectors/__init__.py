"""Drift detection implementations."""

# Import all detectors to trigger registration
from drift_autopsy.detectors.statistical.ks_test import KSTest
from drift_autopsy.detectors.statistical.psi import PSI
from drift_autopsy.detectors.distribution.mmd import MMD
from drift_autopsy.detectors.distribution.pca_reconstruction import PCAReconstructionError
from drift_autopsy.detectors.distribution.fid_distance import FIDDistance
from drift_autopsy.detectors.model_based.domain_classifier import DomainClassifier
from drift_autopsy.detectors.proxy.cbpe import CBPE

__all__ = [
    "KSTest",
    "PSI",
    "MMD",
    "PCAReconstructionError",
    "FIDDistance",
    "DomainClassifier",
    "CBPE",
]
