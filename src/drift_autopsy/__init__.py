"""
Data Drift Autopsy - A modular drift detection framework.
"""

__version__ = "0.1.0"

# Core components
from drift_autopsy.core.pipeline import DriftPipeline
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import (
    DetectionResult,
    LocalizationResult,
    RCAResult,
    PipelineResult,
    DriftSeverity,
)

# Registries
from drift_autopsy.registry import DetectorRegistry, ExtractorRegistry, LocalizerRegistry, RCARegistry

# Import detectors to trigger registration
from drift_autopsy.detectors import (
    KSTest,
    PSI,
    MMD,
    PCAReconstructionError,
    FIDDistance,
    DomainClassifier,
    CBPE,
)

# Import localizers
from drift_autopsy.localizers import UnivariateLocalizer

# Import RCA analyzers
from drift_autopsy.rca import SHAPAnalyzer

# Data utilities
from drift_autopsy.data import (
    DataLoader,
    FolktablesLoader,
    EmbeddingBaselineClassifier,
    create_monitored_model,
    ImageTabularizationRunner,
    MulticlassProxyEstimator,
    build_image_tabularized_buckets,
)

__all__ = [
    "DriftPipeline",
    "Dataset",
    "DetectionResult",
    "LocalizationResult",
    "RCAResult",
    "PipelineResult",
    "DriftSeverity",
    "DetectorRegistry",
    "ExtractorRegistry",
    "LocalizerRegistry",
    "RCARegistry",
    "KSTest",
    "PSI",
    "MMD",
    "PCAReconstructionError",
    "FIDDistance",
    "DomainClassifier",
    "CBPE",
    "UnivariateLocalizer",
    "SHAPAnalyzer",
    "DataLoader",
    "FolktablesLoader",
    "EmbeddingBaselineClassifier",
    "create_monitored_model",
    "ImageTabularizationRunner",
    "MulticlassProxyEstimator",
    "build_image_tabularized_buckets",
]
