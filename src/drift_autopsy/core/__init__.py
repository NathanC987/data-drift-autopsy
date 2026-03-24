"""Core abstractions and protocols for drift detection."""

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import (
    DetectionResult,
    LocalizationResult,
    RCAResult,
    PipelineResult,
    FeatureDrift,
    DriftSeverity,
)
from drift_autopsy.core.detector import DriftDetector, BaseDriftDetector
from drift_autopsy.core.extractor import FeatureExtractor, BaseFeatureExtractor
from drift_autopsy.core.localizer import DriftLocalizer, BaseDriftLocalizer
from drift_autopsy.core.rca import RootCauseAnalyzer, BaseRootCauseAnalyzer

__all__ = [
    "Dataset",
    "DetectionResult",
    "LocalizationResult",
    "RCAResult",
    "PipelineResult",
    "FeatureDrift",
    "DriftSeverity",
    "DriftDetector",
    "BaseDriftDetector",
    "FeatureExtractor",
    "BaseFeatureExtractor",
    "DriftLocalizer",
    "BaseDriftLocalizer",
    "RootCauseAnalyzer",
    "BaseRootCauseAnalyzer",
]
