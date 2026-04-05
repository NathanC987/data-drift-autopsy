"""Component registry for discovery."""

from drift_autopsy.registry.detector_registry import DetectorRegistry
from drift_autopsy.registry.extractor_registry import ExtractorRegistry
from drift_autopsy.registry.localizer_registry import LocalizerRegistry
from drift_autopsy.registry.rca_registry import RCARegistry

__all__ = [
    "DetectorRegistry",
    "ExtractorRegistry",
    "LocalizerRegistry",
    "RCARegistry",
]
