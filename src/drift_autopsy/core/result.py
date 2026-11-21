"""Result dataclasses for drift detection and analysis."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and containers to plain Python for JSON output."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value


class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DetectionResult:
    """Result of drift detection."""
    
    detector_name: str
    drift_detected: bool
    severity: DriftSeverity
    score: float
    threshold: float
    p_value: Optional[float] = None
    statistic: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detector_name": self.detector_name,
            "drift_detected": self.drift_detected,
            "severity": self.severity.value,
            "score": self.score,
            "threshold": self.threshold,
            "p_value": self.p_value,
            "statistic": self.statistic,
            "metadata": self.metadata,
        }


@dataclass
class FeatureDrift:
    """Drift information for a single feature."""
    
    feature_name: str
    drift_detected: bool
    score: float
    p_value: Optional[float] = None
    severity: DriftSeverity = DriftSeverity.NONE
    distribution_shift: Optional[Dict[str, Any]] = None


@dataclass
class LocalizationResult:
    """Result of drift localization."""
    
    method_name: str
    feature_drifts: List[FeatureDrift]
    drifted_features: List[str] = field(default_factory=list)
    drift_scores: Dict[str, float] = field(default_factory=dict)
    
    # Slice-based localization
    slice_drifts: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Extract drifted features for convenience."""
        self.drifted_features = [
            fd.feature_name for fd in self.feature_drifts if fd.drift_detected
        ]
        self.drift_scores = {
            fd.feature_name: fd.score for fd in self.feature_drifts
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method_name": self.method_name,
            "drifted_features": self.drifted_features,
            "drift_scores": self.drift_scores,
            "feature_drifts": [
                {
                    "feature_name": fd.feature_name,
                    "drift_detected": fd.drift_detected,
                    "score": fd.score,
                    "p_value": fd.p_value,
                    "severity": fd.severity.value,
                    "distribution_shift": _json_safe(fd.distribution_shift),
                }
                for fd in self.feature_drifts
            ],
            "slice_drifts": self.slice_drifts,
            "metadata": self.metadata,
        }


@dataclass
class RCAResult:
    """Result of root cause analysis."""
    
    analyzer_name: str
    explanations: Dict[str, Any]
    feature_importances: Optional[Dict[str, float]] = None
    distribution_changes: Optional[Dict[str, Any]] = None
    visualizations: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "analyzer_name": self.analyzer_name,
            "explanations": self.explanations,
            "feature_importances": self.feature_importances,
            "distribution_changes": self.distribution_changes,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


@dataclass
class PipelineResult:
    """Complete result from drift analysis pipeline."""
    
    detection: DetectionResult
    localization: Optional[LocalizationResult] = None
    rca: Optional[RCAResult] = None
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "detection": self.detection.to_dict(),
            "localization": self.localization.to_dict() if self.localization else None,
            "rca": self.rca.to_dict() if self.rca else None,
            "execution_time_seconds": self.execution_time_seconds,
            "metadata": self.metadata,
        }
