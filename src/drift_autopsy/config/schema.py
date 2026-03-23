"""Pydantic schemas for configuration management."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator


class DetectorConfig(BaseModel):
    """Configuration for a drift detector."""
    
    type: str = Field(..., description="Detector type/name (as registered)")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Detection threshold")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional detector parameters")
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate detector type is not empty."""
        if not v or not v.strip():
            raise ValueError("Detector type cannot be empty")
        return v.strip()


class LocalizerConfig(BaseModel):
    """Configuration for a drift localizer."""
    
    type: str = Field(..., description="Localizer type/name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Localizer parameters")


class RCAConfig(BaseModel):
    """Configuration for root cause analysis."""
    
    type: str = Field(..., description="RCA analyzer type/name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Analyzer parameters")


class SliceConfig(BaseModel):
    """Configuration for generic slice-based analysis."""

    enabled: bool = Field(default=False, description="Enable slice-based drift analysis")
    column: Optional[str] = Field(None, description="Metadata column name used for slicing")
    reference_slice_value: Optional[str] = Field(
        None,
        description="Optional fixed reference slice value (for cross-slice comparisons)",
    )
    min_samples_per_slice: int = Field(
        default=30,
        ge=1,
        description="Minimum required rows in both reference and test slice datasets",
    )


class FeatureGroupConfig(BaseModel):
    """Placeholder configuration for future feature-group drift analysis."""

    enabled: bool = Field(default=False, description="Enable feature-group analysis")
    groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Mapping from group name to feature list",
    )


class DataConfig(BaseModel):
    """Configuration for data loading."""
    
    reference_path: str = Field(..., description="Path to reference data")
    test_path: str = Field(..., description="Path to test data")
    format: str = Field(default="csv", description="Data format (csv, parquet, etc.)")
    target_col: Optional[str] = Field(None, description="Name of target column")
    feature_cols: Optional[List[str]] = Field(None, description="List of feature columns")
    metadata_cols: Optional[List[str]] = Field(None, description="Metadata columns")


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""
    
    name: str = Field(..., description="Pipeline name/identifier")
    detector: DetectorConfig = Field(..., description="Detector configuration")
    localizer: Optional[LocalizerConfig] = Field(None, description="Localizer configuration")
    rca: Optional[RCAConfig] = Field(None, description="RCA configuration")
    slice_analysis: Optional[SliceConfig] = Field(None, description="Optional slice analysis configuration")
    feature_groups: Optional[FeatureGroupConfig] = Field(
        None,
        description="Optional feature-group analysis placeholders",
    )
    data: Optional[DataConfig] = Field(None, description="Data configuration")
    enable_localization: bool = Field(default=True, description="Enable drift localization")
    enable_rca: bool = Field(default=False, description="Enable root cause analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "name": "temporal_drift_analysis",
                "detector": {
                    "type": "ks_test",
                    "threshold": 0.05,
                    "params": {"correction": "bonferroni"}
                },
                "localizer": {
                    "type": "univariate",
                    "params": {"method": "sequential"}
                },
                "enable_localization": True,
                "enable_rca": False
            }
        }
