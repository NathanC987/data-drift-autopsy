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


class ImageDataConfig(BaseModel):
    """Configuration for image-derived (embedding-first) pipelines."""

    dataset: str = Field(default="clear10", description="Image dataset adapter name")
    root_path: str = Field(..., description="Dataset root path")
    reference_bucket: int = Field(default=1, ge=1, description="Reference bucket id")
    analysis_buckets: List[int] = Field(
        default_factory=lambda: [2, 3, 4, 5, 6, 7, 8, 9, 10],
        description="Analysis bucket ids",
    )
    include_background: bool = Field(
        default=True,
        description="Whether to include BACKGROUND class records",
    )
    extractor_name: str = Field(default="resnet", description="Embedding extractor key")
    extractor_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extractor constructor parameters",
    )
    expected_embedding_dim: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional expected embedding dimensionality",
    )
    expected_class_count: Optional[int] = Field(
        default=None,
        ge=2,
        description="Optional expected class count",
    )
    metadata_fields: List[str] = Field(
        default_factory=lambda: ["device", "user_tags", "lon", "lat"],
        description="Metadata keys to carry into tabularized records",
    )
    allow_missing_analysis_y_true: bool = Field(
        default=True,
        description="Allow missing/delayed y_true values in analysis buckets",
    )
    chunking_strategy: str = Field(
        default="fixed_bucket",
        description="Chunking strategy: fixed_bucket, quantity, temporal, or sliding_window",
    )
    chunk_size_records: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional chunk size for quantity-based chunking",
    )
    chunk_duration: Optional[str] = Field(
        default=None,
        description="Optional duration (e.g. '1M', '7D') for temporal chunking",
    )
    reference_mode: str = Field(
        default="fixed_reference",
        description="Reference mode: fixed_reference or previous_chunk",
    )
    artifacts_dir: Optional[str] = Field(
        default=None,
        description="Optional output directory for extracted embeddings and reports",
    )
    max_samples_per_class: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional cap per class per bucket for extraction/debug runs",
    )
    bootstrap_predictions_from_y_true: bool = Field(
        default=True,
        description=(
            "If true, generate temporary y_pred/pred_proba_* from y_true so downstream "
            "contract checks and dashboard wiring can run before classifier training."
        ),
    )
    baseline_model_name: str = Field(
        default="logistic_regression",
        description=(
            "Monitored-model adapter name for predictions. "
            "Current default is logistic_regression over embeddings."
        ),
    )
    baseline_model_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Baseline classifier constructor parameters",
    )
    monitored_model_name: Optional[str] = Field(
        default=None,
        description=(
            "Optional explicit monitored-model adapter name. "
            "If unset, baseline_model_name is used for backward compatibility."
        ),
    )
    monitored_model_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional explicit monitored-model adapter parameters. "
            "If empty, baseline_model_params is used for backward compatibility."
        ),
    )
    baseline_train_fraction: float = Field(
        default=0.7,
        gt=0.0,
        lt=1.0,
        description="Reference-bucket train fraction for baseline evaluation split",
    )
    baseline_random_state: int = Field(
        default=42,
        description="Random seed for baseline split/training",
    )

    @field_validator("analysis_buckets")
    @classmethod
    def validate_analysis_buckets(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("analysis_buckets cannot be empty")
        if any(bucket < 1 for bucket in v):
            raise ValueError("analysis_buckets must contain positive bucket ids")
        return v

    @field_validator("chunking_strategy")
    @classmethod
    def validate_chunking_strategy(cls, v: str) -> str:
        valid = {"fixed_bucket", "quantity", "temporal", "sliding_window"}
        value = v.strip().lower()
        if value not in valid:
            raise ValueError(f"chunking_strategy must be one of: {sorted(valid)}")
        return value

    @field_validator("reference_mode")
    @classmethod
    def validate_reference_mode(cls, v: str) -> str:
        valid = {"fixed_reference", "previous_chunk"}
        value = v.strip().lower()
        if value not in valid:
            raise ValueError(f"reference_mode must be one of: {sorted(valid)}")
        return value


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
    image_data: Optional[ImageDataConfig] = Field(
        None,
        description="Image/embedding pipeline configuration",
    )
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
