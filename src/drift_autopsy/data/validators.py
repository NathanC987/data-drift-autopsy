"""Data validators for sanity checks."""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List

from drift_autopsy.core.dataset import Dataset

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate datasets for common issues.
    
    Checks for missing values, data types, shape consistency, etc.
    """
    
    @staticmethod
    def validate_dataset(
        dataset: Dataset,
        name: str = "dataset",
        check_missing: bool = True,
        check_inf: bool = True,
        check_variance: bool = True,
        min_samples: int = 10,
    ) -> None:
        """
        Validate a dataset and log warnings for issues.
        
        Args:
            dataset: Dataset to validate
            name: Name for logging
            check_missing: Check for missing values
            check_inf: Check for infinite values
            check_variance: Check for zero-variance features
            min_samples: Minimum number of samples required
        
        Raises:
            ValueError: If critical validation fails
        """
        logger.info(f"Validating {name}: shape={dataset.shape}")
        
        # Check minimum samples
        if dataset.n_samples < min_samples:
            raise ValueError(
                f"{name} has only {dataset.n_samples} samples, "
                f"minimum {min_samples} required"
            )
        
        # Convert to DataFrame for easier validation
        df = dataset.to_pandas()
        
        # Check for missing values
        if check_missing:
            missing_counts = df.isnull().sum()
            if missing_counts.any():
                missing_features = missing_counts[missing_counts > 0]
                logger.warning(
                    f"{name} has missing values in {len(missing_features)} features: "
                    f"{dict(missing_features.head())}"
                )
        
        # Check for infinite values
        if check_inf:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    logger.warning(
                        f"{name} has {inf_count} infinite values in feature '{col}'"
                    )
        
        # Check for zero-variance features
        if check_variance:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].std() == 0:
                    logger.warning(
                        f"{name} has zero variance in feature '{col}' (constant value)"
                    )
        
        logger.info(f"{name} validation complete")
    
    @staticmethod
    def validate_compatibility(
        reference: Dataset,
        test: Dataset,
        check_feature_names: bool = True,
        check_feature_order: bool = True,
    ) -> None:
        """
        Validate that two datasets are compatible for drift detection.
        
        Args:
            reference: Reference dataset
            test: Test dataset
            check_feature_names: Check feature names match
            check_feature_order: Check feature order matches
        
        Raises:
            ValueError: If datasets are incompatible
        """
        logger.info("Validating dataset compatibility")
        
        # Check same number of features
        if reference.n_features != test.n_features:
            raise ValueError(
                f"Feature count mismatch: reference has {reference.n_features}, "
                f"test has {test.n_features}"
            )
        
        # Check feature names match
        if check_feature_names:
            ref_features = set(reference.feature_names)
            test_features = set(test.feature_names)
            
            missing_in_test = ref_features - test_features
            extra_in_test = test_features - ref_features
            
            if missing_in_test:
                raise ValueError(
                    f"Features in reference but not in test: {missing_in_test}"
                )
            
            if extra_in_test:
                raise ValueError(
                    f"Features in test but not in reference: {extra_in_test}"
                )
        
        # Check feature order
        if check_feature_order:
            if reference.feature_names != test.feature_names:
                logger.warning(
                    "Feature order differs between reference and test. "
                    "This may affect some detectors."
                )
        
        logger.info("Dataset compatibility check passed")

    @staticmethod
    def validate_embedding_contract(
        df: pd.DataFrame,
        name: str = "embedding_dataset",
        embedding_prefix: str = "feature_",
        proba_prefix: str = "pred_proba_",
        require_timestamp: bool = True,
        require_sample_id: bool = True,
        allow_missing_y_true: bool = True,
        expected_embedding_dim: Optional[int] = None,
        expected_class_count: Optional[int] = None,
    ) -> None:
        """
        Validate standardized image-derived tabular contract.

        Required by default:
          - embedding columns: feature_*
          - probability columns: pred_proba_*
          - y_pred
          - timestamp
          - sample_id
        """
        if df.empty:
            raise ValueError(f"{name} is empty")

        required_cols = ["y_pred"]
        if require_timestamp:
            required_cols.append("timestamp")
        if require_sample_id:
            required_cols.append("sample_id")

        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            raise ValueError(f"{name} missing required columns: {missing_required}")

        embedding_cols = [col for col in df.columns if col.startswith(embedding_prefix)]
        if not embedding_cols:
            raise ValueError(
                f"{name} has no embedding columns with prefix '{embedding_prefix}'"
            )

        proba_cols = [col for col in df.columns if col.startswith(proba_prefix)]
        if not proba_cols:
            raise ValueError(
                f"{name} has no probability columns with prefix '{proba_prefix}'"
            )

        if expected_embedding_dim is not None and len(embedding_cols) != expected_embedding_dim:
            raise ValueError(
                f"{name} embedding dim mismatch: expected {expected_embedding_dim}, "
                f"found {len(embedding_cols)}"
            )

        if expected_class_count is not None and len(proba_cols) != expected_class_count:
            raise ValueError(
                f"{name} probability class count mismatch: expected {expected_class_count}, "
                f"found {len(proba_cols)}"
            )

        # Ensure numeric embedding/probability values.
        if df[embedding_cols].isnull().any().any():
            raise ValueError(f"{name} contains missing values in embedding columns")
        if df[proba_cols].isnull().any().any():
            raise ValueError(f"{name} contains missing values in probability columns")

        if not np.isfinite(df[embedding_cols].to_numpy()).all():
            raise ValueError(f"{name} contains non-finite values in embeddings")
        if not np.isfinite(df[proba_cols].to_numpy()).all():
            raise ValueError(f"{name} contains non-finite values in probabilities")

        proba_values = df[proba_cols].to_numpy(dtype=float)
        if ((proba_values < 0.0) | (proba_values > 1.0)).any():
            raise ValueError(f"{name} probability values must be within [0, 1]")

        row_sums = proba_values.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-3, atol=1e-3):
            raise ValueError(f"{name} probability rows must sum to 1")

        if not allow_missing_y_true and "y_true" not in df.columns:
            raise ValueError(f"{name} requires y_true but column is missing")

        if "y_true" in df.columns and not allow_missing_y_true:
            if df["y_true"].isnull().any():
                raise ValueError(f"{name} contains missing y_true values")

        logger.info(
            "%s embedding contract valid: n_rows=%s embedding_dim=%s n_classes=%s",
            name,
            len(df),
            len(embedding_cols),
            len(proba_cols),
        )

    @staticmethod
    def validate_embedding_compatibility(
        reference_df: pd.DataFrame,
        test_df: pd.DataFrame,
        embedding_prefix: str = "feature_",
        proba_prefix: str = "pred_proba_",
    ) -> None:
        """Validate reference/test embedding datasets share the same contract shape."""
        ref_embedding_cols = sorted(
            [col for col in reference_df.columns if col.startswith(embedding_prefix)]
        )
        test_embedding_cols = sorted(
            [col for col in test_df.columns if col.startswith(embedding_prefix)]
        )
        if ref_embedding_cols != test_embedding_cols:
            raise ValueError("Reference/test embedding columns do not match")

        ref_proba_cols = sorted(
            [col for col in reference_df.columns if col.startswith(proba_prefix)]
        )
        test_proba_cols = sorted(
            [col for col in test_df.columns if col.startswith(proba_prefix)]
        )
        if ref_proba_cols != test_proba_cols:
            raise ValueError("Reference/test probability columns do not match")

        logger.info("Embedding compatibility check passed")
