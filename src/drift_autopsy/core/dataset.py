"""Dataset abstraction for drift detection."""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class Dataset:
    """
    Abstract dataset container that wraps data and metadata.
    
    Provides a unified interface for different data sources and formats.
    Supports both pandas DataFrames and numpy arrays.

    ``categorical_features`` names the nominal features (codes whose numeric
    order means nothing, e.g. an occupation code). Detectors test them with
    count-based statistics instead of distribution-shape statistics. When it is
    ``None`` it is inferred from the column dtypes (object, category, bool).
    """
    
    data: Union[pd.DataFrame, np.ndarray]
    feature_names: Optional[List[str]] = None
    target: Optional[Union[pd.Series, np.ndarray]] = None
    target_name: Optional[str] = None
    predictions: Optional[np.ndarray] = None
    prediction_probabilities: Optional[np.ndarray] = None
    metadata: Optional[pd.DataFrame] = None
    categorical_features: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize and validate dataset."""
        # Extract feature names from DataFrame if not provided
        if self.feature_names is None:
            if isinstance(self.data, pd.DataFrame):
                self.feature_names = list(self.data.columns)
            else:
                # Generate default names for numpy arrays
                n_features = self.data.shape[1] if len(self.data.shape) > 1 else 1
                self.feature_names = [f"feature_{i}" for i in range(n_features)]

        if self.categorical_features is None:
            if isinstance(self.data, pd.DataFrame):
                self.categorical_features = [
                    c for c in self.data.columns
                    if self.data[c].dtype == object
                    or isinstance(self.data[c].dtype, pd.CategoricalDtype)
                    or self.data[c].dtype == bool
                ]
            else:
                self.categorical_features = []
        else:
            unknown = set(self.categorical_features) - set(self.feature_names)
            if unknown:
                raise ValueError(f"categorical_features not in feature_names: {sorted(unknown)}")
            self.categorical_features = list(self.categorical_features)

    @property
    def numeric_features(self) -> List[str]:
        """Features that are not categorical, in feature order."""
        categorical = set(self.categorical_features or [])
        return [f for f in self.feature_names if f not in categorical]
    
    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        prediction_col: Optional[str] = None,
        prediction_proba_col: Optional[str] = None,
        metadata_cols: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ) -> "Dataset":
        """
        Create Dataset from pandas DataFrame.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column (if present)
            feature_cols: List of feature column names (if None, infer from df)
            prediction_col: Name of prediction column
            prediction_proba_col: Name of prediction probability column
            metadata_cols: Columns to treat as metadata (not features)
            categorical_features: Nominal feature names (inferred from dtypes if None)
        
        Returns:
            Dataset instance
        """
        df = df.copy()
        
        # Extract target
        target = None
        target_name = None
        if target_col and target_col in df.columns:
            target = df[target_col]
            target_name = target_col
            df = df.drop(columns=[target_col])
        
        # Extract predictions
        predictions = None
        if prediction_col and prediction_col in df.columns:
            predictions = df[prediction_col].values
            df = df.drop(columns=[prediction_col])
        
        # Extract prediction probabilities
        prediction_probabilities = None
        if prediction_proba_col and prediction_proba_col in df.columns:
            prediction_probabilities = df[prediction_proba_col].values
            df = df.drop(columns=[prediction_proba_col])
        
        # Extract metadata
        metadata = None
        if metadata_cols:
            metadata = df[metadata_cols].copy()
            df = df.drop(columns=metadata_cols)
        
        # Determine feature columns
        if feature_cols:
            features = df[feature_cols]
        else:
            features = df
        
        return cls(
            data=features,
            feature_names=list(features.columns),
            target=target,
            target_name=target_name,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            metadata=metadata,
            categorical_features=categorical_features,
        )
    
    @classmethod
    def from_numpy(
        cls,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        y: Optional[np.ndarray] = None,
        target_name: Optional[str] = None,
        predictions: Optional[np.ndarray] = None,
        prediction_probabilities: Optional[np.ndarray] = None,
    ) -> "Dataset":
        """
        Create Dataset from numpy arrays.
        
        Args:
            X: Feature array
            feature_names: Names of features
            y: Target array
            target_name: Name of target variable
            predictions: Model predictions
            prediction_probabilities: Prediction probabilities
        
        Returns:
            Dataset instance
        """
        return cls(
            data=X,
            feature_names=feature_names,
            target=y,
            target_name=target_name,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
        )
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.copy()
        else:
            return pd.DataFrame(self.data, columns=self.feature_names)
    
    def to_numpy(self) -> np.ndarray:
        """Convert dataset to numpy array."""
        if isinstance(self.data, np.ndarray):
            return self.data
        else:
            return self.data.values
    
    @property
    def n_samples(self) -> int:
        """Number of samples in dataset."""
        return len(self.data)
    
    @property
    def n_features(self) -> int:
        """Number of features in dataset."""
        return len(self.feature_names) if self.feature_names else 0
    
    @property
    def shape(self) -> tuple:
        """Shape of dataset."""
        return (self.n_samples, self.n_features)
    
    def get_feature(self, feature_name: str) -> np.ndarray:
        """
        Get values for a specific feature.
        
        Args:
            feature_name: Name of feature
        
        Returns:
            Feature values as numpy array
        """
        if isinstance(self.data, pd.DataFrame):
            return self.data[feature_name].values
        else:
            idx = self.feature_names.index(feature_name)
            return self.data[:, idx]
    
    def get_features(self, feature_names: List[str]) -> np.ndarray:
        """
        Get values for multiple features.
        
        Args:
            feature_names: List of feature names
        
        Returns:
            Feature matrix as numpy array
        """
        if isinstance(self.data, pd.DataFrame):
            return self.data[feature_names].values
        else:
            indices = [self.feature_names.index(name) for name in feature_names]
            return self.data[:, indices]
    
    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Dataset(n_samples={self.n_samples}, n_features={self.n_features})"
