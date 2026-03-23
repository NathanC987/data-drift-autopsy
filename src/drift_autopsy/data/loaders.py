"""Data loaders for various formats."""

from pathlib import Path
from typing import Optional, List, Union
import pandas as pd
import logging

from drift_autopsy.core.dataset import Dataset

logger = logging.getLogger(__name__)


class DataLoader:
    """
    General data loader supporting multiple formats.
    """
    
    @staticmethod
    def from_csv(
        path: Union[str, Path],
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        metadata_cols: Optional[List[str]] = None,
        **read_kwargs
    ) -> Dataset:
        """
        Load dataset from CSV file.
        
        Args:
            path: Path to CSV file
            target_col: Target column name
            feature_cols: Feature column names
            metadata_cols: Metadata column names
            **read_kwargs: Additional arguments for pd.read_csv
        
        Returns:
            Dataset instance
        """
        logger.info(f"Loading CSV from: {path}")
        df = pd.read_csv(path, **read_kwargs)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return Dataset.from_pandas(
            df,
            target_col=target_col,
            feature_cols=feature_cols,
            metadata_cols=metadata_cols,
        )
    
    @staticmethod
    def from_parquet(
        path: Union[str, Path],
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        metadata_cols: Optional[List[str]] = None,
        **read_kwargs
    ) -> Dataset:
        """
        Load dataset from Parquet file.
        
        Args:
            path: Path to Parquet file
            target_col: Target column name
            feature_cols: Feature column names
            metadata_cols: Metadata column names
            **read_kwargs: Additional arguments for pd.read_parquet
        
        Returns:
            Dataset instance
        """
        logger.info(f"Loading Parquet from: {path}")
        df = pd.read_parquet(path, **read_kwargs)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return Dataset.from_pandas(
            df,
            target_col=target_col,
            feature_cols=feature_cols,
            metadata_cols=metadata_cols,
        )


class FolktablesLoader:
    """
    Loader for Folktables datasets.
    
    Provides convenient interface for loading ACS data by year and state.
    """
    
    @staticmethod
    def load_acs_employment(
        year: int,
        states: List[str],
        horizon: str = "1-Year",
        survey: str = "person",
        download: bool = True,
    ) -> Dataset:
        """
        Load ACS Employment dataset.
        
        Args:
            year: Survey year
            states: List of state abbreviations (e.g., ["CA", "TX"])
            horizon: Survey horizon ("1-Year" or "5-Year")
            survey: Survey type ("person" or "household")
            download: Whether to download data if not cached
        
        Returns:
            Dataset instance
        """
        try:
            from folktables import ACSDataSource, ACSEmployment
        except ImportError:
            raise ImportError(
                "folktables is required for this loader. "
                "Install with: pip install folktables"
            )
        
        logger.info(f"Loading ACS Employment data: year={year}, states={states}")
        
        data_source = ACSDataSource(
            survey_year=str(year),
            horizon=horizon,
            survey=survey
        )
        acs_data = data_source.get_data(states=states, download=download)
        features, label, group = ACSEmployment.df_to_pandas(acs_data)
        
        # Combine features and metadata
        df = features.copy()
        df['target'] = label
        df['group'] = group
        
        logger.info(f"Loaded {len(df)} samples with {len(features.columns)} features")
        
        return Dataset.from_pandas(
            df,
            target_col='target',
            feature_cols=list(features.columns),
            metadata_cols=['group'],
        )
    
    @staticmethod
    def load_acs_income(
        year: int,
        states: List[str],
        horizon: str = "1-Year",
        survey: str = "person",
        download: bool = True,
    ) -> Dataset:
        """
        Load ACS Income dataset.
        
        Args:
            year: Survey year
            states: List of state abbreviations
            horizon: Survey horizon
            survey: Survey type
            download: Whether to download data if not cached
        
        Returns:
            Dataset instance
        """
        try:
            from folktables import ACSDataSource, ACSIncome
        except ImportError:
            raise ImportError(
                "folktables is required for this loader. "
                "Install with: pip install folktables"
            )
        
        logger.info(f"Loading ACS Income data: year={year}, states={states}")
        
        data_source = ACSDataSource(
            survey_year=str(year),
            horizon=horizon,
            survey=survey
        )
        acs_data = data_source.get_data(states=states, download=download)
        features, label, group = ACSIncome.df_to_pandas(acs_data)
        
        # Combine features and metadata
        df = features.copy()
        df['target'] = label
        df['group'] = group
        # Keep original ACS state code for true geographic slicing.
        df['state'] = acs_data['ST'].reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} samples with {len(features.columns)} features")
        
        return Dataset.from_pandas(
            df,
            target_col='target',
            feature_cols=list(features.columns),
            metadata_cols=['group', 'state'],
        )
