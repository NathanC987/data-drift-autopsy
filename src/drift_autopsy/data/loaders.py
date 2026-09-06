"""Data loaders for various formats."""

from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import pandas as pd
import logging
import json

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

    ACS_INCOME_FEATURES = [
        "AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P",
    ]

    @classmethod
    def load_acs_income_cached(
        cls,
        year: int,
        state: str,
        data_root: Union[str, Path] = "data",
        dataset_name: str = "folktables_us_census",
        download: bool = False,
    ) -> "pd.DataFrame":
        """Load one ACS Income (state, year) as a DataFrame, local cache first.

        Returns a frame with the 10 ACS Income feature columns, a ``target``
        column, and a ``state`` metadata column. Falls back to a Folktables
        download only when ``download=True`` and no cache exists.
        """
        cache = (
            Path(data_root) / dataset_name / "acs_income" / f"acs_income_{state}_{year}.parquet"
        )
        if cache.exists():
            return pd.read_parquet(cache)

        csv_cache = cache.with_suffix(".csv")
        if csv_cache.exists():
            return pd.read_csv(csv_cache)

        if not download:
            raise FileNotFoundError(
                f"No cached ACS Income data at {cache}. Pass download=True to fetch it."
            )

        dataset = cls.load_acs_income(year=year, states=[state], download=True)
        frame = dataset.data.copy()
        frame["target"] = dataset.target.to_numpy() if hasattr(dataset.target, "to_numpy") else dataset.target
        if dataset.metadata is not None and "state" in dataset.metadata.columns:
            frame["state"] = dataset.metadata["state"].to_numpy()
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache, index=False)
        return frame


class Clear10Loader:
    """
    Loader utilities for CLEAR-10 folder-based image and metadata layout.

    Expected root structure:
      - labeled_images/<bucket>/<class>/*.jpg
      - labeled_metadata/<bucket>/<class>.json
      - class_names.txt
    """

    @staticmethod
    def list_buckets(root_path: Union[str, Path]) -> List[int]:
        """List available integer bucket directories under labeled_images."""
        root = Path(root_path)
        image_root = root / "labeled_images"
        if not image_root.exists():
            raise FileNotFoundError(f"CLEAR-10 labeled_images folder not found: {image_root}")

        buckets: List[int] = []
        for child in image_root.iterdir():
            if child.is_dir() and child.name.isdigit():
                buckets.append(int(child.name))
        return sorted(buckets)

    @staticmethod
    def load_class_names(root_path: Union[str, Path]) -> List[str]:
        """Load ordered class names from class_names.txt."""
        root = Path(root_path)
        class_names_path = root / "class_names.txt"
        if not class_names_path.exists():
            raise FileNotFoundError(f"CLEAR-10 class_names.txt not found: {class_names_path}")

        class_names: List[str] = []
        with open(class_names_path, "r") as f:
            for line in f:
                value = line.strip()
                if value:
                    class_names.append(value)

        if not class_names:
            raise ValueError("CLEAR-10 class_names.txt is empty")

        return class_names

    @staticmethod
    def load_metadata_index(root_path: Union[str, Path]) -> Dict[str, Dict[str, str]]:
        """Load the top-level metadata index mapping bucket->class->json path."""
        root = Path(root_path)
        index_path = root / "labeled_metadata.json"
        if not index_path.exists():
            raise FileNotFoundError(f"CLEAR-10 labeled_metadata.json not found: {index_path}")

        with open(index_path, "r") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError("CLEAR-10 labeled_metadata.json must be a dictionary")

        return payload

    @staticmethod
    def _safe_bucket_timestamp(date_taken: str, bucket: int) -> str:
        """
        Return ISO-like timestamp from DATE_TAKEN, with deterministic fallback.

        DATE_TAKEN in CLEAR metadata often looks like "YYYY-MM-DD HH:MM:SS.0".
        """
        if isinstance(date_taken, str) and date_taken.strip():
            return date_taken.strip()
        # Fallback keeps temporal ordering by bucket even if date metadata is missing.
        return f"bucket-{bucket:02d}"

    @staticmethod
    def _resolve_image_path(root: Path, bucket: int, class_name: str, sample_id: str) -> Optional[Path]:
        """Resolve image path for a sample id using common extensions."""
        class_dir = root / "labeled_images" / str(bucket) / class_name
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            candidate = class_dir / f"{sample_id}{ext}"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def build_bucket_dataframe(
        root_path: Union[str, Path],
        bucket: int,
        include_background: bool = True,
        max_samples_per_class: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build normalized tabular records for one CLEAR-10 bucket.

        Output columns include:
          sample_id, bucket, class_name, y_true, image_path,
          timestamp, source, device, user_tags, lon, lat
        """
        root = Path(root_path)
        class_names = Clear10Loader.load_class_names(root)
        metadata_index = Clear10Loader.load_metadata_index(root)

        bucket_key = str(bucket)
        if bucket_key not in metadata_index:
            raise ValueError(f"Bucket {bucket} not found in CLEAR-10 metadata index")

        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

        rows: List[Dict[str, Any]] = []
        for class_name in class_names:
            if not include_background and class_name.upper() == "BACKGROUND":
                continue

            class_metadata_rel = metadata_index[bucket_key].get(class_name)
            if class_metadata_rel is None:
                logger.warning(
                    "Missing metadata for bucket=%s class=%s in labeled_metadata.json",
                    bucket,
                    class_name,
                )
                continue

            class_metadata_path = root / class_metadata_rel
            if not class_metadata_path.exists():
                logger.warning("Metadata file not found: %s", class_metadata_path)
                continue

            with open(class_metadata_path, "r") as f:
                class_records = json.load(f)

            if not isinstance(class_records, dict):
                logger.warning("Unexpected metadata format in %s", class_metadata_path)
                continue

            n_taken = 0
            for sample_id, meta in class_records.items():
                if max_samples_per_class is not None and n_taken >= max_samples_per_class:
                    break

                sample_id_str = str(sample_id)
                image_path = Clear10Loader._resolve_image_path(root, bucket, class_name, sample_id_str)
                if image_path is None:
                    # Keep ingestion robust in presence of occasional metadata/image mismatches.
                    continue

                date_taken = meta.get("DATE_TAKEN", "") if isinstance(meta, dict) else ""
                rows.append(
                    {
                        "sample_id": sample_id_str,
                        "bucket": int(bucket),
                        "class_name": class_name,
                        "y_true": class_to_idx[class_name],
                        "image_path": str(image_path),
                        "timestamp": Clear10Loader._safe_bucket_timestamp(date_taken, bucket),
                        "source": "clear10",
                        "device": meta.get("DEVICE", "") if isinstance(meta, dict) else "",
                        "user_tags": meta.get("USER_TAGS", "") if isinstance(meta, dict) else "",
                        "lon": meta.get("LON", "") if isinstance(meta, dict) else "",
                        "lat": meta.get("LAT", "") if isinstance(meta, dict) else "",
                    }
                )
                n_taken += 1

        if not rows:
            raise ValueError(f"No samples were resolved for CLEAR-10 bucket {bucket}")

        return pd.DataFrame(rows)
