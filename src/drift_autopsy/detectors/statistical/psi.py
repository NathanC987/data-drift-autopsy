"""Population Stability Index (PSI) for drift detection."""

import numpy as np
import pandas as pd
from typing import Optional
import logging

from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import DetectionResult, DriftSeverity
from drift_autopsy.registry import DetectorRegistry
from drift_autopsy.stats import MIN_EXPECTED_COUNT, severity_from_ratio

logger = logging.getLogger(__name__)


@DetectorRegistry.register("psi")
class PSI(BaseDriftDetector):
    """
    Population Stability Index for detecting distribution drift.
    
    PSI measures the shift in distributions by binning data and comparing
    the proportions in each bin. It works for both continuous and categorical features.
    
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    
    Interpretation:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change
    - PSI >= 0.2: Significant change
    
    Args:
        threshold: PSI threshold for detecting drift (default: 0.2)
        n_bins: Number of bins for continuous features (default: 10)
        aggregate_method: How to aggregate PSI across features ("max" or "mean")
    """
    
    def __init__(
        self,
        threshold: float = 0.2,
        n_bins: int = 10,
        aggregate_method: str = "max",
    ):
        super().__init__(name="psi")
        if aggregate_method not in ("max", "mean"):
            raise ValueError(f"aggregate_method must be 'max' or 'mean', got {aggregate_method!r}")
        self.threshold = threshold
        self.n_bins = n_bins
        self.aggregate_method = aggregate_method
        self._bin_edges = {}
        self._categorical_cols = []
    
    def fit(self, reference_data: Dataset) -> None:
        """
        Learn the binning scheme from reference data.

        Numeric features get reference quantile bins whose outer edges are
        -inf and +inf, so production values outside the reference range are
        counted in the end bins instead of being dropped. Categorical features
        are compared category by category.
        
        Args:
            reference_data: Reference dataset
        """
        super().fit(reference_data)
        self._bin_edges = {}
        
        ref_df = self._reference_data.to_pandas()
        categorical = set(reference_data.categorical_features or [])
        numeric_cols = [
            c for c in ref_df.select_dtypes(include=[np.number]).columns if c not in categorical
        ]
        self._categorical_cols = [c for c in reference_data.feature_names if c in categorical]
        
        for col in numeric_cols:
            values = ref_df[col].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            inner = np.unique(np.quantile(values, np.linspace(0, 1, self.n_bins + 1))[1:-1])
            self._bin_edges[col] = np.concatenate([[-np.inf], inner, [np.inf]])
        
        logger.info(
            f"PSI fitted on {reference_data.n_samples} samples with {len(self._bin_edges)} "
            f"binned and {len(self._categorical_cols)} categorical features"
        )
    
    def _calculate_psi(self, expected_counts: np.ndarray, actual_counts: np.ndarray) -> float:
        """
        PSI between two count vectors over the same cells.

        Counts get a Jeffreys pseudo-count of 0.5 per cell so empty cells give
        a finite value without the mass distortion of a fixed proportion floor.
        """
        expected = (np.asarray(expected_counts, dtype=float) + 0.5)
        actual = (np.asarray(actual_counts, dtype=float) + 0.5)
        expected = expected / expected.sum()
        actual = actual / actual.sum()
        return float(np.sum((actual - expected) * np.log(actual / expected)))

    @staticmethod
    def _pooled_category_counts(ref_values: pd.Series, test_values: pd.Series):
        """Category counts with categories whose expected count is below five pooled."""
        categories = sorted(set(ref_values.unique()) | set(test_values.unique()), key=str)
        ref_counts = ref_values.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
        test_counts = test_values.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
        total = ref_counts.sum() + test_counts.sum()
        expected = np.outer([ref_counts.sum(), test_counts.sum()], ref_counts + test_counts) / total
        keep = (expected >= MIN_EXPECTED_COUNT).all(axis=0)
        if (~keep).any():
            ref_counts = np.append(ref_counts[keep], ref_counts[~keep].sum())
            test_counts = np.append(test_counts[keep], test_counts[~keep].sum())
        return ref_counts, test_counts, int((~keep).sum())
    
    def detect(self, test_data: Dataset) -> DetectionResult:
        """
        Detect drift using PSI for each feature.
        
        Args:
            test_data: Test dataset
        
        Returns:
            DetectionResult with aggregated PSI
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        
        logger.info(f"Running PSI on {test_data.n_samples} test samples")
        
        ref_df = self._reference_data.to_pandas()
        test_df = test_data.to_pandas()
        
        psi_values = []
        feature_results = {}
        
        for col, bin_edges in self._bin_edges.items():
            if col not in test_df.columns:
                logger.warning(f"Feature '{col}' not in test data, skipping")
                continue
            ref_values = ref_df[col].dropna().to_numpy(dtype=float)
            test_values = test_df[col].dropna().to_numpy(dtype=float)
            if len(ref_values) == 0 or len(test_values) == 0:
                logger.warning(f"Skipping feature '{col}' due to insufficient data")
                continue
            ref_counts, _ = np.histogram(ref_values, bins=bin_edges)
            test_counts, _ = np.histogram(test_values, bins=bin_edges)
            psi = self._calculate_psi(ref_counts, test_counts)
            psi_values.append(psi)
            feature_results[col] = {"psi": float(psi), "n_bins": len(bin_edges) - 1}
        
        for col in self._categorical_cols:
            if col not in test_df.columns:
                continue
            ref_values = ref_df[col].dropna()
            test_values = test_df[col].dropna()
            if len(ref_values) == 0 or len(test_values) == 0:
                continue
            ref_counts, test_counts, n_pooled = self._pooled_category_counts(ref_values, test_values)
            psi = self._calculate_psi(ref_counts, test_counts)
            psi_values.append(psi)
            feature_results[col] = {
                "psi": float(psi),
                "n_categories": int(len(ref_counts)),
                "n_categories_pooled": n_pooled,
            }
        
        if len(psi_values) == 0:
            logger.warning("No features could be tested")
            return DetectionResult(
                detector_name=self.name,
                drift_detected=False,
                severity=DriftSeverity.NONE,
                score=0.0,
                threshold=self.threshold,
            )
        
        # Aggregate PSI values
        aggregated_psi = float(np.mean(psi_values) if self.aggregate_method == "mean" else np.max(psi_values))
        
        # Determine drift and severity
        drift_detected = aggregated_psi >= self.threshold
        
        severity = severity_from_ratio(aggregated_psi, self.threshold)
        
        logger.info(
            f"PSI result: drift={drift_detected}, "
            f"psi={aggregated_psi:.4f}, severity={severity.value}"
        )
        
        return DetectionResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            severity=severity,
            score=aggregated_psi,
            threshold=self.threshold,
            metadata={
                "n_features_tested": len(psi_values),
                "feature_results": feature_results,
                "aggregate_method": self.aggregate_method,
            }
        )
