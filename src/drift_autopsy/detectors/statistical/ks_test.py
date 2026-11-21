"""Kolmogorov-Smirnov test for drift detection."""

import numpy as np
from scipy import stats
from typing import Optional
import logging

from drift_autopsy.core.detector import BaseDriftDetector
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import DetectionResult, DriftSeverity
from drift_autopsy.registry import DetectorRegistry
from drift_autopsy.stats import adjusted_threshold as adjusted_threshold_for
from drift_autopsy.stats import reject, severity_from_p_value

logger = logging.getLogger(__name__)


@DetectorRegistry.register("ks_test")
class KSTest(BaseDriftDetector):
    """
    Kolmogorov-Smirnov test for detecting univariate distribution drift.
    
    The KS test is a non-parametric test that compares the empirical
    cumulative distribution functions of two samples. It works well for
    continuous features.
    
    Args:
        threshold: P-value threshold for detecting drift (default: 0.05)
        correction: Multiple testing correction method
            - None: No correction
            - "bonferroni": Bonferroni correction (default)
            - "holm": Holm-Bonferroni step-down correction
        aggregate_method: How to aggregate scores across features
            - "max": Largest per-feature KS statistic
            - "mean": Mean KS statistic
            - "vote": Proportion of features rejected under the correction

    Nominal features (``Dataset.categorical_features``) are skipped: their codes
    have no order, so a KS statistic on them is not meaningful. Use PSI or the
    univariate localizer's chi-square test for those.
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        correction: Optional[str] = "bonferroni",
        aggregate_method: str = "max",
    ):
        super().__init__(name="ks_test")
        if aggregate_method not in ("max", "mean", "vote"):
            raise ValueError(f"aggregate_method must be 'max', 'mean' or 'vote', got {aggregate_method!r}")
        if correction not in (None, "bonferroni", "holm"):
            raise ValueError(f"correction must be None, 'bonferroni' or 'holm', got {correction!r}")
        self.threshold = threshold
        self.correction = correction
        self.aggregate_method = aggregate_method
        self._feature_statistics = {}
    
    def fit(self, reference_data: Dataset) -> None:
        """
        Store reference data for comparison.
        
        Args:
            reference_data: Reference dataset
        """
        super().fit(reference_data)
        logger.info(f"KS Test fitted on {reference_data.n_samples} samples")
    
    def detect(self, test_data: Dataset) -> DetectionResult:
        """
        Detect drift using KS test for each numeric feature.
        
        Args:
            test_data: Test dataset
        
        Returns:
            DetectionResult with aggregated drift signal
        """
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before calling detect()")
        
        logger.info(f"Running KS test on {test_data.n_samples} test samples")
        
        # KS compares ordered distributions, so it is applied to numeric
        # (continuous or ordinal) features only; nominal codes have no order.
        ref_df = self._reference_data.to_pandas()
        test_df = test_data.to_pandas()
        categorical = set(self._reference_data.categorical_features or [])
        numeric_cols = [
            c for c in ref_df.select_dtypes(include=[np.number]).columns
            if c not in categorical
        ]
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric features found for KS test")
            return DetectionResult(
                detector_name=self.name,
                drift_detected=False,
                severity=DriftSeverity.NONE,
                score=0.0,
                threshold=self.threshold,
                metadata={"message": "No numeric features"}
            )
        
        # Run KS test for each feature
        statistics = []
        p_values = []
        feature_results = {}
        
        for col in numeric_cols:
            ref_values = ref_df[col].dropna().values
            test_values = test_df[col].dropna().values
            
            if len(ref_values) == 0 or len(test_values) == 0:
                logger.warning(f"Skipping feature '{col}' due to insufficient data")
                continue
            
            statistic, p_value = stats.ks_2samp(ref_values, test_values)
            statistics.append(statistic)
            p_values.append(p_value)
            
            feature_results[col] = {
                "statistic": float(statistic),
                "p_value": float(p_value),
            }
        
        if len(statistics) == 0:
            logger.warning("No features could be tested")
            return DetectionResult(
                detector_name=self.name,
                drift_detected=False,
                severity=DriftSeverity.NONE,
                score=0.0,
                threshold=self.threshold,
            )
        
        # Global decision: reject if the smallest p-value clears the corrected
        # threshold (Bonferroni and Holm coincide for this first step).
        adjusted_threshold = adjusted_threshold_for(self.threshold, len(p_values), self.correction)
        rejected = reject(p_values, self.threshold, self.correction)
        
        # Aggregate results
        if self.aggregate_method == "mean":
            aggregated_score = float(np.mean(statistics))
        elif self.aggregate_method == "vote":
            aggregated_score = float(np.mean(rejected))
        else:
            aggregated_score = float(np.max(statistics))
        aggregated_p = float(np.min(p_values))
        
        drift_detected = aggregated_p < adjusted_threshold
        severity = severity_from_p_value(aggregated_p, adjusted_threshold)
        
        logger.info(
            f"KS Test result: drift={drift_detected}, "
            f"score={aggregated_score:.4f}, p={aggregated_p:.4f}, "
            f"severity={severity.value}"
        )
        
        return DetectionResult(
            detector_name=self.name,
            drift_detected=drift_detected,
            severity=severity,
            score=aggregated_score,
            threshold=adjusted_threshold,
            p_value=aggregated_p,
            statistic=aggregated_score,
            metadata={
                "n_features_tested": len(statistics),
                "feature_results": feature_results,
                "correction": self.correction,
                "aggregate_method": self.aggregate_method,
                "n_features_rejected": int(np.sum(rejected)),
            }
        )
