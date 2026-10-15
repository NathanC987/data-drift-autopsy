"""Univariate drift localization."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, List, Dict, Any
import logging

from drift_autopsy.core.localizer import BaseDriftLocalizer
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import DetectionResult, LocalizationResult, FeatureDrift, DriftSeverity
from drift_autopsy.registry import LocalizerRegistry
from drift_autopsy.stats import chi2_homogeneity, reject, severity_from_p_value

logger = logging.getLogger(__name__)


@LocalizerRegistry.register("univariate")
class UnivariateLocalizer(BaseDriftLocalizer):
    """
    Univariate drift localization using per-feature statistical tests.
    
    Runs a two-sample KS test on each numeric (continuous or ordinal) feature
    and a two-sample chi-square test of homogeneity on each nominal feature
    (``Dataset.categorical_features``), then controls the family-wise error
    rate across features. Feature scores are effect sizes in [0, 1]: the KS
    statistic or Cramer's V.
    
    Args:
        threshold: P-value threshold for feature-level drift (default: 0.05)
        correction: Multiple testing correction ("bonferroni", "holm", or None)
        top_k: Return only top k drifted features (default: None = all)
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        correction: Optional[str] = "bonferroni",
        top_k: Optional[int] = None,
    ):
        super().__init__(name="univariate")
        if correction not in (None, "bonferroni", "holm"):
            raise ValueError(f"correction must be None, 'bonferroni' or 'holm', got {correction!r}")
        self.threshold = threshold
        self.correction = correction
        self.top_k = top_k
    
    def _test_numeric_feature(
        self,
        ref_values: np.ndarray,
        test_values: np.ndarray,
        feature_name: str,
    ) -> tuple:
        """
        Test numeric feature using KS test.
        
        Returns:
            (statistic, p_value, distribution_info)
        """
        statistic, p_value = stats.ks_2samp(ref_values, test_values)
        
        # Compute distribution statistics
        dist_info = {
            "ref_mean": float(np.mean(ref_values)),
            "ref_std": float(np.std(ref_values)),
            "test_mean": float(np.mean(test_values)),
            "test_std": float(np.std(test_values)),
            "mean_shift": float(np.mean(test_values) - np.mean(ref_values)),
            "std_shift": float(np.std(test_values) - np.std(ref_values)),
        }
        
        return float(statistic), float(p_value), dist_info
    
    def _test_categorical_feature(
        self,
        ref_values: pd.Series,
        test_values: pd.Series,
        feature_name: str,
    ) -> tuple:
        """
        Test a nominal feature with a two-sample chi-square test of homogeneity.

        Categories with an expected count below five are pooled (Cochran's rule).
        
        The returned effect size is Cramer's V, which lies in [0, 1] like the
        KS statistic used for numeric features, so feature scores are comparable.
        The chi-square statistic itself is kept in the distribution info.

        Returns:
            (cramers_v, p_value, distribution_info)
        """
        result = chi2_homogeneity(ref_values.to_numpy(), test_values.to_numpy())
        seen = set(ref_values.unique().tolist())
        unseen_mask = ~test_values.isin(seen)
        ref_mode = ref_values.mode()
        test_mode = test_values.mode()
        dist_info = {
            "n_categories": int(result["n_categories"]),
            "n_tested_cells": int(result["n_cells"]),
            "n_categories_pooled": int(result["n_pooled"]),
            "dof": int(result["dof"]),
            "chi2_statistic": float(result["statistic"]),
            "unseen_category_share": float(unseen_mask.mean()) if len(test_values) else 0.0,
            "n_unseen_categories": int(test_values[unseen_mask].nunique()),
            "ref_mode": ref_mode.iloc[0] if len(ref_mode) else None,
            "test_mode": test_mode.iloc[0] if len(test_mode) else None,
        }
        return float(result["cramers_v"]), float(result["p_value"]), dist_info
    
    def localize(
        self,
        reference_data: Dataset,
        test_data: Dataset,
        drift_signal: Optional[DetectionResult] = None,
    ) -> LocalizationResult:
        """
        Localize drift to individual features.
        
        Args:
            reference_data: Reference dataset
            test_data: Test dataset
            drift_signal: Optional upstream drift detection result
        
        Returns:
            LocalizationResult with per-feature drift information
        """
        logger.info("Running univariate localization")
        
        ref_df = reference_data.to_pandas()
        test_df = test_data.to_pandas()
        
        feature_drifts = []
        
        categorical = set(reference_data.categorical_features or []) | set(
            ref_df.select_dtypes(include=["object", "category", "bool"]).columns
        )
        numeric_cols = [
            c for c in ref_df.select_dtypes(include=[np.number]).columns if c not in categorical
        ]
        categorical_cols = [c for c in reference_data.feature_names if c in categorical]

        # Test numeric (continuous or ordinal) features with KS
        
        for col in numeric_cols:
            if col not in test_df.columns:
                logger.warning(f"Feature '{col}' not in test data, skipping")
                continue
            
            ref_values = ref_df[col].dropna().values
            test_values = test_df[col].dropna().values
            
            if len(ref_values) == 0 or len(test_values) == 0:
                logger.warning(f"Skipping '{col}' due to insufficient data")
                continue
            
            statistic, p_value, dist_info = self._test_numeric_feature(
                ref_values, test_values, col
            )
            
            feature_drifts.append({
                "name": col,
                "type": "numeric",
                "statistic": statistic,
                "p_value": p_value,
                "dist_info": dist_info,
            })
        
        # Test nominal features with the chi-square homogeneity test
        for col in categorical_cols:
            if col not in test_df.columns:
                continue
            
            ref_values = ref_df[col].dropna()
            test_values = test_df[col].dropna()
            
            if len(ref_values) == 0 or len(test_values) == 0:
                continue
            
            statistic, p_value, dist_info = self._test_categorical_feature(
                ref_values, test_values, col
            )
            
            feature_drifts.append({
                "name": col,
                "type": "categorical",
                "statistic": statistic,
                "p_value": p_value,
                "dist_info": dist_info,
            })
        
        # Apply multiple testing correction
        if len(feature_drifts) == 0:
            logger.warning("No features could be tested")
            return LocalizationResult(
                method_name=self.name,
                feature_drifts=[],
            )
        
        p_values = np.array([fd["p_value"] for fd in feature_drifts])
        m = len(p_values)
        rejected = reject(p_values, self.threshold, self.correction)

        # Threshold each feature was compared against: alpha/m (Bonferroni),
        # alpha/(m - rank) (Holm step-down) or alpha (no correction).
        if self.correction == "bonferroni":
            feature_thresholds = np.full(m, self.threshold / m)
        elif self.correction == "holm":
            ranks = np.empty(m, dtype=int)
            ranks[np.argsort(p_values, kind="stable")] = np.arange(m)
            feature_thresholds = self.threshold / (m - ranks)
        else:
            feature_thresholds = np.full(m, self.threshold)
        adjusted_threshold = float(feature_thresholds.min())
        
        feature_drift_objects = []
        for i, fd in enumerate(feature_drifts):
            drift_detected = bool(rejected[i])
            severity = (
                severity_from_p_value(fd["p_value"], float(feature_thresholds[i]))
                if drift_detected else DriftSeverity.NONE
            )
            feature_drift_objects.append(
                FeatureDrift(
                    feature_name=fd["name"],
                    drift_detected=drift_detected,
                    score=fd["statistic"],
                    p_value=fd["p_value"],
                    severity=severity,
                    distribution_shift={**fd["dist_info"], "test": fd["type"]},
                )
            )
        
        # Sort by p-value (most drifted first)
        feature_drift_objects.sort(key=lambda x: x.p_value)
        
        # Limit to top_k if specified
        if self.top_k is not None:
            feature_drift_objects = feature_drift_objects[:self.top_k]
        
        n_drifted = sum(1 for fd in feature_drift_objects if fd.drift_detected)
        
        logger.info(
            f"Localization complete: {n_drifted}/{len(feature_drift_objects)} features drifted"
        )
        
        return LocalizationResult(
            method_name=self.name,
            feature_drifts=feature_drift_objects,
            metadata={
                "threshold": adjusted_threshold,
                "correction": self.correction,
                "n_tested": len(feature_drifts),
                "n_drifted": n_drifted,
            }
        )
