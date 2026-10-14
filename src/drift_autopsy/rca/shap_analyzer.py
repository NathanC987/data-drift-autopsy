"""SHAP-based root cause analysis."""

import numpy as np
from typing import Optional, Any, Dict, List
import logging

# NumPy 2.0 compatibility shims
if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid

if not hasattr(np, 'in1d'):
    np.in1d = np.isin

from drift_autopsy.core.rca import BaseRootCauseAnalyzer
from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.result import LocalizationResult, RCAResult
from drift_autopsy.registry import RCARegistry

logger = logging.getLogger(__name__)


@RCARegistry.register("shap")
class SHAPAnalyzer(BaseRootCauseAnalyzer):
    """
    SHAP-based root cause analysis for drift.
    
    Computes SHAP values on reference and test data to understand:
    1. How feature importances have changed
    2. Which features contribute most to predictions
    3. How the model's reliance on features has shifted
    
    Requires:
        - A trained model that is compatible with SHAP
        - Reference and test data
    
    Args:
        n_background_samples: Reference rows forming the SHAP background (default: 100)
        n_test_samples: Rows explained from each of the reference and test data (default: 100)
        feature_subset: Optional list of features to focus on
        random_state: Seed for the row samples and the bootstrap (default: 0)
        n_bootstrap: Bootstrap resamples for the change intervals (default: 2000)
        alpha: Joint error level of the change intervals, Bonferroni-split over
            features (default: 0.05)

    The model is explained on the raw feature matrix it receives in deployment
    (all features, nominal codes included), so a pipeline that encodes or
    scales internally is explained as a whole.
    """
    
    def __init__(
        self,
        n_background_samples: int = 100,
        n_test_samples: int = 100,
        feature_subset: Optional[List[str]] = None,
        random_state: Optional[int] = 0,
        n_bootstrap: int = 2000,
        alpha: float = 0.05,
    ):
        super().__init__(name="shap")
        self.n_background_samples = n_background_samples
        self.n_test_samples = n_test_samples
        self.feature_subset = feature_subset
        self.random_state = random_state
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
    
    @staticmethod
    def _prediction_function(model: Any, X_sample: np.ndarray):
        """
        Pick the model output SHAP should explain.

        Hard class labels give near-zero, noisy attributions, so probabilistic
        models are explained through their probabilities: the positive class for
        binary models, every class for multiclass models.

        Returns:
            (callable, name of the explained output)
        """
        if hasattr(model, "predict_proba"):
            n_classes = np.asarray(model.predict_proba(X_sample[:1])).shape[-1]
            if n_classes == 2:
                return (lambda X: model.predict_proba(X)[:, 1]), "predict_proba[:,1]"
            return model.predict_proba, "predict_proba"
        return model.predict, "predict"

    def analyze(
        self,
        reference_data: Dataset,
        test_data: Dataset,
        localization: Optional[LocalizationResult] = None,
        model: Optional[Any] = None,
    ) -> RCAResult:
        """
        Analyze root causes using SHAP.
        
        Args:
            reference_data: Reference dataset
            test_data: Test dataset
            localization: Optional localization result
            model: Trained model for SHAP analysis
        
        Returns:
            RCAResult with SHAP-based explanations
        """
        if model is None:
            logger.warning("No model provided for SHAP analysis. Skipping.")
            return RCAResult(
                analyzer_name=self.name,
                explanations={"error": "Model required for SHAP analysis"},
                recommendations=["Provide a trained model to enable SHAP analysis"],
            )
        
        try:
            import shap
        except ImportError:
            logger.error("SHAP library not installed")
            return RCAResult(
                analyzer_name=self.name,
                explanations={"error": "SHAP library not installed"},
                recommendations=["Install SHAP: pip install shap"],
            )
        
        logger.info("Running SHAP analysis")
        
        # Sample data for efficiency
        ref_df = reference_data.to_pandas()
        test_df = test_data.to_pandas()
        
        # Explain every feature the model consumes, in the model's column order.
        feature_cols = list(reference_data.feature_names)
        non_numeric = [c for c in feature_cols if not np.issubdtype(ref_df[c].dtype, np.number)]
        if non_numeric:
            return RCAResult(
                analyzer_name=self.name,
                explanations={"error": f"SHAP needs numerically coded features; got {non_numeric}"},
            )
        
        # Determine which features to focus analysis on (for reporting/recommendations)
        # But we'll compute SHAP on ALL features so model gets correct input shape
        if self.feature_subset:
            features_to_focus = [f for f in self.feature_subset if f in feature_cols]
        elif localization and localization.drifted_features:
            # Focus analysis on drifted features
            features_to_focus = [f for f in localization.drifted_features if f in feature_cols]
        else:
            features_to_focus = feature_cols
        
        if len(features_to_focus) == 0:
            features_to_focus = feature_cols  # Fallback
        
        logger.info(f"Analyzing {len(features_to_focus)} features with SHAP")
        
        X_ref = ref_df[feature_cols].to_numpy(dtype=float)
        X_test = test_df[feature_cols].to_numpy(dtype=float)
        
        # Three disjoint-purpose samples: a background set from the reference
        # (the SHAP baseline distribution), and n_test explained rows from each
        # of the reference and test data. Reference rows used as background are
        # not explained, so the two explained sets are exchangeable under no drift.
        rng = np.random.default_rng(self.random_state)
        n_bg = min(self.n_background_samples, len(X_ref))
        ref_perm = rng.permutation(len(X_ref))
        bg_idx = ref_perm[:n_bg]
        remaining = ref_perm[n_bg:] if len(ref_perm) > n_bg else ref_perm
        ref_idx = remaining[: min(self.n_test_samples, len(remaining))]
        test_idx = rng.choice(len(X_test), min(self.n_test_samples, len(X_test)), replace=False)
        n_ref, n_test = len(ref_idx), len(test_idx)
        
        X_background = X_ref[bg_idx]
        X_ref_sample = X_ref[ref_idx]
        X_test_sample = X_test[test_idx]
        
        try:
            predict_fn, explained_output = self._prediction_function(model, X_background)
            explainer = shap.Explainer(predict_fn, X_background)
            
            logger.debug("Computing SHAP values for reference data")
            shap_values_ref = explainer(X_ref_sample, silent=True)
            
            # Compute SHAP values for test
            logger.debug("Computing SHAP values for test data")
            shap_values_test = explainer(X_test_sample, silent=True)
            
            # Extract values (handle different SHAP return types)
            if hasattr(shap_values_ref, 'values'):
                shap_ref = shap_values_ref.values
                shap_test = shap_values_test.values
            else:
                shap_ref = shap_values_ref
                shap_test = shap_values_test
            
            # Handle multi-class (take absolute mean across classes)
            if len(shap_ref.shape) == 3:
                shap_ref = np.abs(shap_ref).mean(axis=2)
                shap_test = np.abs(shap_test).mean(axis=2)
            
            # Feature importance = mean |SHAP| per feature.
            abs_ref = np.abs(shap_ref)
            abs_test = np.abs(shap_test)
            ref_importance = abs_ref.mean(axis=0)
            test_importance = abs_test.mean(axis=0)
            importance_shift = test_importance - ref_importance

            # Percentile bootstrap CI for each change, resampling explained rows
            # within each sample. The CI level is Bonferroni-adjusted over the
            # features so all intervals hold jointly at 1 - alpha.
            m = len(feature_cols)
            boot_rng = np.random.default_rng(self.random_state)
            boot = np.empty((self.n_bootstrap, m))
            for k in range(self.n_bootstrap):
                ir = boot_rng.integers(0, len(abs_ref), len(abs_ref))
                it = boot_rng.integers(0, len(abs_test), len(abs_test))
                boot[k] = abs_test[it].mean(axis=0) - abs_ref[ir].mean(axis=0)
            tail = 100.0 * self.alpha / (2 * m)
            ci_low, ci_high = np.percentile(boot, [tail, 100.0 - tail], axis=0)
            significant = (ci_low > 0) | (ci_high < 0)
            
            feature_importances = {}
            importance_changes = {}
            for i, feature_name in enumerate(feature_cols):
                feature_importances[feature_name] = {
                    "ref_importance": float(ref_importance[i]),
                    "test_importance": float(test_importance[i]),
                    "change": float(importance_shift[i]),
                    "change_ci_low": float(ci_low[i]),
                    "change_ci_high": float(ci_high[i]),
                    "change_significant": bool(significant[i]),
                }
                importance_changes[feature_name] = float(importance_shift[i])
            
            # Order: focus features (drifted ones) first, each group by |change|.
            focus_changes = {f: importance_changes[f] for f in features_to_focus if f in importance_changes}
            sorted_focus = sorted(focus_changes.items(), key=lambda x: abs(x[1]), reverse=True)
            other_changes = {f: c for f, c in importance_changes.items() if f not in features_to_focus}
            sorted_other = sorted(other_changes.items(), key=lambda x: abs(x[1]), reverse=True)
            sorted_changes = sorted_focus + sorted_other
            
            # Recommendations only cite changes whose joint CI excludes zero.
            sig = {f for f in feature_cols if feature_importances[f]["change_significant"]}
            recommendations = []
            increased = [f for f, c in sorted_focus if c > 0 and f in sig]
            decreased = [f for f, c in sorted_focus if c < 0 and f in sig]
            if increased:
                recommendations.append(f"Significantly increased model reliance: {', '.join(increased)}")
            if decreased:
                recommendations.append(f"Significantly decreased model reliance: {', '.join(decreased)}")
            if localization and localization.drifted_features:
                both = [f for f in localization.drifted_features if f in sig]
                if both:
                    recommendations.append(
                        f"Drifted and with a significant attribution change: {', '.join(both)} "
                        "- likely contributors"
                    )
            if not recommendations:
                recommendations.append(
                    "No attribution change is significant at the joint "
                    f"{100 * (1 - self.alpha):.0f}% level: the model uses its features as before"
                )
            
            logger.info("SHAP analysis complete")
            
            return RCAResult(
                analyzer_name=self.name,
                explanations={
                    "method": "SHAP feature importance comparison",
                    "n_background_samples": n_bg,
                    "n_reference_samples": n_ref,
                    "n_test_samples": n_test,
                    "top_importance_changes": sorted_changes[:10],
                },
                feature_importances=importance_changes,
                distribution_changes=feature_importances,
                recommendations=recommendations,
                metadata={
                    "explained_output": explained_output,
                    "n_bootstrap": self.n_bootstrap,
                    "alpha": self.alpha,
                    "n_significant_changes": int(significant.sum()),
                    "n_features_analyzed": len(feature_cols),
                    "n_features_focused": len(features_to_focus),
                    "features_focused": features_to_focus,
                }
            )
        
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return RCAResult(
                analyzer_name=self.name,
                explanations={"error": f"SHAP analysis failed: {str(e)}"},
                recommendations=["Check model compatibility with SHAP", "Verify data format"],
            )
