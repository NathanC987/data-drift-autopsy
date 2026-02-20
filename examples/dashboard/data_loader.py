"""Data loader for drift analysis results."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class DriftResultsLoader:
    """Load and parse drift detection results from JSON."""
    
    def __init__(self, results_path: str):
        """
        Initialize loader with results file path.
        
        Args:
            results_path: Path to JSON results file
        """
        self.results_path = Path(results_path)
        self.raw_data: Optional[Dict] = None
        
    def load(self) -> Dict:
        """
        Load results from JSON file.
        
        Returns:
            Raw results dictionary
        """
        with open(self.results_path, 'r') as f:
            self.raw_data = json.load(f)
        return self.raw_data
    
    def get_detector_timeline(self, detector_name: str) -> pd.DataFrame:
        """
        Get timeline data for a specific detector.
        
        Args:
            detector_name: Name of detector (e.g., "ks_test", "psi", "mmd", "cbpe")
        
        Returns:
            DataFrame with columns: year, drift_detected, severity, score, p_value
        """
        if self.raw_data is None:
            self.load()
        
        timeline_data = []
        
        # Handle both "yearly_results" format and direct year keys
        yearly_data = self.raw_data.get("yearly_results", self.raw_data)
        
        for year, year_data in yearly_data.items():
            # Skip non-year keys
            if not year.isdigit():
                continue
            
            # Check both "detectors" dict and "pipelines" dict
            detector_results = year_data.get("detectors", {})
            if not detector_results:
                # Try pipeline format
                pipelines = year_data.get("pipelines", {})
                for pipeline_name, pipeline_data in pipelines.items():
                    detection = pipeline_data.get("detection", {})
                    if detection.get("detector_name") == detector_name:
                        detector_results[detector_name] = detection
                        break
            
            detector_result = detector_results.get(detector_name)
            
            if detector_result:
                timeline_data.append({
                    "year": int(year),
                    "drift_detected": detector_result.get("drift_detected", False),
                    "severity": detector_result.get("severity", "none"),
                    "score": detector_result.get("score", 0.0),
                    "p_value": detector_result.get("p_value"),
                    "threshold": detector_result.get("threshold"),
                })
        
        if not timeline_data:
            return pd.DataFrame(columns=["year", "drift_detected", "severity", "score", "p_value", "threshold"])
        
        return pd.DataFrame(timeline_data).sort_values("year")
    
    def get_all_detectors_timeline(self) -> pd.DataFrame:
        """
        Get timeline data for all detectors combined.
        
        Returns:
            DataFrame with columns: year, detector, drift_detected, severity, score
        """
        if self.raw_data is None:
            self.load()
        
        timeline_data = []
        
        # Handle both "yearly_results" format and direct year keys
        yearly_data = self.raw_data.get("yearly_results", self.raw_data)
        
        for year, year_data in yearly_data.items():
            # Skip non-year keys
            if not year.isdigit():
                continue
            
            # Check "detectors" dict first
            detector_results = year_data.get("detectors", {})
            
            # If not found, try "pipelines" format
            if not detector_results:
                pipelines = year_data.get("pipelines", {})
                for pipeline_name, pipeline_data in pipelines.items():
                    detection = pipeline_data.get("detection", {})
                    detector_name = detection.get("detector_name")
                    if detector_name:
                        timeline_data.append({
                            "year": int(year),
                            "detector": detector_name.replace("_", " ").title(),
                            "drift_detected": detection.get("drift_detected", False),
                            "severity": detection.get("severity", "none"),
                            "score": detection.get("score", 0.0),
                        })
            else:
                for detector_name, detector_result in detector_results.items():
                    timeline_data.append({
                        "year": int(year),
                        "detector": detector_name.replace("_", " ").title(),
                        "drift_detected": detector_result.get("drift_detected", False),
                        "severity": detector_result.get("severity", "none"),
                        "score": detector_result.get("score", 0.0),
                    })
        
        if not timeline_data:
            return pd.DataFrame(columns=["year", "detector", "drift_detected", "severity", "score"])
        
        return pd.DataFrame(timeline_data).sort_values(["year", "detector"])
    
    def get_feature_drift_timeline(self) -> pd.DataFrame:
        """
        Get feature-level drift over time.
        
        Returns:
            DataFrame with columns: year, feature, drift_score, drift_detected
        """
        if self.raw_data is None:
            self.load()
        
        feature_data = []
        
        # Handle both formats
        yearly_data = self.raw_data.get("yearly_results", self.raw_data)
        
        for year, year_data in yearly_data.items():
            # Skip non-year keys
            if not year.isdigit():
                continue
            
            # Try direct localization first
            localization = year_data.get("localization")
            
            # If not found, check pipelines
            if not localization:
                pipelines = year_data.get("pipelines", {})
                for pipeline_data in pipelines.values():
                    if "localization" in pipeline_data:
                        localization = pipeline_data["localization"]
                        break
            
            if localization and localization.get("feature_drifts"):
                for feature_drift in localization["feature_drifts"]:
                    feature_data.append({
                        "year": int(year),
                        "feature": feature_drift["feature_name"],
                        "drift_score": feature_drift["score"],
                        "drift_detected": feature_drift["drift_detected"],
                        "severity": feature_drift.get("severity", "none"),
                    })
        
        if not feature_data:
            return pd.DataFrame(columns=["year", "feature", "drift_score", "drift_detected", "severity"])
        
        return pd.DataFrame(feature_data)
    
    def get_performance_metrics(self) -> pd.DataFrame:
        """
        Get model performance metrics over time.
        
        Returns:
            DataFrame with columns: year, accuracy, accuracy_delta
        """
        if self.raw_data is None:
            self.load()
        
        perf_data = []
        
        # Handle both formats
        yearly_data = self.raw_data.get("yearly_results", self.raw_data)
        
        for year, year_data in yearly_data.items():
            # Skip non-year keys
            if not year.isdigit():
                continue
            
            # Try metadata first
            metadata = year_data.get("metadata", {})
            
            # If not in metadata, check direct keys
            accuracy = metadata.get("test_accuracy") or year_data.get("actual_accuracy", 0.0)
            accuracy_delta = metadata.get("accuracy_delta") or year_data.get("accuracy_drop", 0.0)
            
            perf_data.append({
                "year": int(year),
                "accuracy": accuracy,
                "accuracy_delta": accuracy_delta,
            })
        
        if not perf_data:
            return pd.DataFrame(columns=["year", "accuracy", "accuracy_delta"])
        
        return pd.DataFrame(perf_data).sort_values("year")
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics across all years.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.raw_data is None:
            self.load()
        
        all_detectors_df = self.get_all_detectors_timeline()
        perf_df = self.get_performance_metrics()
        feature_df = self.get_feature_drift_timeline()
        
        # Count years (they are at top level of JSON, not under "yearly_results")
        yearly_data = self.raw_data.get("yearly_results", self.raw_data)
        total_years = len([k for k in yearly_data.keys() if k.isdigit()])
        
        return {
            "total_years": total_years,
            "detectors_count": all_detectors_df["detector"].nunique(),
            "total_drift_events": all_detectors_df["drift_detected"].sum(),
            "avg_accuracy": perf_df["accuracy"].mean() if not perf_df.empty else 0.0,
            "accuracy_range": (
                perf_df["accuracy"].min(), 
                perf_df["accuracy"].max()
            ) if not perf_df.empty else (0.0, 0.0),
            "unique_drifted_features": feature_df[feature_df["drift_detected"]]["feature"].nunique() if not feature_df.empty else 0,
        }
    
    def get_available_years(self) -> List[int]:
        """Get list of available years in results."""
        if self.raw_data is None:
            self.load()
        
        yearly_data = self.raw_data.get("yearly_results", self.raw_data)
        return sorted([int(year) for year in yearly_data.keys() if year.isdigit()])
    
    def get_available_detectors(self) -> List[str]:
        """Get list of available detector names."""
        if self.raw_data is None:
            self.load()
        
        detectors = set()
        yearly_data = self.raw_data.get("yearly_results", self.raw_data)
        
        for year, year_data in yearly_data.items():
            if not year.isdigit():
                continue
            
            # Check detectors dict
            detectors.update(year_data.get("detectors", {}).keys())
            
            # Check pipelines
            pipelines = year_data.get("pipelines", {})
            for pipeline_data in pipelines.values():
                detection = pipeline_data.get("detection", {})
                detector_name = detection.get("detector_name")
                if detector_name:
                    detectors.add(detector_name)
        
        return sorted(list(detectors))
    
    def get_rca_results(self) -> pd.DataFrame:
        """
        Get root cause analysis results over time.
        
        Returns:
            DataFrame with columns: year, detector, feature_importances, recommendations
        """
        if self.raw_data is None:
            self.load()
        
        rca_data = []
        yearly_data = self.raw_data.get("yearly_results", self.raw_data)
        
        for year, year_data in yearly_data.items():
            if not year.isdigit():
                continue
            
            # Check pipelines for RCA results
            pipelines = year_data.get("pipelines", {})
            for pipeline_name, pipeline_data in pipelines.items():
                rca = pipeline_data.get("rca")
                if rca:
                    detection = pipeline_data.get("detection", {})
                    detector_name = detection.get("detector_name", "unknown")
                    
                    rca_data.append({
                        "year": int(year),
                        "detector": detector_name,
                        "analyzer": rca.get("analyzer_name", "unknown"),
                        "feature_importances": rca.get("feature_importances", {}),
                        "recommendations": rca.get("recommendations", []),
                        "n_recommendations": len(rca.get("recommendations", [])),
                    })
        
        if not rca_data:
            return pd.DataFrame(columns=["year", "detector", "analyzer", "feature_importances", "recommendations", "n_recommendations"])
        
        return pd.DataFrame(rca_data)
    
    def get_feature_importance_changes(self) -> pd.DataFrame:
        """
        Get feature importance changes from RCA over time.
        
        Returns:
            DataFrame with columns: year, feature, ref_importance, test_importance, change
        """
        if self.raw_data is None:
            self.load()
        
        importance_data = []
        yearly_data = self.raw_data.get("yearly_results", self.raw_data)
        
        for year, year_data in yearly_data.items():
            if not year.isdigit():
                continue
            
            pipelines = year_data.get("pipelines", {})
            for pipeline_data in pipelines.values():
                rca = pipeline_data.get("rca")
                if rca and rca.get("distribution_changes"):
                    # distribution_changes has the nested structure with ref/test values
                    distribution_changes = rca["distribution_changes"]
                    
                    # Extract feature-level importance data
                    for feature, feature_data in distribution_changes.items():
                        if isinstance(feature_data, dict):
                            ref_imp = feature_data.get("ref_importance", 0.0)
                            test_imp = feature_data.get("test_importance", 0.0)
                            change = feature_data.get("change", test_imp - ref_imp)
                            
                            importance_data.append({
                                "year": int(year),
                                "feature": feature,
                                "ref_importance": ref_imp,
                                "test_importance": test_imp,
                                "change": change,
                                "abs_change": abs(change),
                            })
        
        if not importance_data:
            return pd.DataFrame(columns=["year", "feature", "ref_importance", "test_importance", "change", "abs_change"])
        
        return pd.DataFrame(importance_data)
