"""Data loader for drift analysis results."""

import json
import math
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

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """Convert values to finite floats, falling back to default when invalid."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(default)

        return numeric if math.isfinite(numeric) else float(default)
    
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

    def get_slice_analysis_results(self) -> pd.DataFrame:
        """
        Get flattened slice analysis results from pipeline metadata.

        Returns:
            DataFrame with columns:
                analysis_key, analysis_type, detector, slice_key, slice_key_label,
                reference_slice, test_slice, reference_slice_label, test_slice_label,
                drift_detected, severity, score,
                reference_samples, test_samples
        """
        if self.raw_data is None:
            self.load()

        rows = []
        yearly_data = self.raw_data.get("yearly_results", self.raw_data)

        for analysis_key, analysis_payload in yearly_data.items():
            # Skip non-dict payloads
            if not isinstance(analysis_payload, dict):
                continue

            analysis_type = analysis_payload.get("analysis_type", "temporal")
            pipelines = analysis_payload.get("pipelines", {})
            slice_value_labels = analysis_payload.get("slice_value_labels", {})

            for pipeline_name, pipeline_data in pipelines.items():
                metadata = pipeline_data.get("metadata", {})
                slice_analysis = metadata.get("slice_analysis", {})
                if not slice_analysis.get("enabled"):
                    continue

                detector_name = pipeline_data.get("detection", {}).get("detector_name", pipeline_name)
                slices = slice_analysis.get("slices", {})

                for slice_key, slice_payload in slices.items():
                    slice_result = slice_payload.get("result", {})
                    detection = slice_result.get("detection", {})
                    reference_slice = slice_payload.get("reference_slice_value")
                    test_slice = slice_payload.get("test_slice_value")
                    reference_slice_label = slice_value_labels.get(str(reference_slice), str(reference_slice))
                    test_slice_label = slice_value_labels.get(str(test_slice), str(test_slice))
                    slice_key_label = f"{reference_slice_label}->{test_slice_label}"

                    rows.append({
                        "analysis_key": analysis_key,
                        "analysis_type": analysis_type,
                        "detector": detector_name.replace("_", " ").title(),
                        "slice_key": slice_key,
                        "slice_key_label": slice_key_label,
                        "reference_slice": reference_slice,
                        "test_slice": test_slice,
                        "reference_slice_label": reference_slice_label,
                        "test_slice_label": test_slice_label,
                        "drift_detected": detection.get("drift_detected", False),
                        "severity": detection.get("severity", "none"),
                        "score": detection.get("score", 0.0),
                        "reference_samples": slice_payload.get("reference_samples", 0),
                        "test_samples": slice_payload.get("test_samples", 0),
                    })

        if not rows:
            return pd.DataFrame(
                columns=[
                    "analysis_key",
                    "analysis_type",
                    "detector",
                    "slice_key",
                    "slice_key_label",
                    "reference_slice",
                    "test_slice",
                    "reference_slice_label",
                    "test_slice_label",
                    "drift_detected",
                    "severity",
                    "score",
                    "reference_samples",
                    "test_samples",
                ]
            )

        return pd.DataFrame(rows)

    def get_clear10_baseline_performance(self) -> Dict:
        """
        Get baseline model performance for CLEAR-10 (bucket-1 test split).

        Returns:
            Dictionary with baseline metrics if available.
        """
        if self.raw_data is None:
            self.load()

        baseline = self.raw_data.get("baseline_performance", {})
        if isinstance(baseline, dict) and baseline:
            return baseline

        metadata = self.raw_data.get("metadata", {})
        baseline = metadata.get("baseline_performance", {})
        if isinstance(baseline, dict):
            return baseline

        return {}

    def get_clear10_proxy_metrics(self) -> pd.DataFrame:
        """
        Get CLEAR-10 proxy-vs-actual metrics by bucket.

        Expected normalized output columns:
            bucket, metric, estimated, actual
        """
        if self.raw_data is None:
            self.load()

        rows = []

        # Preferred contract: top-level proxy_metrics list
        proxy_metrics = self.raw_data.get("proxy_metrics", [])
        if isinstance(proxy_metrics, list):
            for entry in proxy_metrics:
                bucket = entry.get("bucket")
                metric = str(entry.get("metric", "")).lower()
                if bucket is None or not metric:
                    continue
                rows.append(
                    {
                        "bucket": int(bucket),
                        "metric": metric,
                        "estimated": self._safe_float(entry.get("estimated")),
                        "actual": self._safe_float(entry.get("actual")),
                    }
                )

        # Fallback contract: per-bucket records with nested estimated/actual maps
        if not rows:
            bucket_results = self.raw_data.get("bucket_results", {})
            if isinstance(bucket_results, dict):
                for bucket_key, payload in bucket_results.items():
                    if not str(bucket_key).isdigit() or int(bucket_key) < 2:
                        continue
                    proxy = payload.get("proxy_performance", {})
                    estimated = proxy.get("estimated", {})
                    actual = proxy.get("actual", {})
                    metric_keys = set(estimated.keys()) | set(actual.keys())
                    for metric in metric_keys:
                        rows.append(
                            {
                                "bucket": int(bucket_key),
                                "metric": str(metric).lower(),
                                "estimated": self._safe_float(estimated.get(metric)),
                                "actual": self._safe_float(actual.get(metric)),
                            }
                        )

        if not rows:
            return pd.DataFrame(columns=["bucket", "metric", "estimated", "actual"])

        return pd.DataFrame(rows).sort_values(["metric", "bucket"])

    def get_clear10_proxy_metrics_classwise(self) -> pd.DataFrame:
        """
        Get CLEAR-10 class-wise proxy metric trends by bucket.

        Expected normalized output columns:
            bucket, class_id, class_name, metric, estimated, actual, gap
        """
        if self.raw_data is None:
            self.load()

        rows = []

        # Preferred contract: top-level class-wise rows.
        classwise = self.raw_data.get("proxy_metrics_classwise", [])
        if isinstance(classwise, list):
            for entry in classwise:
                bucket = entry.get("bucket")
                class_id = entry.get("class_id")
                metric = entry.get("metric")
                if bucket is None or class_id is None or not metric:
                    continue
                rows.append(
                    {
                        "bucket": int(bucket),
                        "class_id": int(class_id),
                        "class_name": str(entry.get("class_name", f"class_{class_id}")),
                        "metric": str(metric).lower(),
                        "estimated": self._safe_float(entry.get("estimated")),
                        "actual": self._safe_float(entry.get("actual")),
                        "gap": self._safe_float(entry.get("gap")),
                    }
                )

        # Fallback: nested class-wise maps from per-bucket payloads.
        if not rows:
            bucket_results = self.raw_data.get("bucket_results", {})
            if isinstance(bucket_results, dict):
                for bucket_key, payload in bucket_results.items():
                    if not str(bucket_key).isdigit() or int(bucket_key) < 2:
                        continue
                    proxy = payload.get("proxy_performance", {})
                    estimated_map = proxy.get("class_wise_estimated", {})
                    actual_map = proxy.get("class_wise_actual", {})
                    gap_map = proxy.get("class_wise_proxy_quality_gap", {})

                    for class_key, est_values in estimated_map.items():
                        try:
                            class_id = int(str(class_key).split("_")[-1])
                        except ValueError:
                            continue

                        actual_values = actual_map.get(class_key, {})
                        class_gaps = gap_map.get(class_key, {})
                        for metric_name in ("precision", "recall", "f1"):
                            rows.append(
                                {
                                    "bucket": int(bucket_key),
                                    "class_id": class_id,
                                    "class_name": str(class_key),
                                    "metric": metric_name,
                                    "estimated": self._safe_float(est_values.get(metric_name)),
                                    "actual": self._safe_float(actual_values.get(metric_name)),
                                    "gap": self._safe_float(class_gaps.get(metric_name)),
                                }
                            )

        if not rows:
            return pd.DataFrame(
                columns=["bucket", "class_id", "class_name", "metric", "estimated", "actual", "gap"]
            )

        return pd.DataFrame(rows).sort_values(["class_id", "metric", "bucket"])

    def get_clear10_drift_timeline(self) -> pd.DataFrame:
        """
        Get CLEAR-10 detector trends by bucket.

        Expected normalized output columns:
            bucket, detector, score, threshold
        """
        if self.raw_data is None:
            self.load()

        rows = []

        # Preferred contract: top-level drift_results list
        drift_results = self.raw_data.get("drift_results", [])
        if isinstance(drift_results, list):
            for entry in drift_results:
                bucket = entry.get("bucket")
                detector = entry.get("detector")
                if bucket is None or detector is None:
                    continue
                rows.append(
                    {
                        "bucket": int(bucket),
                        "detector": str(detector),
                        "score": self._safe_float(entry.get("score")),
                        "threshold": entry.get("threshold"),
                    }
                )

        # Fallback: per-bucket detector map
        if not rows:
            bucket_results = self.raw_data.get("bucket_results", {})
            if isinstance(bucket_results, dict):
                for bucket_key, payload in bucket_results.items():
                    if not str(bucket_key).isdigit() or int(bucket_key) < 2:
                        continue
                    detectors = payload.get("detectors", {})
                    if not isinstance(detectors, dict):
                        continue
                    for detector_name, detector_payload in detectors.items():
                        rows.append(
                            {
                                "bucket": int(bucket_key),
                                "detector": str(detector_name).replace("_", " ").title(),
                                "score": self._safe_float(detector_payload.get("score")),
                                "threshold": detector_payload.get("threshold"),
                            }
                        )

        if not rows:
            return pd.DataFrame(columns=["bucket", "detector", "score", "threshold"])

        return pd.DataFrame(rows).sort_values(["detector", "bucket"])

    def get_clear10_localization_summary(self) -> pd.DataFrame:
        """
        Get CLEAR-10 localization summaries by bucket.

        Expected normalized output columns:
            bucket, top_features, n_drifted_features
        """
        if self.raw_data is None:
            self.load()

        rows = []

        bucket_results = self.raw_data.get("bucket_results", {})
        if isinstance(bucket_results, dict):
            for bucket_key, payload in bucket_results.items():
                if not str(bucket_key).isdigit() or int(bucket_key) < 2:
                    continue

                localization = payload.get("localization", {})
                drifted = localization.get("drifted_features", [])

                rows.append(
                    {
                        "bucket": int(bucket_key),
                        "top_features": ", ".join(drifted[:5]) if drifted else "-",
                        "n_drifted_features": len(drifted),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["bucket", "top_features", "n_drifted_features"])

        return pd.DataFrame(rows).sort_values("bucket")

    def get_clear10_rca_summary(self) -> pd.DataFrame:
        """
        Get CLEAR-10 RCA summaries by bucket.

        Expected normalized output columns:
            bucket, top_changes, n_recommendations
        """
        if self.raw_data is None:
            self.load()

        rows = []
        bucket_results = self.raw_data.get("bucket_results", {})

        if isinstance(bucket_results, dict):
            for bucket_key, payload in bucket_results.items():
                if not str(bucket_key).isdigit() or int(bucket_key) < 2:
                    continue

                rca = payload.get("rca", {})
                top_changes = rca.get("top_changes", [])
                recommendations = rca.get("recommendations", [])

                rows.append(
                    {
                        "bucket": int(bucket_key),
                        "top_changes": ", ".join(top_changes[:5]) if top_changes else "-",
                        "n_recommendations": len(recommendations),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["bucket", "top_changes", "n_recommendations"])

        return pd.DataFrame(rows).sort_values("bucket")
