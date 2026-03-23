"""Pipeline orchestration for drift analysis."""

import time
from typing import Optional, Union, Any, Dict
import logging

import numpy as np
import pandas as pd

from drift_autopsy.core.dataset import Dataset
from drift_autopsy.core.detector import DriftDetector
from drift_autopsy.core.localizer import DriftLocalizer
from drift_autopsy.core.rca import RootCauseAnalyzer
from drift_autopsy.core.result import PipelineResult, DetectionResult, LocalizationResult, RCAResult
from drift_autopsy.registry import DetectorRegistry, LocalizerRegistry, RCARegistry
from drift_autopsy.data.validators import DataValidator

logger = logging.getLogger(__name__)


class DriftPipeline:
    """
    Composable pipeline for drift detection, localization, and RCA.
    
    Orchestrates the full drift analysis workflow with conditional execution
    and proper error handling.
    
    Args:
        detector: Drift detector instance or name (for registry)
        localizer: Optional localizer instance or name
        rca: Optional RCA analyzer instance or name
        enable_localization: Enable localization step (default: True)
        enable_rca: Enable RCA step (default: False)
        validate_data: Validate input data (default: True)
        model: Optional model for RCA (e.g., for SHAP)
    
    Example:
        >>> from drift_autopsy import DriftPipeline
        >>> from drift_autopsy.detectors import KSTest
        >>> 
        >>> pipeline = DriftPipeline(
        ...     detector=KSTest(threshold=0.05),
        ...     localizer="univariate",
        ...     enable_rca=False
        ... )
        >>> result = pipeline.run(reference_data, test_data)
    """
    
    def __init__(
        self,
        detector: Union[DriftDetector, str],
        localizer: Optional[Union[DriftLocalizer, str]] = None,
        rca: Optional[Union[RootCauseAnalyzer, str]] = None,
        enable_localization: bool = True,
        enable_rca: bool = False,
        validate_data: bool = True,
        model: Optional[Any] = None,
        detector_params: Optional[Dict[str, Any]] = None,
        localizer_params: Optional[Dict[str, Any]] = None,
        rca_params: Optional[Dict[str, Any]] = None,
    ):
        detector_params = detector_params or {}
        localizer_params = localizer_params or {}
        rca_params = rca_params or {}

        # Setup detector
        if isinstance(detector, str):
            self.detector = DetectorRegistry.create(detector, **detector_params)
            logger.info(f"Created detector from registry: {detector} with params={detector_params}")
        else:
            self.detector = detector
        
        # Setup localizer
        self.enable_localization = enable_localization and localizer is not None
        if self.enable_localization:
            if isinstance(localizer, str):
                self.localizer = LocalizerRegistry.create(localizer, **localizer_params)
                logger.info(f"Created localizer from registry: {localizer} with params={localizer_params}")
            else:
                self.localizer = localizer
        else:
            self.localizer = None
        
        # Setup RCA
        self.enable_rca = enable_rca and rca is not None
        if self.enable_rca:
            if isinstance(rca, str):
                self.rca = RCARegistry.create(rca, **rca_params)
                logger.info(f"Created RCA analyzer from registry: {rca} with params={rca_params}")
            else:
                self.rca = rca
        else:
            self.rca = None
        
        self.validate_data = validate_data
        self.model = model
        
        logger.info(
            f"Pipeline initialized: "
            f"detector={self.detector.name}, "
            f"localization={self.enable_localization}, "
            f"rca={self.enable_rca}"
        )

    @staticmethod
    def _slice_dataset(dataset: Dataset, mask: np.ndarray) -> Dataset:
        """Create a dataset slice using a boolean mask."""
        if len(mask) != dataset.n_samples:
            raise ValueError("Slice mask length does not match dataset length")

        # Slice feature data
        if isinstance(dataset.data, pd.DataFrame):
            sliced_data = dataset.data.loc[mask].reset_index(drop=True)
        else:
            sliced_data = dataset.data[mask]

        # Slice target
        sliced_target = None
        if dataset.target is not None:
            if isinstance(dataset.target, pd.Series):
                sliced_target = dataset.target.loc[mask].reset_index(drop=True)
            else:
                sliced_target = dataset.target[mask]

        # Slice predictions
        sliced_predictions = dataset.predictions[mask] if dataset.predictions is not None else None
        sliced_prediction_probabilities = (
            dataset.prediction_probabilities[mask]
            if dataset.prediction_probabilities is not None
            else None
        )

        # Slice metadata
        sliced_metadata = None
        if dataset.metadata is not None:
            sliced_metadata = dataset.metadata.loc[mask].reset_index(drop=True)

        return Dataset(
            data=sliced_data,
            feature_names=dataset.feature_names,
            target=sliced_target,
            target_name=dataset.target_name,
            predictions=sliced_predictions,
            prediction_probabilities=sliced_prediction_probabilities,
            metadata=sliced_metadata,
        )

    def _run_single(self, reference_data: Dataset, test_data: Dataset) -> PipelineResult:
        """Run one detection-localization-rca pass on a single dataset pair."""
        # Validate data
        if self.validate_data:
            DataValidator.validate_dataset(reference_data, name="reference")
            DataValidator.validate_dataset(test_data, name="test")
            DataValidator.validate_compatibility(reference_data, test_data)

        # Step 1: Drift Detection
        logger.info(f"[1/3] Running drift detection with {self.detector.name}")
        detection_result = self.detector.fit_detect(reference_data, test_data)
        logger.info(
            f"Detection complete: drift_detected={detection_result.drift_detected}, "
            f"severity={detection_result.severity.value}, "
            f"score={detection_result.score:.4f}"
        )

        # Step 2: Drift Localization (conditional)
        localization_result = None
        if self.enable_localization:
            logger.info(f"[2/3] Running drift localization with {self.localizer.name}")
            try:
                localization_result = self.localizer.localize(
                    reference_data,
                    test_data,
                    drift_signal=detection_result,
                )
                logger.info(
                    f"Localization complete: "
                    f"{len(localization_result.drifted_features)} drifted features"
                )
            except Exception as e:
                logger.error(f"Drift localization failed: {e}")
                # Continue pipeline even if localization fails
                localization_result = None
        else:
            logger.info("[2/3] Localization disabled, skipping")

        # Step 3: Root Cause Analysis (conditional)
        rca_result = None
        if self.enable_rca:
            logger.info(f"[3/3] Running RCA with {self.rca.name}")
            try:
                rca_result = self.rca.analyze(
                    reference_data,
                    test_data,
                    localization=localization_result,
                    model=self.model,
                )
                logger.info("RCA complete")
            except Exception as e:
                logger.error(f"RCA failed: {e}")
                # Continue even if RCA fails
                rca_result = None
        else:
            logger.info("[3/3] RCA disabled, skipping")

        return PipelineResult(
            detection=detection_result,
            localization=localization_result,
            rca=rca_result,
        )

    @staticmethod
    def _set_threshold(component: Any, threshold_value: Optional[float], component_name: str) -> Optional[float]:
        """Set a runtime threshold override and return original value for restoration."""
        if threshold_value is None or component is None:
            return None

        if not hasattr(component, "threshold"):
            logger.warning(
                f"{component_name} does not expose a 'threshold' attribute; "
                "runtime override ignored"
            )
            return None

        original_value = getattr(component, "threshold")
        setattr(component, "threshold", threshold_value)
        logger.info(
            f"Applied runtime threshold override for {component_name}: "
            f"{original_value} -> {threshold_value}"
        )
        return original_value
    
    def run(
        self,
        reference_data: Dataset,
        test_data: Dataset,
        detection_threshold: Optional[float] = None,
        localization_threshold: Optional[float] = None,
        slice_config: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Run the complete drift analysis pipeline.
        
        Args:
            reference_data: Reference dataset (e.g., training data)
            test_data: Test dataset (e.g., production data)
            detection_threshold: Optional runtime threshold override for detector
            localization_threshold: Optional runtime threshold override for localizer
            slice_config: Optional slice analysis config
                - enabled: bool
                - column: metadata column name
                - min_samples_per_slice: minimum rows required in both ref/test per slice
                - reference_slice_value: optional fixed reference slice value for cross-slice comparisons
        
        Returns:
            PipelineResult with detection, localization, and RCA results
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("Starting drift analysis pipeline")
        logger.info(f"Reference: {reference_data.shape}, Test: {test_data.shape}")
        logger.info("=" * 60)
        
        original_detection_threshold = None
        original_localization_threshold = None

        try:
            # Apply runtime threshold overrides (if provided)
            original_detection_threshold = self._set_threshold(
                self.detector,
                detection_threshold,
                component_name="detector",
            )
            original_localization_threshold = self._set_threshold(
                self.localizer,
                localization_threshold,
                component_name="localizer",
            )

            # Run aggregate/global analysis first
            aggregate_result = self._run_single(reference_data, test_data)

            # Optional slice analysis
            slice_results = {}
            if slice_config and slice_config.get("enabled", False):
                slice_column = slice_config.get("column")
                min_samples_per_slice = int(slice_config.get("min_samples_per_slice", 30))
                reference_slice_value = slice_config.get("reference_slice_value")

                if not slice_column:
                    raise ValueError("slice_config.enabled=True requires slice_config.column")

                if reference_data.metadata is None or test_data.metadata is None:
                    raise ValueError("Slice analysis requires metadata in both reference and test datasets")

                if slice_column not in reference_data.metadata.columns:
                    raise ValueError(f"Slice column '{slice_column}' not found in reference metadata")

                if slice_column not in test_data.metadata.columns:
                    raise ValueError(f"Slice column '{slice_column}' not found in test metadata")

                ref_series = reference_data.metadata[slice_column].astype(str)
                test_series = test_data.metadata[slice_column].astype(str)

                if reference_slice_value is not None:
                    # Cross-slice mode: compare one reference slice to every test slice
                    ref_value = str(reference_slice_value)
                    available_ref = set(ref_series.dropna().unique())
                    if ref_value not in available_ref:
                        available_preview = sorted(list(available_ref))[:10]
                        raise ValueError(
                            f"reference_slice_value '{ref_value}' not present in reference metadata. "
                            f"Sample available values: {available_preview}"
                        )

                    ref_mask_fixed = (ref_series.values == ref_value)
                    test_slices = sorted(set(test_series.dropna().unique()))
                    logger.info(
                        f"Running cross-slice analysis on column='{slice_column}' "
                        f"using reference_slice_value='{ref_value}' against {len(test_slices)} test slices"
                    )

                    slice_pairs = [(ref_value, test_slice) for test_slice in test_slices]
                else:
                    # Same-slice mode: compare matching slice values in ref and test
                    ref_values = set(ref_series.dropna().unique())
                    test_values = set(test_series.dropna().unique())
                    common_slices = sorted(ref_values & test_values)
                    logger.info(
                        f"Running same-slice analysis on column='{slice_column}' "
                        f"for {len(common_slices)} common slices"
                    )
                    slice_pairs = [(slice_value, slice_value) for slice_value in common_slices]

                for ref_slice_value, test_slice_value in slice_pairs:
                    if reference_slice_value is not None:
                        ref_mask = ref_mask_fixed
                    else:
                        ref_mask = (ref_series.values == ref_slice_value)

                    test_mask = (test_series.values == test_slice_value)

                    n_ref = int(np.sum(ref_mask))
                    n_test = int(np.sum(test_mask))
                    if n_ref < min_samples_per_slice or n_test < min_samples_per_slice:
                        logger.info(
                            f"Skipping slice pair ref='{ref_slice_value}', test='{test_slice_value}' "
                            f"due to low sample size "
                            f"(ref={n_ref}, test={n_test}, min={min_samples_per_slice})"
                        )
                        continue

                    ref_slice = self._slice_dataset(reference_data, ref_mask)
                    test_slice = self._slice_dataset(test_data, test_mask)

                    logger.info(
                        f"Running slice pair ref='{ref_slice_value}', test='{test_slice_value}' "
                        f"(ref={ref_slice.n_samples}, test={test_slice.n_samples})"
                    )
                    slice_result = self._run_single(ref_slice, test_slice)

                    # Serialize slice result now so metadata stays JSON-friendly
                    slice_key = (
                        f"{ref_slice_value}->{test_slice_value}"
                        if reference_slice_value is not None
                        else str(test_slice_value)
                    )
                    slice_results[slice_key] = {
                        "reference_slice_value": ref_slice_value,
                        "test_slice_value": test_slice_value,
                        "reference_samples": ref_slice.n_samples,
                        "test_samples": test_slice.n_samples,
                        "result": slice_result.to_dict(),
                    }

                # Store localization slice details in aggregate localization result if available
                if aggregate_result.localization is not None:
                    aggregate_result.localization.slice_drifts = {
                        slice_name: slice_payload["result"].get("localization")
                        for slice_name, slice_payload in slice_results.items()
                        if slice_payload["result"].get("localization") is not None
                    }

            # Compute execution time
            execution_time = time.time() - start_time

            logger.info("=" * 60)
            logger.info(f"Pipeline complete in {execution_time:.2f}s")
            logger.info("=" * 60)

            # Build result with metadata
            result = PipelineResult(
                detection=aggregate_result.detection,
                localization=aggregate_result.localization,
                rca=aggregate_result.rca,
                execution_time_seconds=execution_time,
                metadata={
                    "detector": self.detector.name,
                    "localizer": self.localizer.name if self.localizer else None,
                    "rca": self.rca.name if self.rca else None,
                    "reference_samples": reference_data.n_samples,
                    "test_samples": test_data.n_samples,
                    "n_features": reference_data.n_features,
                    "effective_detection_threshold": getattr(self.detector, "threshold", None),
                    "effective_localization_threshold": getattr(self.localizer, "threshold", None)
                    if self.localizer
                    else None,
                    "threshold_source": {
                        "detection": "runtime_override" if detection_threshold is not None else "component_default",
                        "localization": "runtime_override" if localization_threshold is not None else "component_default",
                    },
                    "slice_analysis": {
                        "enabled": bool(slice_config and slice_config.get("enabled", False)),
                        "column": slice_config.get("column") if slice_config else None,
                        "slice_count": len(slice_results),
                        "slices": slice_results,
                    },
                },
            )

            return result
        finally:
            # Restore component thresholds to original values
            if original_detection_threshold is not None:
                setattr(self.detector, "threshold", original_detection_threshold)
            if original_localization_threshold is not None and self.localizer is not None:
                setattr(self.localizer, "threshold", original_localization_threshold)
    
    def __repr__(self) -> str:
        """String representation."""
        components = [f"detector={self.detector.name}"]
        if self.enable_localization:
            components.append(f"localizer={self.localizer.name}")
        if self.enable_rca:
            components.append(f"rca={self.rca.name}")
        
        return f"DriftPipeline({', '.join(components)})"
