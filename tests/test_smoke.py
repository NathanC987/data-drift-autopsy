"""Basic smoke tests for drift autopsy components."""

import numpy as np
import pytest
from drift_autopsy import DriftPipeline, Dataset
from drift_autopsy.detectors import KSTest
from drift_autopsy.localizers import UnivariateLocalizer
from drift_autopsy.registry import DetectorRegistry


def test_dataset_creation():
    """Test Dataset creation from numpy."""
    X = np.random.randn(100, 5)
    dataset = Dataset.from_numpy(X)
    
    assert dataset.n_samples == 100
    assert dataset.n_features == 5
    assert len(dataset.feature_names) == 5


def test_detector_registry():
    """Test detector registry."""
    detectors = DetectorRegistry.list()
    
    assert "ks_test" in detectors
    assert "psi" in detectors
    assert "mmd" in detectors
    assert "cbpe" in detectors


def test_ks_detector_no_drift():
    """Test KS detector with no drift."""
    np.random.seed(42)
    
    # Same distribution
    ref = Dataset.from_numpy(np.random.randn(500, 3))
    test = Dataset.from_numpy(np.random.randn(500, 3))
    
    detector = KSTest(threshold=0.05)
    result = detector.fit_detect(ref, test)
    
    assert result.drift_detected is False
    assert result.p_value is not None


def test_ks_detector_with_drift():
    """Test KS detector with obvious drift."""
    np.random.seed(42)
    
    # Different distributions
    ref = Dataset.from_numpy(np.random.randn(500, 3))
    test = Dataset.from_numpy(np.random.randn(500, 3) + 2.0)  # Large shift
    
    detector = KSTest(threshold=0.05)
    result = detector.fit_detect(ref, test)
    
    assert result.drift_detected is True


def test_pipeline_basic():
    """Test basic pipeline execution."""
    np.random.seed(42)
    
    ref = Dataset.from_numpy(np.random.randn(300, 4))
    test = Dataset.from_numpy(np.random.randn(300, 4) + 0.5)
    
    pipeline = DriftPipeline(
        detector=KSTest(threshold=0.05),
        localizer="univariate",
        enable_localization=True,
    )
    
    result = pipeline.run(ref, test)
    
    assert result.detection is not None
    assert result.localization is not None
    assert result.execution_time_seconds > 0


def test_univariate_localizer():
    """Test univariate localizer."""
    np.random.seed(42)
    
    # Feature 0 has drift, others don't
    ref_data = np.random.randn(400, 3)
    test_data = np.random.randn(400, 3)
    test_data[:, 0] += 1.5  # Add drift to first feature
    
    ref = Dataset.from_numpy(ref_data)
    test = Dataset.from_numpy(test_data)
    
    localizer = UnivariateLocalizer(threshold=0.05)
    result = localizer.localize(ref, test)
    
    assert len(result.drifted_features) > 0
    assert "feature_0" in result.drifted_features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
