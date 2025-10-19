"""Tests for calibration pipeline."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from aicra.pipelines.calibration import CalibrationPipeline, PlattCalibrator, IsotonicCalibrator
from aicra.config import Settings
from aicra.core.data import Dataset


class TestCalibrationPipeline:
    """Test calibration pipeline functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = Settings()
        self.pipeline = CalibrationPipeline(self.settings)
        
        # Create test data
        self.train_data = Dataset(
            features=pd.DataFrame(np.random.rand(5, 10)),
            labels=pd.Series([0, 1, 0, 1, 0]),
            families=pd.Series(["family1", "family2", "family1", "family2", "family1"])
        )
        self.val_data = Dataset(
            features=pd.DataFrame(np.random.rand(3, 10)),
            labels=pd.Series([0, 1, 0]),
            families=pd.Series(["family1", "family2", "family1"])
        )
        self.y_prob_train = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        self.y_prob_val = np.array([0.15, 0.85, 0.25])
    
    def test_create_calibrator_platt(self):
        """Test creating Platt calibrator."""
        calibrator = self.pipeline._create_calibrator("platt")
        assert isinstance(calibrator, PlattCalibrator)
    
    def test_create_calibrator_isotonic(self):
        """Test creating isotonic calibrator."""
        calibrator = self.pipeline._create_calibrator("isotonic")
        assert isinstance(calibrator, IsotonicCalibrator)
    
    def test_create_calibrator_invalid(self):
        """Test creating calibrator with invalid method."""
        with pytest.raises(ValueError):
            self.pipeline._create_calibrator("invalid")
    
    @patch('aicra.pipelines.calibration.mlflow.log_param')
    @patch('aicra.pipelines.calibration.mlflow.log_metrics')
    @patch('aicra.pipelines.calibration.mlflow.log_artifacts')
    def test_run_platt(self, mock_log_artifacts, mock_log_metrics, mock_log_param):
        """Test running calibration pipeline with Platt scaling."""
        calibrator = self.pipeline.run(
            self.train_data,
            self.val_data,
            self.y_prob_train,
            self.y_prob_val,
            method="platt"
        )
        
        assert isinstance(calibrator, PlattCalibrator)
        mock_log_param.assert_called()
        mock_log_metrics.assert_called()
        mock_log_artifacts.assert_called()
    
    @patch('aicra.pipelines.calibration.mlflow.log_param')
    @patch('aicra.pipelines.calibration.mlflow.log_metrics')
    @patch('aicra.pipelines.calibration.mlflow.log_artifacts')
    def test_run_isotonic(self, mock_log_artifacts, mock_log_metrics, mock_log_param):
        """Test running calibration pipeline with isotonic regression."""
        calibrator = self.pipeline.run(
            self.train_data,
            self.val_data,
            self.y_prob_train,
            self.y_prob_val,
            method="isotonic"
        )
        
        assert isinstance(calibrator, IsotonicCalibrator)
        mock_log_param.assert_called()
        mock_log_metrics.assert_called()
        mock_log_artifacts.assert_called()
    
    def test_compute_ece(self):
        """Test ECE computation."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        
        ece = self.pipeline._compute_ece(y_true, y_prob)
        
        assert isinstance(ece, float)
        assert 0 <= ece <= 1
    
    def test_compute_ece_perfect_calibration(self):
        """Test ECE with perfectly calibrated probabilities."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        
        ece = self.pipeline._compute_ece(y_true, y_prob)
        
        assert ece == 0.0
    
    def test_post_ensemble_calibration_check_improvement(self):
        """Test post-ensemble calibration check with improvement."""
        calibrator = IsotonicCalibrator()
        calibrator.fit(self.y_prob_train, self.train_data.labels)
        
        # Mock ECE computation to simulate improvement
        with patch.object(self.pipeline, '_compute_ece', side_effect=[0.1, 0.05]):
            result = self.pipeline._post_ensemble_calibration_check(
                calibrator, self.y_prob_train, self.train_data.labels, "isotonic"
            )
        
        assert isinstance(result, IsotonicCalibrator)
    
    def test_post_ensemble_calibration_check_degradation(self):
        """Test post-ensemble calibration check with degradation."""
        calibrator = IsotonicCalibrator()
        calibrator.fit(self.y_prob_train, self.train_data.labels)
        
        # Mock ECE computation to simulate degradation
        with patch.object(self.pipeline, '_compute_ece', side_effect=[0.05, 0.1]):
            result = self.pipeline._post_ensemble_calibration_check(
                calibrator, self.y_prob_train, self.train_data.labels, "isotonic"
            )
        
        assert isinstance(result, IsotonicCalibrator)


class TestPlattCalibrator:
    """Test Platt scaling calibrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calibrator = PlattCalibrator()
    
    def test_fit_and_transform(self):
        """Test fitting and transforming with Platt scaling."""
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        y_true = np.array([0, 1, 0, 1])
        
        self.calibrator.fit(y_prob, y_true)
        
        assert self.calibrator.is_fitted
        
        calibrated_probs = self.calibrator.transform(y_prob)
        
        assert len(calibrated_probs) == len(y_prob)
        assert all(0 <= p <= 1 for p in calibrated_probs)
    
    def test_transform_without_fit(self):
        """Test transforming without fitting first."""
        y_prob = np.array([0.1, 0.9])
        
        with pytest.raises(ValueError):
            self.calibrator.transform(y_prob)
    
    def test_edge_cases(self):
        """Test edge cases for Platt scaling."""
        # Test with extreme probabilities
        y_prob = np.array([0.0, 1.0, 0.5])
        y_true = np.array([0, 1, 0])
        
        self.calibrator.fit(y_prob, y_true)
        calibrated_probs = self.calibrator.transform(y_prob)
        
        assert len(calibrated_probs) == len(y_prob)
        assert all(0 <= p <= 1 for p in calibrated_probs)


class TestIsotonicCalibrator:
    """Test isotonic regression calibrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calibrator = IsotonicCalibrator()
    
    def test_fit_and_transform(self):
        """Test fitting and transforming with isotonic regression."""
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        y_true = np.array([0, 1, 0, 1])
        
        self.calibrator.fit(y_prob, y_true)
        
        assert self.calibrator.is_fitted
        
        calibrated_probs = self.calibrator.transform(y_prob)
        
        assert len(calibrated_probs) == len(y_prob)
        assert all(0 <= p <= 1 for p in calibrated_probs)
    
    def test_transform_without_fit(self):
        """Test transforming without fitting first."""
        y_prob = np.array([0.1, 0.9])
        
        with pytest.raises(ValueError):
            self.calibrator.transform(y_prob)
    
    def test_monotonic_property(self):
        """Test that isotonic regression maintains monotonicity."""
        y_prob = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_true = np.array([0, 0, 1, 1, 1])
        
        self.calibrator.fit(y_prob, y_true)
        calibrated_probs = self.calibrator.transform(y_prob)
        
        # Check monotonicity
        for i in range(len(calibrated_probs) - 1):
            assert calibrated_probs[i] <= calibrated_probs[i + 1]
    
    def test_edge_cases(self):
        """Test edge cases for isotonic regression."""
        # Test with extreme probabilities
        y_prob = np.array([0.0, 1.0, 0.5])
        y_true = np.array([0, 1, 0])
        
        self.calibrator.fit(y_prob, y_true)
        calibrated_probs = self.calibrator.transform(y_prob)
        
        assert len(calibrated_probs) == len(y_prob)
        assert all(0 <= p <= 1 for p in calibrated_probs)