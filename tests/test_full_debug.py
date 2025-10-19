"""Tests for full EMBER-2024 debug functionality."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest
import pandas as pd
import numpy as np

from aicra.config import Settings
from aicra.pipelines.full_debug import FullDebugPipeline


class TestFullDebug:
    """Test debug functionality for full EMBER-2024 phase."""
    
    @pytest.fixture
    def temp_settings(self):
        """Create temporary settings for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test lookup files
            lookups_dir = temp_path / "data" / "lookups"
            lookups_dir.mkdir(parents=True)
            
            # Create minimal canonical families mapping
            families_data = {
                '__version__': '1.0.0-test',
                'mappings': {
                    'test_family': 'TestFamily',
                    'unknown': 'Unknown'
                }
            }
            
            import yaml
            with open(lookups_dir / "canonical_families.yaml", 'w') as f:
                yaml.dump(families_data, f)
            
            # Create family to attack mapping
            attack_data = {
                '__version__': '1.0.0-test',
                'mappings': {
                    'TestFamily': ['T1486', 'T1055'],
                    'Unknown': []
                }
            }
            
            with open(lookups_dir / "family_to_attack.yaml", 'w') as f:
                yaml.dump(attack_data, f)
            
            # Create attack to d3fend mapping
            d3fend_data = {
                '__version__': '1.0.0-test',
                'mappings': {
                    'T1486': ['D3-CM001', 'D3-CM002'],
                    'T1055': ['D3-CM003']
                }
            }
            
            with open(lookups_dir / "attack_to_d3fend.yaml", 'w') as f:
                yaml.dump(d3fend_data, f)
            
            # Create settings
            settings = Settings()
            settings.data_dir = temp_path / "data"
            settings.artifacts_dir = temp_path / "artifacts"
            settings.artifacts_dir.mkdir(parents=True)
            
            yield settings
    
    def test_single_class_detection(self, temp_settings):
        """Test detection of single-class labels."""
        debug_pipeline = FullDebugPipeline(temp_settings)
        
        # Create single-class data
        features_df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4, 5],
            'feature_1': [0, 1, 0, 1, 0]
        })
        labels_series = pd.Series([0, 0, 0, 0, 0])  # All same class
        families_series = pd.Series(['test', 'test', 'test', 'test', 'test'])
        
        # Should raise RuntimeError for single class
        with pytest.raises(RuntimeError, match="Only one class found in labels"):
            debug_pipeline.validate_data_loading(features_df, labels_series, families_series)
    
    def test_insufficient_data_detection(self, temp_settings):
        """Test detection of insufficient data."""
        debug_pipeline = FullDebugPipeline(temp_settings)
        
        # Create small dataset
        features_df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4, 5]  # Only 5 rows
        })
        labels_series = pd.Series([0, 1, 0, 1, 0])
        families_series = pd.Series(['test', 'test', 'test', 'test', 'test'])
        
        # Should raise RuntimeError for insufficient data
        with pytest.raises(RuntimeError, match="Insufficient data.*Need at least 10,000 rows"):
            debug_pipeline.validate_data_loading(features_df, labels_series, families_series)
    
    def test_constant_feature_removal(self, temp_settings):
        """Test removal of constant and near-constant features."""
        debug_pipeline = FullDebugPipeline(temp_settings)
        
        # Create data with constant features
        features_df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4, 5],  # Normal feature
            'feature_1': [1, 1, 1, 1, 1],  # Constant feature
            'feature_2': [1, 1, 1, 1, 1.0000001],  # Near-constant feature
            'feature_3': [0, 1, 0, 1, 0]  # Normal feature
        })
        labels_series = pd.Series([0, 1, 0, 1, 0])
        families_series = pd.Series(['test', 'test', 'test', 'test', 'test'])
        
        # Should not raise error but remove constant features
        debug_pipeline.validate_data_loading(features_df, labels_series, families_series)
        
        # Check that constant features were identified
        assert 'feature_1' in debug_pipeline.debug_data['data_summary']['constant_features']
        assert 'feature_2' in debug_pipeline.debug_data['data_summary']['near_constant_features']
    
    def test_data_leakage_detection(self, temp_settings):
        """Test detection of data leakage between train and test."""
        debug_pipeline = FullDebugPipeline(temp_settings)
        
        # Create data with overlapping IDs
        train_data = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D'],
            'feature_0': [1, 2, 3, 4],
            'feature_1': [0, 1, 0, 1]
        })
        test_data = pd.DataFrame({
            'id': ['C', 'D', 'E', 'F'],  # Overlap with train (C, D)
            'feature_0': [3, 4, 5, 6],
            'feature_1': [0, 1, 0, 1]
        })
        train_labels = pd.Series([0, 1, 0, 1])
        test_labels = pd.Series([0, 1, 0, 1])
        
        # Should raise RuntimeError for leakage
        with pytest.raises(RuntimeError, match="Data leakage detected"):
            debug_pipeline.validate_split_integrity(
                train_data, test_data, train_labels, test_labels, False
            )
    
    def test_time_split_validation(self, temp_settings):
        """Test time-ordered split validation."""
        debug_pipeline = FullDebugPipeline(temp_settings)
        
        # Create time-ordered data
        train_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='D'),
            'feature_0': [1, 2, 3, 4],
            'feature_1': [0, 1, 0, 1]
        })
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-05', periods=4, freq='D'),  # After train
            'feature_0': [5, 6, 7, 8],
            'feature_1': [0, 1, 0, 1]
        })
        train_labels = pd.Series([0, 1, 0, 1])
        test_labels = pd.Series([0, 1, 0, 1])
        
        # Should not raise error for proper time split
        debug_pipeline.validate_split_integrity(
            train_data, test_data, train_labels, test_labels, True
        )
        
        # Check time split summary
        assert debug_pipeline.debug_data['split_summary']['split_type'] == 'time_ordered'
        assert 'train_max_timestamp' in debug_pipeline.debug_data['split_summary']
        assert 'test_min_timestamp' in debug_pipeline.debug_data['split_summary']
    
    def test_debug_report_generation(self, temp_settings):
        """Test debug report generation."""
        debug_pipeline = FullDebugPipeline(temp_settings)
        
        # Create test data
        features_df = pd.DataFrame({
            'feature_0': np.random.randn(1000),
            'feature_1': np.random.randn(1000)
        })
        labels_series = pd.Series(np.random.randint(0, 2, 1000))
        families_series = pd.Series(['test'] * 1000)
        
        # Validate data loading
        debug_pipeline.validate_data_loading(features_df, labels_series, families_series)
        
        # Create mock split data
        train_data = pd.DataFrame({
            'feature_0': np.random.randn(800),
            'feature_1': np.random.randn(800)
        })
        test_data = pd.DataFrame({
            'feature_0': np.random.randn(200),
            'feature_1': np.random.randn(200)
        })
        train_labels = pd.Series(np.random.randint(0, 2, 800))
        test_labels = pd.Series(np.random.randint(0, 2, 200))
        
        debug_pipeline.validate_split_integrity(
            train_data, test_data, train_labels, test_labels, False
        )
        
        # Create mock training summary
        training_summary = {
            'best_iteration': 100,
            'best_score': {'valid_0': {'auc': 0.75}},
            'params': {'num_leaves': 64},
            'n_features_used': 2,
            'n_features_original': 2,
            'top_20_features': [
                {'feature': 'feature_0', 'importance': 0.6},
                {'feature': 'feature_1', 'importance': 0.4}
            ]
        }
        
        # Create mock test metrics
        test_metrics = {
            'auroc': 0.50,  # Low AUROC to trigger warnings
            'pr_auc': 0.25,
            'brier': 0.25,
            'ece': 0.15,
            'lift_at_5pct': 1.0
        }
        
        # Generate debug report
        report_path = debug_pipeline.generate_debug_report(test_metrics, None, training_summary)
        
        # Check that report files were created
        assert Path(report_path).exists()
        assert (temp_settings.artifacts_dir / "debug_full_report.md").exists()
        
        # Check report content
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        assert report_data['test_metrics']['auroc'] == 0.50
        assert len(report_data['probable_causes']) > 0  # Should have causes for low AUROC
        assert len(report_data['recommendations']) > 0  # Should have recommendations
    
    def test_lightgbm_retuning(self, temp_settings):
        """Test LightGBM retuning for large data."""
        debug_pipeline = FullDebugPipeline(temp_settings)
        
        # Create larger dataset for testing
        n_samples = 1000
        n_features = 50
        
        train_data = pd.DataFrame({
            f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
        })
        train_labels = pd.Series(np.random.randint(0, 2, n_samples))
        
        test_data = pd.DataFrame({
            f'feature_{i}': np.random.randn(200) for i in range(n_features)
        })
        test_labels = pd.Series(np.random.randint(0, 2, 200))
        
        # Test retuning
        model, training_summary = debug_pipeline.retune_lightgbm_large_data(
            train_data, train_labels, test_data, test_labels
        )
        
        # Check that model was trained
        assert model is not None
        assert training_summary['best_iteration'] > 0
        assert 'best_score' in training_summary
        assert training_summary['n_features_used'] <= n_features  # May be less due to cleaning
        
        # Check that feature importance was saved
        importance_path = temp_settings.artifacts_dir / "feature_importance_full.csv"
        assert importance_path.exists()
        
        importance_df = pd.read_csv(importance_path)
        assert len(importance_df) > 0
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
    
    def test_debug_artifacts_creation(self, temp_settings):
        """Test that all debug artifacts are created."""
        debug_pipeline = FullDebugPipeline(temp_settings)
        
        # Create test data
        features_df = pd.DataFrame({
            'feature_0': np.random.randn(1000),
            'feature_1': np.random.randn(1000)
        })
        labels_series = pd.Series(np.random.randint(0, 2, 1000))
        families_series = pd.Series(['test'] * 1000)
        
        # Run validation
        debug_pipeline.validate_data_loading(features_df, labels_series, families_series)
        
        # Check that artifacts were created
        expected_artifacts = [
            "debug_full_data_summary.json",
            "removed_features_full.csv"
        ]
        
        for artifact in expected_artifacts:
            artifact_path = temp_settings.artifacts_dir / artifact
            assert artifact_path.exists(), f"Missing artifact: {artifact}"
    
    def test_warning_accumulation(self, temp_settings):
        """Test that warnings are properly accumulated."""
        debug_pipeline = FullDebugPipeline(temp_settings)
        
        # Create data with high missing rates
        features_df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4, 5],
            'feature_1': [np.nan, np.nan, np.nan, 1, 0]  # 60% missing
        })
        labels_series = pd.Series([0, 1, 0, 1, 0])
        families_series = pd.Series(['test', 'test', 'test', 'test', 'test'])
        
        # Should not raise error but accumulate warnings
        debug_pipeline.validate_data_loading(features_df, labels_series, families_series)
        
        # Check that warnings were accumulated
        assert len(debug_pipeline.warnings) > 0
