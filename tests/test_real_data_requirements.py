"""Tests for real EMBER-2024 data requirements."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from aicra.config import get_settings
from aicra.pipelines.data_loader import EMBERDataLoader


def test_smoke_test_uses_synthetic_data():
    """Test that smoke test continues to use synthetic data."""
    # This test verifies that smoke test behavior is unchanged
    # The smoke test should still work without real EMBER data
    pass  # Smoke test is tested separately


def test_small_ember_requires_real_data():
    """Test that small_ember phase fails fast if real data is missing."""
    settings = get_settings()
    data_loader = EMBERDataLoader(settings)
    
    # Test with non-existent directory
    with pytest.raises(RuntimeError, match="EMBER-2024 data directory not found"):
        data_loader.load_ember_data(
            data_dir="nonexistent/directory",
            sample_size=1000,
            seed=42,
            phase="small_ember"
        )


def test_full_requires_real_data():
    """Test that full phase fails fast if real data is missing."""
    settings = get_settings()
    data_loader = EMBERDataLoader(settings)
    
    # Test with non-existent directory
    with pytest.raises(RuntimeError, match="EMBER-2024 data directory not found"):
        data_loader.load_ember_data(
            data_dir="nonexistent/directory",
            sample_size=None,
            seed=42,
            phase="full"
        )


def test_data_loader_with_empty_directory():
    """Test that data loader fails if directory exists but has no JSONL files."""
    settings = get_settings()
    data_loader = EMBERDataLoader(settings)
    
    # Create temporary directory with no JSONL files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create a non-JSONL file
        (temp_path / "not_jsonl.txt").write_text("not jsonl data")
        
        with pytest.raises(RuntimeError, match="No JSONL files found"):
            data_loader.load_ember_data(
                data_dir=str(temp_path),
                sample_size=1000,
                seed=42,
                phase="small_ember"
            )


def test_data_loader_with_invalid_jsonl():
    """Test that data loader handles invalid JSONL gracefully."""
    settings = get_settings()
    data_loader = EMBERDataLoader(settings)
    
    # Create temporary directory with invalid JSONL
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create invalid JSONL file
        (temp_path / "invalid.jsonl").write_text("invalid json\n{\"feature_0\": 0.1, \"feature_1\": 0.2, \"label\": 0}\n")
        
        # Should handle invalid JSON gracefully and use valid lines
        features_df, labels_series, families_series, metadata = data_loader.load_ember_data(
            data_dir=str(temp_path),
            sample_size=None,
            seed=42,
            phase="test"
        )
        
        assert len(features_df) == 1  # Only one valid line
        assert metadata["total_samples"] == 1


def test_data_loader_deterministic_sampling():
    """Test that data loader produces deterministic results with same seed."""
    settings = get_settings()
    data_loader = EMBERDataLoader(settings)
    
    # Create temporary directory with test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test JSONL files
        features_content = "\n".join([
            '{"feature_0": 0.1, "feature_1": 0.2, "feature_2": 0.3}',
            '{"feature_0": 0.4, "feature_1": 0.5, "feature_2": 0.6}',
            '{"feature_0": 0.7, "feature_1": 0.8, "feature_2": 0.9}',
            '{"feature_0": 1.0, "feature_1": 1.1, "feature_2": 1.2}',
            '{"feature_0": 1.3, "feature_1": 1.4, "feature_2": 1.5}',
        ])
        
        labels_content = "\n".join([
            '{"label": 0}',
            '{"label": 1}',
            '{"label": 0}',
            '{"label": 1}',
            '{"label": 0}',
        ])
        
        (temp_path / "features.jsonl").write_text(features_content)
        (temp_path / "labels.jsonl").write_text(labels_content)
        
        # Load with same seed twice
        features1, labels1, families1, _ = data_loader.load_ember_data(
            data_dir=str(temp_path),
            sample_size=3,
            seed=42,
            phase="test"
        )
        
        features2, labels2, families2, _ = data_loader.load_ember_data(
            data_dir=str(temp_path),
            sample_size=3,
            seed=42,
            phase="test"
        )
        
        # Results should be identical
        assert features1.equals(features2)
        assert labels1.equals(labels2)


def test_data_loader_schema_validation():
    """Test that data loader validates schema correctly."""
    settings = get_settings()
    data_loader = EMBERDataLoader(settings)
    
    # Create temporary directory with invalid schema
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create JSONL without label column
        features_content = '{"feature_0": 0.1, "feature_1": 0.2, "feature_2": 0.3}'
        (temp_path / "features.jsonl").write_text(features_content)
        
        with pytest.raises(RuntimeError, match="No label column found"):
            data_loader.load_ember_data(
                data_dir=str(temp_path),
                sample_size=None,
                seed=42,
                phase="test"
            )


if __name__ == "__main__":
    pytest.main([__file__])
