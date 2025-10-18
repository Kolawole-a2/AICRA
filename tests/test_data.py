"""Tests for data handling functionality."""

from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from aicra.core.data import Dataset, _synthetic_dataset, load_ember_2024


def test_dataset_creation() -> None:
    """Test Dataset creation."""
    # Create synthetic data
    features = pd.DataFrame({
        'feature_0': [1, 2, 3],
        'feature_1': [4, 5, 6]
    })
    labels = pd.Series([0, 1, 0])
    families = pd.Series(['benign', 'ransomware', 'benign'])
    timestamps = pd.Series(pd.date_range('2024-01-01', periods=3))

    dataset = Dataset(
        features=features,
        labels=labels,
        families=families,
        timestamps=timestamps
    )

    assert len(dataset.features) == 3
    assert len(dataset.labels) == 3
    assert len(dataset.families) == 3
    assert len(dataset.timestamps) == 3


def test_synthetic_dataset() -> None:
    """Test synthetic dataset generation."""
    train, test = _synthetic_dataset(n=1000, d=50, seed=42)

    assert isinstance(train, Dataset)
    assert isinstance(test, Dataset)

    # Check sizes
    assert len(train.features) == 800  # 80% of 1000
    assert len(test.features) == 200   # 20% of 1000

    # Check features
    assert train.features.shape[1] == 50
    assert test.features.shape[1] == 50

    # Check labels
    assert set(train.labels.unique()) <= {0, 1}
    assert set(test.labels.unique()) <= {0, 1}

    # Check families
    assert set(train.families.unique()) <= {'lockbit', 'blackcat', 'benign'}
    assert set(test.families.unique()) <= {'lockbit', 'blackcat', 'benign'}


def test_synthetic_dataset_deterministic() -> None:
    """Test that synthetic dataset is deterministic with same seed."""
    train1, test1 = _synthetic_dataset(n=100, d=10, seed=42)
    train2, test2 = _synthetic_dataset(n=100, d=10, seed=42)

    # Should be identical with same seed
    pd.testing.assert_frame_equal(train1.features, train2.features)
    pd.testing.assert_series_equal(train1.labels, train2.labels)
    pd.testing.assert_series_equal(train1.families, train2.families)

    pd.testing.assert_frame_equal(test1.features, test2.features)
    pd.testing.assert_series_equal(test1.labels, test2.labels)
    pd.testing.assert_series_equal(test1.families, test2.families)


def test_synthetic_dataset_different_seeds() -> None:
    """Test that synthetic dataset differs with different seeds."""
    train1, test1 = _synthetic_dataset(n=100, d=10, seed=42)
    train2, test2 = _synthetic_dataset(n=100, d=10, seed=123)

    # Should be different with different seeds
    assert not train1.features.equals(train2.features)
    assert not train1.labels.equals(train2.labels)


@patch('aicra.core.data.get_settings')
def test_load_ember_2024_file_not_found(mock_get_settings) -> None:
    """Test load_ember_2024 when files don't exist."""
    # Mock settings
    mock_settings = mock_get_settings.return_value
    mock_settings.ember_dir = Path("/nonexistent/path")

    with pytest.raises(FileNotFoundError):
        load_ember_2024()


@patch('aicra.core.data.get_settings')
@patch('builtins.open', new_callable=mock_open)
@patch('pandas.read_json')
def test_load_ember_2024_success(mock_read_json, mock_file, mock_get_settings):
    """Test successful load_ember_2024."""
    # Mock settings
    mock_settings = mock_get_settings.return_value
    mock_settings.ember_dir = Path("/mock/path")

    # Mock file existence
    with patch('pathlib.Path.exists', return_value=True):
        # Mock pandas read_json
        mock_features = pd.DataFrame({
            'feature_0': [1, 2, 3],
            'feature_1': [4, 5, 6],
            'family': ['benign', 'ransomware', 'benign']
        })
        mock_labels = pd.DataFrame({'label': [0, 1, 0]})

        mock_read_json.side_effect = [mock_features, mock_labels, mock_features, mock_labels]

        train, test = load_ember_2024()

        assert isinstance(train, Dataset)
        assert isinstance(test, Dataset)
        assert len(train.features) == 3
        assert len(test.features) == 3
