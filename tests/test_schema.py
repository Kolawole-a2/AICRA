"""Tests for schema validation."""


import pytest

from aicra.utils.schema_validator import validate_input_schema


def test_validate_input_schema_valid() -> None:
    """Test schema validation with valid input."""
    valid_data = {
        "features": [
            {
                "feature_0": 1.0,
                "feature_1": 2.0,
                "feature_2": 3.0
            }
        ],
        "metadata": {
            "family": "benign",
            "timestamp": "2024-01-01T00:00:00Z",
            "file_hash": "abc123",
            "file_size": 1024
        }
    }

    # Should not raise exception
    validate_input_schema(valid_data)


def test_validate_input_schema_invalid() -> None:
    """Test schema validation with invalid input."""
    invalid_data = {
        "features": [
            {
                "feature_0": "invalid",  # Should be number
                "feature_1": 2.0,
                "feature_2": 3.0
            }
        ],
        "metadata": {
            "family": "benign",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    }

    with pytest.raises(ValueError):
        validate_input_schema(invalid_data)


def test_validate_input_schema_missing_required() -> None:
    """Test schema validation with missing required fields."""
    invalid_data = {
        "features": [
            {
                "feature_0": 1.0,
                "feature_1": 2.0,
                "feature_2": 3.0
            }
        ]
        # Missing metadata
    }

    with pytest.raises(ValueError):
        validate_input_schema(invalid_data)


def test_validate_input_schema_empty_features() -> None:
    """Test schema validation with empty features."""
    invalid_data = {
        "features": [],  # Empty features
        "metadata": {
            "family": "benign",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    }

    with pytest.raises(ValueError):
        validate_input_schema(invalid_data)


def test_validate_input_schema_additional_properties() -> None:
    """Test schema validation with additional properties."""
    valid_data = {
        "features": [
            {
                "feature_0": 1.0,
                "feature_1": 2.0,
                "feature_2": 3.0,
                "feature_3": 4.0  # Additional feature
            }
        ],
        "metadata": {
            "family": "benign",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    }

    # Should not raise exception (additional properties allowed)
    validate_input_schema(valid_data)
