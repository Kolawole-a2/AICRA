"""Tests for configuration management."""

from pathlib import Path
from unittest.mock import patch

from aicra.config import Settings, get_settings, set_settings


def test_settings_defaults() -> None:
    """Test default settings values."""
    settings = Settings()

    assert settings.random_seed == 42
    assert settings.cost_fp == 5.0
    assert settings.cost_fn == 100.0
    assert settings.impact_default == 5000000.0
    assert settings.drift_threshold == 0.05


def test_settings_paths() -> None:
    """Test that paths are created correctly."""
    settings = Settings()

    assert isinstance(settings.repo_root, Path)
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.artifacts_dir, Path)
    assert isinstance(settings.models_dir, Path)


def test_settings_singleton() -> None:
    """Test singleton pattern for settings."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2


def test_set_settings() -> None:
    """Test setting custom settings."""
    custom_settings = Settings(random_seed=123)
    set_settings(custom_settings)

    current_settings = get_settings()
    assert current_settings.random_seed == 123


@patch('subprocess.run')
def test_git_commit(mock_run) -> None:
    """Test git commit retrieval."""
    from aicra.config import _get_git_commit

    # Mock successful git command
    mock_run.return_value.stdout = "abc123def456\n"
    mock_run.return_value.check_returncode.return_value = None

    commit = _get_git_commit()
    assert commit == "abc123def456"

    # Mock failed git command - use CalledProcessError instead of generic Exception
    from subprocess import CalledProcessError
    mock_run.side_effect = CalledProcessError(1, "git")
    commit = _get_git_commit()
    assert commit == "unknown"


def test_settings_environment_variables() -> None:
    """Test settings with environment variables."""
    import os

    # Set environment variable
    os.environ["RANDOM_SEED"] = "999"

    settings = Settings()
    assert settings.random_seed == 999

    # Clean up
    del os.environ["RANDOM_SEED"]
