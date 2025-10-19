"""Configuration management using pydantic-settings."""

import subprocess
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Paths
    repo_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")
    ember_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "ember2024")
    artifacts_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "artifacts")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "models")
    metrics_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "metrics")
    register_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "register")
    policies_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "policies")
    reports_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "reports")
    mappings_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "mappings")

    # Training configuration
    random_seeds: tuple[int, int, int] = (17, 42, 73)
    random_seed: int = 42
    use_isotonic: bool = False
    class_weight: str | None = "balanced"
    learning_rate: float = 0.05
    num_leaves: int = 64
    n_estimators: int = 400
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    goss: bool = False  # MUST be off per constraints

    # Cost-sensitive thresholding
    cost_fp: float = 5.0  # Cost of false positive
    cost_fn: float = 100.0  # Cost of false negative
    impact_default: float = 5000000.0  # Default impact in dollars

    # Drift detection
    drift_threshold: float = 0.05

    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "aicra"

    # Git information
    git_commit: str = Field(default_factory=lambda: _get_git_commit())

    # Coverage threshold
    coverage_fail_under: float = 40.0
    
    # Mapping configuration
    max_unmapped_rate: float = 0.05  # Maximum allowed unmapped rate (5%)
    mapping_cache_size: int = 1000  # LRU cache size for mapping operations
    mapping_timeout_seconds: int = 30  # Timeout for mapping operations
    
    # Dataset type and mapping requirements
    dataset_type: str = "ember"  # Dataset type: "ember", "bank_logs", etc.
    require_family_mapping: bool = False  # Whether to require family mapping (False for EMBER)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Ensure directories exist
        for attr_name in dir(self):
            if attr_name.endswith('_dir'):
                dir_path = getattr(self, attr_name)
                if isinstance(dir_path, Path):
                    dir_path.mkdir(parents=True, exist_ok=True)


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# Global settings instance - will be replaced with dependency injection
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get settings instance (singleton pattern)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Set settings instance (for testing)."""
    global _settings
    _settings = settings
