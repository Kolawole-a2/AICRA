from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    repo_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = repo_root / "data"
    ember_dir: Path = data_dir / "ember2024"
    artifacts_dir: Path = repo_root / "artifacts"
    models_dir: Path = repo_root / "models"
    metrics_dir: Path = repo_root / "metrics"
    register_dir: Path = repo_root / "register"
    policies_dir: Path = repo_root / "policies"
    reports_dir: Path = repo_root / "reports"
    mappings_dir: Path = repo_root / "mappings"


@dataclass
class TrainingConfig:
    random_seeds: tuple[int, int, int] = (17, 42, 73)
    use_isotonic: bool = False
    class_weight: str | None = "balanced"
    learning_rate: float = 0.05
    num_leaves: int = 64
    n_estimators: int = 400
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    goss: bool = False  # MUST be off per constraints


PATHS = Paths()
TRAINING = TrainingConfig()
