"""AI Cyber Risk Advisor (AICRA) - Machine Learning-Based Cyber Risk Assessment."""

__version__ = "1.0.0"
__author__ = "AICRA Team"

from .config import Settings
from .core.calibration import Calibrator, create_calibrator
from .core.data import Dataset
from .core.evaluation import Metrics, cost_sensitive_threshold, evaluate_probs
from .models.lightgbm import BaggedLightGBM, train_bagged_lightgbm
from .pipelines.calibration import CalibrationPipeline
from .pipelines.evaluation import EvaluationPipeline
from .pipelines.training import TrainingPipeline
from .register import Policy, compute_register, write_register

# Optional import for drift pipeline (may fail with newer evidently versions)
try:
    from .pipelines.drift import DriftPipeline
except ImportError:
    DriftPipeline = None  # type: ignore

__all__ = [
    "Settings",
    "Dataset",
    "Metrics",
    "evaluate_probs",
    "cost_sensitive_threshold",
    "Calibrator",
    "create_calibrator",
    "BaggedLightGBM",
    "train_bagged_lightgbm",
    "TrainingPipeline",
    "EvaluationPipeline",
    "CalibrationPipeline",
    "Policy",
    "compute_register",
    "write_register",
]

# Add DriftPipeline to __all__ only if it was successfully imported
if DriftPipeline is not None:
    __all__.append("DriftPipeline")
