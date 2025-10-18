"""Core AICRA modules for data handling, evaluation, and calibration."""

from .calibration import Calibrator, create_calibrator
from .data import Dataset, _synthetic_dataset, load_ember_2024
from .evaluation import (
    Metrics,
    compute_lift_at_k,
    cost_sensitive_threshold,
    evaluate_probs,
    expected_calibration_error,
)

__all__ = [
    "Dataset",
    "load_ember_2024",
    "_synthetic_dataset",
    "Metrics",
    "evaluate_probs",
    "cost_sensitive_threshold",
    "expected_calibration_error",
    "compute_lift_at_k",
    "Calibrator",
    "create_calibrator",
]

