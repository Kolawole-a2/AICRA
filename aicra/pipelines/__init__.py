"""ML pipelines for AICRA."""

from .calibration import CalibrationPipeline
from .evaluation import EvaluationPipeline
from .training import TrainingPipeline

# Optional import for drift pipeline (may fail with newer evidently versions)
try:
    from .drift import DriftPipeline
except ImportError:
    DriftPipeline = None  # type: ignore

__all__ = [
    "TrainingPipeline",
    "EvaluationPipeline",
    "CalibrationPipeline",
]

# Add DriftPipeline to __all__ only if it was successfully imported
if DriftPipeline is not None:
    __all__.append("DriftPipeline")
