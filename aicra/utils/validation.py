"""Validation utilities for risk scores and model outputs."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def assert_non_constant_scores(
    scores: np.ndarray | pd.Series,
    split_name: str,
    min_unique: int = 5,
    min_std: float = 1e-6,
) -> None:
    """
    Assert that risk scores are not constant.

    Args:
        scores: Array or Series of risk scores
        split_name: Name of the split (for error messages)
        min_unique: Minimum number of unique values required
        min_std: Minimum standard deviation required

    Raises:
        RuntimeError: If scores appear constant
    """
    if isinstance(scores, pd.Series):
        scores = scores.values

    unique_vals = np.unique(scores)
    std_val = np.std(scores)

    if len(unique_vals) < min_unique:
        raise RuntimeError(
            f"[{split_name}] Risk scores appear nearly constant: "
            f"{len(unique_vals)} unique values (minimum: {min_unique}). "
            f"Check model, features, or calibration pipeline."
        )

    if std_val < min_std:
        raise RuntimeError(
            f"[{split_name}] Risk scores have very low variance: "
            f"std={std_val:.10f} (minimum: {min_std}). "
            f"Check model, features, or calibration pipeline."
        )


def validate_risk_scores_file(
    file_path: str | Path,
    split_name: str | None = None,
    min_unique: int = 5,
    min_std: float = 1e-6,
) -> dict[str, Any]:
    """
    Validate a risk_scores.csv file.

    Args:
        file_path: Path to risk_scores.csv file
        split_name: Name of the split (defaults to filename)
        min_unique: Minimum number of unique values required
        min_std: Minimum standard deviation required

    Returns:
        Dictionary with validation results

    Raises:
        RuntimeError: If validation fails
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Risk scores file not found: {file_path}")

    if split_name is None:
        split_name = file_path.stem

    df = pd.read_csv(file_path)

    if "risk_score" not in df.columns:
        raise ValueError(f"[{split_name}] Missing 'risk_score' column in {file_path}")

    scores = df["risk_score"]

    # Check for constant scores
    assert_non_constant_scores(scores, split_name, min_unique, min_std)

    # Additional validations
    if scores.isna().any():
        raise ValueError(f"[{split_name}] Risk scores contain NaN values")

    if (scores < 0).any() or (scores > 1).any():
        raise ValueError(
            f"[{split_name}] Risk scores out of [0, 1] range: "
            f"min={scores.min():.6f}, max={scores.max():.6f}"
        )

    return {
        "split_name": split_name,
        "file_path": str(file_path),
        "n_samples": len(df),
        "n_unique": scores.nunique(),
        "std": scores.std(),
        "mean": scores.mean(),
        "min": scores.min(),
        "max": scores.max(),
        "valid": True,
    }
