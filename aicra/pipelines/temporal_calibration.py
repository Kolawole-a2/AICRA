"""
Temporal Calibration: Evaluate calibration drift over time.

Fits calibration on validation set from time window T1,
evaluates on later time window T2 to detect temporal drift.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from ..core.calibration import Calibrator
from ..core.data import Dataset
from ..core.evaluation import expected_calibration_error

logger = logging.getLogger(__name__)


def evaluate_temporal_calibration_drift(
    calibrator: Calibrator,
    y_prob_T1: np.ndarray,
    y_true_T1: np.ndarray,
    y_prob_T2: np.ndarray,
    y_true_T2: np.ndarray,
) -> dict:
    """
    Evaluate calibration drift between time windows T1 and T2.

    Args:
        calibrator: Calibrator fitted on T1
        y_prob_T1: Uncalibrated probabilities from T1 (validation set)
        y_true_T1: True labels from T1
        y_prob_T2: Uncalibrated probabilities from T2 (test set)
        y_true_T2: True labels from T2

    Returns:
        Dictionary with calibration metrics for T1 and T2
    """
    # Calibrate probabilities
    y_prob_cal_T1 = calibrator.transform(y_prob_T1)
    y_prob_cal_T2 = calibrator.transform(y_prob_T2)

    # Compute metrics
    brier_T1 = brier_score_loss(y_true_T1, y_prob_cal_T1)
    brier_T2 = brier_score_loss(y_true_T2, y_prob_cal_T2)
    ece_T1 = expected_calibration_error(y_true_T1, y_prob_cal_T1)
    ece_T2 = expected_calibration_error(y_true_T2, y_prob_cal_T2)

    # Drift metrics
    brier_drift = brier_T2 - brier_T1
    ece_drift = ece_T2 - ece_T1
    brier_drift_pct = (brier_drift / brier_T1 * 100) if brier_T1 > 0 else 0.0
    ece_drift_pct = (ece_drift / ece_T1 * 100) if ece_T1 > 0 else 0.0

    return {
        "T1": {
            "brier_score": float(brier_T1),
            "ece": float(ece_T1),
            "n_samples": len(y_true_T1),
        },
        "T2": {
            "brier_score": float(brier_T2),
            "ece": float(ece_T2),
            "n_samples": len(y_true_T2),
        },
        "drift": {
            "brier_drift": float(brier_drift),
            "ece_drift": float(ece_drift),
            "brier_drift_pct": float(brier_drift_pct),
            "ece_drift_pct": float(ece_drift_pct),
        },
        "interpretation": {
            "significant_drift": abs(brier_drift_pct) > 10.0
            or abs(ece_drift_pct) > 10.0,
            "recommendation": "Recalibrate"
            if abs(brier_drift_pct) > 10.0
            else "Monitor",
        },
    }


def rolling_calibration(
    data: Dataset,
    model,
    window_size_days: int = 30,
    calibration_method: str = "isotonic",
) -> dict:
    """
    Maintain rolling calibration over sliding time windows.

    Args:
        data: Dataset with timestamps
        model: Trained model
        window_size_days: Size of calibration window in days
        calibration_method: "isotonic" or "platt"

    Returns:
        Dictionary with calibration artifacts per window
    """
    # Sort by timestamp
    sort_idx = data.timestamps.argsort()
    data_sorted = Dataset(
        features=data.features.iloc[sort_idx].reset_index(drop=True),
        labels=data.labels.iloc[sort_idx].reset_index(drop=True),
        families=data.families.iloc[sort_idx].reset_index(drop=True)
        if data.families is not None
        else None,
        timestamps=data.timestamps.iloc[sort_idx].reset_index(drop=True),
    )

    # Generate predictions
    y_prob = model.predict_proba(data_sorted.features)
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]

    # Sliding window calibration
    window_start = data_sorted.timestamps.min()
    window_end = window_start + pd.Timedelta(days=window_size_days)
    max_time = data_sorted.timestamps.max()

    calibration_windows = []

    while window_end <= max_time:
        # Get window data
        window_mask = (data_sorted.timestamps >= window_start) & (
            data_sorted.timestamps < window_end
        )
        window_data = Dataset(
            features=data_sorted.features[window_mask].reset_index(drop=True),
            labels=data_sorted.labels[window_mask].reset_index(drop=True),
            families=data_sorted.families[window_mask].reset_index(drop=True)
            if data_sorted.families is not None
            else None,
            timestamps=data_sorted.timestamps[window_mask].reset_index(drop=True),
        )

        if len(window_data.features) < 100:  # Skip small windows
            window_start = window_end
            window_end = window_start + pd.Timedelta(days=window_size_days)
            continue

        # Split window into train/val for calibration
        split_idx = int(len(window_data.features) * 0.8)
        y_true_cal_train = window_data.labels.iloc[:split_idx]
        y_true_cal_val = window_data.labels.iloc[split_idx:]
        y_prob_cal_train = y_prob[window_mask][:split_idx]
        y_prob_cal_val = y_prob[window_mask][split_idx:]

        # Fit calibrator
        from ..config import Settings
        from ..pipelines.calibration import CalibrationPipeline

        settings = Settings()
        cal_pipeline = CalibrationPipeline(settings)
        calibrator = cal_pipeline._create_calibrator(calibration_method)
        calibrator.fit(y_prob_cal_train, y_true_cal_train.values)

        # Evaluate
        y_prob_cal = calibrator.transform(y_prob_cal_val)
        brier = brier_score_loss(y_true_cal_val.values, y_prob_cal)
        ece = expected_calibration_error(y_true_cal_val.values, y_prob_cal)

        calibration_windows.append(
            {
                "window_start": str(window_start),
                "window_end": str(window_end),
                "brier_score": float(brier),
                "ece": float(ece),
                "n_samples": len(window_data.features),
            }
        )

        # Slide window
        window_start = window_end
        window_end = window_start + pd.Timedelta(days=window_size_days)

    return {
        "calibration_windows": calibration_windows,
        "window_size_days": window_size_days,
        "method": calibration_method,
    }
