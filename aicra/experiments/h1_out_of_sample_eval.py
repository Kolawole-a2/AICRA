"""
H1 Out-of-Sample Evaluation: Temporal and Out-of-Family Generalization

Evaluates trained H1 model on:
1. Temporal hold-out: Test on time periods strictly after training period
2. Out-of-family: Test on malware families unseen during training
3. Combined: Out-of-family samples from future time periods (strictest test)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from ..core.data import load_ember_2024
from ..core.evaluation import expected_calibration_error

logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    return expected_calibration_error(y_true, y_prob, n_bins=n_bins)


def evaluate_temporal_holdout(
    model_path: Path,
    train_time_end: pd.Timestamp | None,
    test_time_start: pd.Timestamp | None,
    output_dir: Path,
) -> dict:
    """
    Evaluate model on temporal hold-out (strictly future data).

    Args:
        model_path: Path to trained model
        train_time_end: Maximum timestamp in training data
        test_time_start: Minimum timestamp for test (must be > train_time_end)
        output_dir: Directory to save results

    Returns:
        Dictionary with metrics
    """
    logger.info("=" * 80)
    logger.info("Temporal Hold-Out Evaluation")
    logger.info("=" * 80)

    # Load data with time-ordered split
    train_data, test_data = load_ember_2024(
        time_ordered=True,
        train_time_end=train_time_end,
        test_time_start=test_time_start,
    )

    # Verify temporal integrity
    if train_data.timestamps.max() >= test_data.timestamps.min():
        raise ValueError(
            "Temporal leakage detected: train max timestamp >= test min timestamp"
        )

    # Load model
    model = joblib.load(model_path)

    # Generate predictions
    X_test = test_data.features.values
    y_test = test_data.labels.values

    y_prob = model.predict_proba(pd.DataFrame(X_test))
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]

    # Compute metrics
    auroc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    ece = compute_ece(y_test, y_prob)

    # Operational threshold (banking FN≫FP)
    cost_fn, cost_fp = 100.0, 1.0
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    min_cost = float("inf")

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        cost = (cost_fn * fn) + (cost_fp * fp)
        if cost < min_cost:
            min_cost = cost
            best_threshold = t

    # Metrics at optimal threshold
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt, labels=[0, 1]).ravel()

    metrics = {
        "temporal_holdout": {
            "auroc": float(auroc),
            "pr_auc": float(pr_auc),
            "brier_score": float(brier),
            "ece": float(ece),
            "optimal_threshold": float(best_threshold),
            "min_cost": float(min_cost),
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "n_samples": len(y_test),
            "train_time_end": str(train_data.timestamps.max()),
            "test_time_start": str(test_data.timestamps.min()),
        }
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "temporal_holdout_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Temporal hold-out AUROC: {auroc:.4f}, PR-AUC: {pr_auc:.4f}")
    logger.info(f"Optimal threshold: {best_threshold:.4f}, Cost: {min_cost:.2f}")

    return metrics


def evaluate_out_of_family_temporal(
    model_path: Path,
    train_families: set[str],
    train_time_end: pd.Timestamp | None,
    test_time_start: pd.Timestamp | None,
    output_dir: Path,
) -> dict:
    """
    Evaluate on out-of-family samples from future time periods (strictest test).

    Args:
        model_path: Path to trained model
        train_families: Set of families seen during training
        train_time_end: Maximum timestamp in training
        test_time_start: Minimum timestamp for test
        output_dir: Directory to save results

    Returns:
        Dictionary with metrics
    """
    logger.info("=" * 80)
    logger.info("Out-of-Family + Temporal Evaluation (Strictest Test)")
    logger.info("=" * 80)

    # Load data
    train_data, test_data = load_ember_2024(
        time_ordered=True,
        train_time_end=train_time_end,
        test_time_start=test_time_start,
    )

    # Filter test to out-of-family + future time
    oof_mask = ~test_data.families.isin(train_families)
    temporal_mask = (
        test_data.timestamps >= test_time_start
        if test_time_start
        else pd.Series([True] * len(test_data.timestamps))
    )
    combined_mask = oof_mask & temporal_mask

    if combined_mask.sum() == 0:
        logger.warning("No out-of-family + temporal samples found")
        return {}

    oof_test = Dataset(
        features=test_data.features[combined_mask].reset_index(drop=True),
        labels=test_data.labels[combined_mask].reset_index(drop=True),
        families=test_data.families[combined_mask].reset_index(drop=True),
        timestamps=test_data.timestamps[combined_mask].reset_index(drop=True),
    )

    # Load model and evaluate
    model = joblib.load(model_path)
    X_test = oof_test.features.values
    y_test = oof_test.labels.values

    y_prob = model.predict_proba(pd.DataFrame(X_test))
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]

    # Compute metrics
    auroc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    ece = compute_ece(y_test, y_prob)

    metrics = {
        "oof_temporal": {
            "auroc": float(auroc),
            "pr_auc": float(pr_auc),
            "brier_score": float(brier),
            "ece": float(ece),
            "n_samples": len(y_test),
            "n_families": oof_test.families.nunique(),
            "families": oof_test.families.unique().tolist(),
        }
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "oof_temporal_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        f"OOF+Temporal AUROC: {auroc:.4f}, PR-AUC: {pr_auc:.4f}, n_samples={len(y_test)}"
    )

    return metrics


def main():
    """Main entry point for out-of-sample evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="H1 Out-of-Sample Evaluation")
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/H1_out_of_sample"),
        help="Output directory",
    )
    parser.add_argument(
        "--train-time-end", type=str, help="Training end timestamp (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--test-time-start", type=str, help="Test start timestamp (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--train-families",
        nargs="+",
        help="List of families seen during training (for OOF eval)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train_time_end = pd.Timestamp(args.train_time_end) if args.train_time_end else None
    test_time_start = (
        pd.Timestamp(args.test_time_start) if args.test_time_start else None
    )

    # Run temporal evaluation
    temporal_results = evaluate_temporal_holdout(
        args.model, train_time_end, test_time_start, args.output
    )

    # For OOF+temporal, use provided train families or extract from H1 results
    if args.train_families:
        train_families = set(args.train_families)
        oof_temporal_results = evaluate_out_of_family_temporal(
            args.model, train_families, train_time_end, test_time_start, args.output
        )
    else:
        logger.warning(
            "--train-families not provided, skipping OOF+temporal evaluation"
        )

    logger.info("Out-of-sample evaluation complete")


if __name__ == "__main__":
    main()
