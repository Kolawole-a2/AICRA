#!/usr/bin/env python3
"""
Generate Plots from Canonical H1/H2 Experiment Test Set Predictions

This script generates plots directly from the canonical H1/H2 experiment test set predictions,
ensuring plots match the canonical results exactly.

Outputs:
- results/H1_classification/plots/roc.png
- results/H1_classification/plots/pr.png
- results/H1_classification/plots/confusion.png
- results/H2_calibration_thresholds/plots/reliability.png (calibrated)
- results/H2_calibration_thresholds/plots/reliability_uncalibrated.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from aicra.core.data import Dataset, load_ember_2024
from aicra.pipelines.calibration import CalibrationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, auroc: float, out_path: Path
) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUROC = {auroc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Canonical H1 Experiment", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ROC curve to {out_path}")


def plot_pr_curve(
    y_true: np.ndarray, y_prob: np.ndarray, pr_auc: float, out_path: Path
) -> None:
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (PR-AUC = {pr_auc:.4f})", linewidth=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(
        "Precision-Recall Curve - Canonical H1 Experiment",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved PR curve to {out_path}")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float, out_path: Path
) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (T = {threshold:.4f})", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Benign (0)", "Ransomware (1)"], fontsize=11)
    plt.yticks(tick_marks, ["Benign (0)", "Ransomware (1)"], fontsize=11)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12,
            fontweight="bold",
        )

    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix to {out_path}")


def plot_reliability_diagram(
    y_true: np.ndarray, y_prob: np.ndarray, title: str, out_path: Path
) -> None:
    """Plot reliability (calibration) diagram."""
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="uniform"
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        prob_pred, prob_true, "s-", label="Calibration Curve", linewidth=2, markersize=8
    )
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=1)
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Fraction of Positives", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved reliability diagram to {out_path}")


def generate_h1_plots(repo_root: Path) -> None:
    """Generate plots for H1 experiment from canonical test set predictions."""
    logger.info("=" * 80)
    logger.info("Generating H1 Canonical Plots")
    logger.info("=" * 80)

    # Load H1 results to get threshold and check evaluation mode
    h1_results_path = (
        repo_root / "results" / "H1_classification" / "H1_full_results.json"
    )
    if not h1_results_path.exists():
        logger.error(f"H1 results not found: {h1_results_path}")
        return

    with open(h1_results_path) as f:
        h1_results = json.load(f)

    metrics = h1_results["metrics"]
    threshold = metrics["operational_threshold"]
    use_multi_split = h1_results.get("evaluation_mode") == "multi_split"

    # Load data and model (same as H1 experiment)
    logger.info("Loading EMBER-2024 data...")
    train_data, test_data = load_ember_2024(time_ordered=True)

    logger.info("Loading H1 model...")
    model_path = repo_root / "models" / "h1_lgbm.joblib"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    model = joblib.load(model_path)

    plots_dir = repo_root / "results" / "H1_classification" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if use_multi_split:
        # Generate plots for each split
        logger.info("Multi-split evaluation detected - generating plots for all splits")
        per_split_results = metrics.get("per_split_results", [])

        if not per_split_results:
            logger.warning(
                "Multi-split mode but no per-split results found, using full test set"
            )
            use_multi_split = False

        if use_multi_split:
            # Create splits from test data (same logic as H1 experiment)
            n_test = len(test_data.features)
            splits = {
                "full_ember": (0, n_test),
                "main": (0, min(10_000, n_test)),
                "small_ember": (0, min(2_000, n_test)),
                "smoke_test": (0, min(200, n_test)),
            }

            for split_name, (start_idx, end_idx) in splits.items():
                logger.info(f"Generating plots for split: {split_name}")

                # Get split data
                split_features = test_data.features.iloc[start_idx:end_idx].values
                split_labels = test_data.labels.iloc[start_idx:end_idx].values

                # Generate predictions
                X_split_df = pd.DataFrame(split_features)
                prob_split = model.predict_proba(X_split_df)
                if prob_split.ndim == 1:
                    y_prob = prob_split
                else:
                    y_prob = prob_split[:, 1]

                y_pred = (y_prob >= threshold).astype(int)

                # Compute metrics
                auroc = roc_auc_score(split_labels, y_prob)
                pr_auc = average_precision_score(split_labels, y_prob)

                logger.info(f"  {split_name}: AUROC={auroc:.4f}, PR-AUC={pr_auc:.4f}")

                # Generate plots for this split
                split_plots_dir = plots_dir / split_name
                split_plots_dir.mkdir(parents=True, exist_ok=True)

                plot_roc_curve(split_labels, y_prob, auroc, split_plots_dir / "roc.png")
                plot_pr_curve(split_labels, y_prob, pr_auc, split_plots_dir / "pr.png")
                plot_confusion_matrix(
                    split_labels, y_pred, threshold, split_plots_dir / "confusion.png"
                )

            logger.info("✓ H1 plots generated for all splits")
            return

    # Single-split mode: generate plots for full test set
    logger.info("Generating plots for full test set...")
    X_test = test_data.features.values
    y_true = test_data.labels.values

    # Handle both 1D and 2D outputs
    prob_test = model.predict_proba(X_test)
    if prob_test.ndim == 1:
        y_prob = prob_test
    else:
        y_prob = prob_test[:, 1]

    y_pred = (y_prob >= threshold).astype(int)

    # Compute metrics
    auroc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    logger.info(f"Test set metrics: AUROC={auroc:.4f}, PR-AUC={pr_auc:.4f}")

    plot_roc_curve(y_true, y_prob, auroc, plots_dir / "roc.png")
    plot_pr_curve(y_true, y_prob, pr_auc, plots_dir / "pr.png")
    plot_confusion_matrix(y_true, y_pred, threshold, plots_dir / "confusion.png")

    logger.info("✓ H1 plots generated successfully")


def generate_h2_plots(repo_root: Path) -> None:
    """Generate plots for H2 experiment from canonical test set predictions."""
    logger.info("=" * 80)
    logger.info("Generating H2 Canonical Plots")
    logger.info("=" * 80)

    # Load H2 results
    h2_results_path = (
        repo_root / "results" / "H2_calibration_thresholds" / "H2_full_results.json"
    )
    if not h2_results_path.exists():
        logger.error(f"H2 results not found: {h2_results_path}")
        return

    with open(h2_results_path) as f:
        h2_results = json.load(f)

    metrics = h2_results.get("metrics", {})
    use_multi_split = h2_results.get("evaluation_mode") == "multi_split"

    # Load data and model (same as H2 experiment)
    logger.info("Loading EMBER-2024 data...")
    train_data, test_data = load_ember_2024(time_ordered=True)

    logger.info("Loading H1 model...")
    model_path = repo_root / "models" / "h1_lgbm.joblib"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    model = joblib.load(model_path)

    # Calibrate predictions (same as H2 experiment)
    logger.info("Calibrating predictions...")
    from aicra.config import Settings

    settings = Settings()
    calibration_pipeline = CalibrationPipeline(settings)

    # Split train data for calibration (same as H2)
    from sklearn.model_selection import train_test_split

    X_train = train_data.features.values
    y_train = train_data.labels.values

    X_train_cal, X_val_cal, y_train_cal, y_val_cal = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # Generate predictions for calibration
    X_train_cal_df = pd.DataFrame(X_train_cal)
    X_val_cal_df = pd.DataFrame(X_val_cal)

    prob_train_cal = model.predict_proba(X_train_cal_df)
    prob_val_cal = model.predict_proba(X_val_cal_df)

    if prob_train_cal.ndim == 1:
        y_prob_train_cal = prob_train_cal
        y_prob_val_cal = prob_val_cal
    else:
        y_prob_train_cal = prob_train_cal[:, 1]
        y_prob_val_cal = prob_val_cal[:, 1]

    # Fit calibrator using pipeline
    calibrator = calibration_pipeline.run(
        train_data=Dataset(
            features=pd.DataFrame(X_train_cal),
            labels=pd.Series(y_train_cal),
        ),
        val_data=Dataset(
            features=pd.DataFrame(X_val_cal),
            labels=pd.Series(y_val_cal),
        ),
        y_prob_train=y_prob_train_cal,
        y_prob_val=y_prob_val_cal,
        method="auto",
        skip_mlflow=True,
    )

    plots_dir = repo_root / "results" / "H2_calibration_thresholds" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if use_multi_split:
        # Generate plots for each split
        logger.info("Multi-split evaluation detected - generating plots for all splits")
        per_split_results = metrics.get("per_split_results", [])

        if not per_split_results:
            logger.warning(
                "Multi-split mode but no per-split results found, using full test set"
            )
            use_multi_split = False

        if use_multi_split:
            # Create splits from test data (same logic as H2 experiment)
            n_test = len(test_data.features)
            splits = {
                "full_ember": (0, n_test),
                "main": (0, min(10_000, n_test)),
                "small_ember": (0, min(2_000, n_test)),
                "smoke_test": (0, min(200, n_test)),
            }

            for split_name, (start_idx, end_idx) in splits.items():
                logger.info(f"Generating plots for split: {split_name}")

                # Get split data
                split_features = test_data.features.iloc[start_idx:end_idx].values
                split_labels = test_data.labels.iloc[start_idx:end_idx].values

                # Generate uncalibrated predictions
                X_split_df = pd.DataFrame(split_features)
                prob_split = model.predict_proba(X_split_df)
                if prob_split.ndim == 1:
                    y_prob_uncal = prob_split
                else:
                    y_prob_uncal = prob_split[:, 1]

                # Calibrate predictions
                y_prob_cal = calibrator.transform(y_prob_uncal)

                # Generate plots for this split
                split_plots_dir = plots_dir / split_name
                split_plots_dir.mkdir(parents=True, exist_ok=True)

                plot_reliability_diagram(
                    split_labels,
                    y_prob_uncal,
                    f"Reliability Diagram - Uncalibrated ({split_name})",
                    split_plots_dir / "reliability_uncalibrated.png",
                )
                plot_reliability_diagram(
                    split_labels,
                    y_prob_cal,
                    f"Reliability Diagram - Calibrated ({split_name})",
                    split_plots_dir / "reliability_calibrated.png",
                )

            logger.info("✓ H2 plots generated for all splits")
            return

    # Single-split mode: generate plots for full test set
    logger.info("Generating plots for full test set...")
    X_test = test_data.features.values
    y_true = test_data.labels.values

    prob_test = model.predict_proba(X_test)
    if prob_test.ndim == 1:
        y_prob_uncal = prob_test
    else:
        y_prob_uncal = prob_test[:, 1]

    # Calibrate test predictions
    y_prob_cal = calibrator.transform(y_prob_uncal)

    plot_reliability_diagram(
        y_true,
        y_prob_uncal,
        "Reliability Diagram - Uncalibrated (Canonical H2)",
        plots_dir / "reliability_uncalibrated.png",
    )
    plot_reliability_diagram(
        y_true,
        y_prob_cal,
        "Reliability Diagram - Calibrated (Canonical H2)",
        plots_dir / "reliability_calibrated.png",
    )

    logger.info("✓ H2 plots generated successfully")


def main() -> None:
    """Main entry point."""
    repo_root = Path(__file__).parent.parent

    logger.info("=" * 80)
    logger.info("Generating Plots from Canonical H1/H2 Experiments")
    logger.info("=" * 80)

    # Generate H1 plots
    generate_h1_plots(repo_root)

    # Generate H2 plots
    generate_h2_plots(repo_root)

    logger.info("=" * 80)
    logger.info("All plots generated successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
