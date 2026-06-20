#!/usr/bin/env python3
"""
H1/H2 Rebuild – Phase 3: Generate Plots and Metrics Per Split

This script reads the per-split `risk_scores.csv` files produced by
`scripts/h1h2_rebuild/train_and_score.py` and generates:

- ROC curve
- Precision-Recall curve
- Confusion matrix (at cost-sensitive threshold)
- Reliability (calibration) diagram
- Metrics JSON per split:
    - AUROC, PR-AUC
    - Precision, Recall, F1
    - Brier score, ECE
    - Confusion matrix entries

Outputs:
- results/h1h2_rebuild/<split>/plots/*.png
- results/h1h2_rebuild/<split>/metrics.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def plot_roc_curve(y_true, y_prob, auroc, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUROC={auroc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_pr_curve(y_true, y_prob, pr_auc, out_path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"PR (AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_confusion(y_true, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Benign (0)", "Ransomware (1)"])
    plt.yticks(tick_marks, ["Benign (0)", "Ransomware (1)"])

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_reliability(y_true, y_prob, out_path: Path) -> None:
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=15, strategy="uniform"
    )
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Frequency")
    plt.title("Reliability Diagram")
    plt.legend(loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def generate_metrics_for_split(split_name: str, repo_root: Path) -> dict:
    """Generate metrics and plots for a single split."""
    logger.info("=" * 80)
    logger.info(f"Generating plots and metrics for split: {split_name}")
    logger.info("=" * 80)

    # Load risk_scores
    risk_path = repo_root / "results" / "h1h2_rebuild" / split_name / "risk_scores.csv"
    if not risk_path.exists():
        raise FileNotFoundError(
            f"risk_scores.csv not found for split {split_name}: {risk_path}"
        )

    df = pd.read_csv(risk_path)
    y_true = df["true_label"].values
    y_prob = df["p_ransomware"].values

    logger.info(f"Loaded {len(df)} samples for {split_name}")
    logger.info(
        f"  Ransomware (1): {y_true.sum()}, Benign (0): {len(y_true) - y_true.sum()}"
    )

    # Basic metrics
    auroc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)

    # Cost-sensitive threshold (reuse p_ransomware >= 0.5 as default view)
    # For metrics we will re-derive the decision boundary at 0.5
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "split": split_name,
        "n_samples": int(len(df)),
        "n_ransomware": int(y_true.sum()),
        "n_benign": int(len(y_true) - y_true.sum()),
        "threshold": float(threshold),
        "auroc": float(auroc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "ece": float(ece),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    # Output directory
    out_dir = repo_root / "results" / "h1h2_rebuild" / split_name
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plots
    plot_roc_curve(y_true, y_prob, auroc, plots_dir / "roc.png")
    plot_pr_curve(y_true, y_prob, pr_auc, plots_dir / "pr.png")
    plot_confusion(y_true, y_pred, plots_dir / "confusion.png")
    plot_reliability(y_true, y_prob, plots_dir / "reliability.png")

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"✓ Saved metrics to {metrics_path}")
    logger.info("=" * 80)
    logger.info(
        f"{split_name}: AUROC={auroc:.4f}, PR-AUC={pr_auc:.4f}, "
        f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, "
        f"Brier={brier:.4f}, ECE={ece:.4f}"
    )
    logger.info("=" * 80)

    return metrics


def main() -> None:
    repo_root = Path(__file__).parent.parent.parent
    splits = ["smoke_test", "small_ember", "main", "full_ember"]

    all_metrics: dict[str, dict] = {}
    for split in splits:
        try:
            m = generate_metrics_for_split(split, repo_root)
            all_metrics[split] = m
        except FileNotFoundError as e:
            logger.warning(f"Skipping split {split}: {e}")
            continue

    combined_path = repo_root / "results" / "h1h2_rebuild" / "metrics_summary.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"✓ Saved combined metrics summary to {combined_path}")


if __name__ == "__main__":
    main()
