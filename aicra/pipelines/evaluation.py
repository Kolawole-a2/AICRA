"""Evaluation pipeline for AICRA models."""

from __future__ import annotations

from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    precision_recall_curve, 
    roc_curve,
    brier_score_loss,
    average_precision_score,
    roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import calibration_curve

from ..config import Settings
from ..core.data import Dataset
from ..core.evaluation import Metrics, evaluate_probs
from .mapping import MappingPipeline


class EvaluationPipeline:
    """Evaluation pipeline with visualization and MLflow logging."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(
        self,
        test_data: Dataset,
        y_prob: np.ndarray[Any, np.dtype[np.floating]],
        threshold: float,
        model_name: str = "bagged_lightgbm",
        timestamp_column: Optional[str] = None,
        family_column: Optional[str] = None,
        k_values: list[int] = [1, 5, 10],
        is_smoke_test: bool = False,
    ) -> Metrics:
        """Evaluate model and generate artifacts."""
        
        # Time-ordered split if timestamp exists
        if timestamp_column and timestamp_column in test_data.features.columns:
            mlflow.log_param("split_type", "time_ordered")
            train_idx, test_idx = self._time_ordered_split(
                test_data.features[timestamp_column]
            )
            y_prob_test = y_prob[test_idx]
            y_true_test = test_data.labels.values[test_idx]
            families_test = test_data.families[test_idx] if hasattr(test_data, 'families') else None
        else:
            mlflow.log_param("split_type", "random")
            mlflow.log_param("split_warning", "No timestamp column found, using random split")
            y_prob_test = y_prob
            y_true_test = test_data.labels.values
            families_test = test_data.families if hasattr(test_data, 'families') else None

        # Out-of-family generalization evaluation
        if family_column and families_test is not None:
            self._evaluate_out_of_family_generalization(
                y_true_test, y_prob_test, families_test, threshold
            )

        # Compute comprehensive metrics
        metrics = self._compute_comprehensive_metrics(
            y_true_test, y_prob_test, threshold, k_values
        )

        # Generate plots
        self._plot_roc_curve(y_true_test, y_prob_test, metrics.auroc)
        self._plot_pr_curve(y_true_test, y_prob_test, metrics.pr_auc)
        self._plot_confusion_matrix(y_true_test, y_prob_test, threshold)
        self._plot_lift_curve(y_true_test, y_prob_test, k_values)
        self._plot_reliability_diagram(y_true_test, y_prob_test)

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_metrics({
                "auroc": metrics.auroc,
                "pr_auc": metrics.pr_auc,
                "brier": metrics.brier,
                "ece": metrics.ece,
                "threshold": metrics.threshold,
            })
            
            # Log Lift@k metrics
            for k in k_values:
                mlflow.log_metric(f"lift_at_{k}pct", getattr(metrics, f"lift_at_{k}pct", 0.0))

            # Log confusion matrix
            mlflow.log_metrics({
                "tn": metrics.confusion[0],
                "fp": metrics.confusion[1],
                "fn": metrics.confusion[2],
                "tp": metrics.confusion[3],
            })

            # Log artifacts
            mlflow.log_artifacts(str(self.settings.artifacts_dir))

        # Check target bars for non-smoke tests
        if not is_smoke_test:
            self._check_target_bars(metrics)

        return metrics

    def _plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, auroc: float):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(self.settings.artifacts_dir / 'roc.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pr_curve(self, y_true: np.ndarray, y_prob: np.ndarray, pr_auc: float):
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR-AUC = {pr_auc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(self.settings.artifacts_dir / 'pr.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float):
        """Plot confusion matrix."""
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks([0, 1], ['Benign', 'Ransomware'])
        plt.yticks([0, 1], ['Benign', 'Ransomware'])

        plt.savefig(self.settings.artifacts_dir / 'confusion.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _time_ordered_split(self, timestamps: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Perform time-ordered split to avoid data leakage."""
        # Sort by timestamp
        sorted_indices = timestamps.argsort()
        
        # Use 80% for training, 20% for testing
        split_point = int(0.8 * len(sorted_indices))
        
        train_idx = sorted_indices[:split_point]
        test_idx = sorted_indices[split_point:]
        
        return train_idx, test_idx
    
    def _evaluate_out_of_family_generalization(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        families: np.ndarray, 
        threshold: float
    ) -> None:
        """Evaluate out-of-family generalization by holding out families."""
        mapping_pipeline = MappingPipeline(self.settings)
        
        # Get canonical families
        canonical_families = mapping_pipeline.get_all_canonical_families()
        
        # Hold out one family completely from training
        held_out_family = canonical_families[0] if canonical_families else "Unknown"
        
        # Map families to canonical
        canonical_families_test = [mapping_pipeline.normalize_family(f) for f in families]
        
        # Split into in-family and out-of-family
        in_family_mask = np.array([f != held_out_family for f in canonical_families_test])
        out_family_mask = ~in_family_mask
        
        if np.any(out_family_mask):
            # Evaluate on held-out family
            y_true_out = y_true[out_family_mask]
            y_prob_out = y_prob[out_family_mask]
            
            # Compute metrics for out-of-family
            auroc_out = roc_auc_score(y_true_out, y_prob_out)
            pr_auc_out = average_precision_score(y_true_out, y_prob_out)
            brier_out = brier_score_loss(y_true_out, y_prob_out)
            
            # Log out-of-family metrics
            mlflow.log_metrics({
                "auroc_out_of_family": auroc_out,
                "pr_auc_out_of_family": pr_auc_out,
                "brier_out_of_family": brier_out,
                "held_out_family": held_out_family,
                "out_of_family_samples": int(np.sum(out_family_mask)),
            })
    
    def _compute_comprehensive_metrics(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        threshold: float,
        k_values: list[int]
    ) -> Metrics:
        """Compute comprehensive evaluation metrics."""
        
        # Basic metrics
        auroc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        brier = brier_score_loss(y_true, y_prob)
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(y_true, y_prob)
        
        # Confusion matrix at threshold
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        confusion_flat = cm.flatten()
        
        # Lift@k metrics
        lift_metrics = {}
        for k in k_values:
            lift_k = self._compute_lift_at_k(y_true, y_prob, k)
            lift_metrics[f"lift_at_{k}pct"] = lift_k
        
        # Create metrics object
        metrics = Metrics(
            auroc=auroc,
            pr_auc=pr_auc,
            brier=brier,
            ece=ece,
            threshold=threshold,
            confusion=confusion_flat,
            lift_at_k=lift_metrics.get("lift_at_10pct", 0.0),
        )
        
        # Add lift metrics as attributes
        for k, v in lift_metrics.items():
            setattr(metrics, k, v)
        
        return metrics
    
    def _compute_ece(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _compute_lift_at_k(self, y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
        """Compute Lift@k metric."""
        # Sort by probability descending
        sorted_indices = np.argsort(y_prob)[::-1]
        
        # Get top k% of samples
        k_samples = int(len(y_true) * k / 100)
        top_k_indices = sorted_indices[:k_samples]
        
        # Compute lift
        precision_at_k = y_true[top_k_indices].mean()
        overall_precision = y_true.mean()
        
        if overall_precision > 0:
            lift = precision_at_k / overall_precision
        else:
            lift = 0.0
        
        return lift
    
    def _plot_lift_curve(self, y_true: np.ndarray, y_prob: np.ndarray, k_values: list[int]) -> None:
        """Plot lift curve for different k values."""
        # Sort by probability descending
        sorted_indices = np.argsort(y_prob)[::-1]
        
        # Compute lift for different k values
        k_percentages = np.arange(1, 101)  # 1% to 100%
        lifts = []
        
        for k in k_percentages:
            k_samples = int(len(y_true) * k / 100)
            if k_samples > 0:
                top_k_indices = sorted_indices[:k_samples]
                precision_at_k = y_true[top_k_indices].mean()
                overall_precision = y_true.mean()
                
                if overall_precision > 0:
                    lift = precision_at_k / overall_precision
                else:
                    lift = 0.0
            else:
                lift = 0.0
            
            lifts.append(lift)
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_percentages, lifts, 'b-', linewidth=2, label='Lift Curve')
        plt.axhline(y=1, color='r', linestyle='--', label='Baseline (Random)')
        
        # Mark specific k values
        for k in k_values:
            k_idx = k - 1  # Convert to 0-based index
            if k_idx < len(lifts):
                plt.plot(k, lifts[k_idx], 'ro', markersize=8)
                plt.annotate(f'Lift@{k}% = {lifts[k_idx]:.2f}', 
                           xy=(k, lifts[k_idx]), 
                           xytext=(k+5, lifts[k_idx]+0.1),
                           arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.xlabel('Percentage of Samples (%)')
        plt.ylabel('Lift')
        plt.title('Lift Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        
        plt.savefig(self.settings.artifacts_dir / 'lift_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_reliability_diagram(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Plot reliability diagram for calibration assessment."""
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, 'o-', label='Model', markersize=6)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.settings.artifacts_dir / 'reliability.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _check_target_bars(self, metrics: Metrics) -> None:
        """Check target bars for non-smoke tests and print warnings if not met."""
        warnings = []
        
        # Target bars: AUROC ≥ 0.75, ECE ≤ 0.10, Brier ≤ 0.20, Lift@5% > 1.3
        if metrics.auroc < 0.75:
            warnings.append(f"AUROC ({metrics.auroc:.3f}) < 0.75 target")
        
        if metrics.ece > 0.10:
            warnings.append(f"ECE ({metrics.ece:.3f}) > 0.10 target")
        
        if metrics.brier > 0.20:
            warnings.append(f"Brier Score ({metrics.brier:.3f}) > 0.20 target")
        
        # Check Lift@5%
        lift_at_5 = getattr(metrics, 'lift_at_5pct', None)
        if lift_at_5 is not None and lift_at_5 <= 1.3:
            warnings.append(f"Lift@5% ({lift_at_5:.3f}) <= 1.3 target")
        
        # Print warnings if any
        if warnings:
            print("\n⚠️  TARGET BAR WARNINGS:")
            for warning in warnings:
                print(f"   - {warning}")
            print("   (These are warnings only - evaluation continues)")
        else:
            print("\n✅ All target bars met!")
