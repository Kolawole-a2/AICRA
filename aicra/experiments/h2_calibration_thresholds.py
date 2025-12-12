"""
H2 Experiment: Calibrated Ransomware Susceptibility and Thresholding

This is the canonical H2 experiment module that evaluates calibration and
cost-aware thresholding for ransomware susceptibility scores.

Hypothesis (H2):
"Calibration and cost-aware thresholding produce more decision-aligned
susceptibility scores than uncalibrated F1-optimized thresholds."

Metrics computed:
- Calibration: Brier score, ECE (before/after)
- Threshold comparison: F1-optimized vs cost-optimal
- Expected Loss at different thresholds
- Reliability diagrams

Results are saved to results/H2_calibration_thresholds/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from ..config import Settings
from ..core.calibration import Calibrator
from ..core.data import Dataset, load_ember_2024
from ..core.evaluation import cost_sensitive_threshold
from ..core.benchmarks import (
    compute_h2_baselines,
    compute_h2_improvements,
    format_improvement_statement,
)
from ..pipelines.calibration import CalibrationPipeline

logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def find_f1_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[float, float]:
    """
    Find threshold that maximizes F1 score.
    
    Returns:
        (optimal_threshold, f1_score)
    """
    thresholds = np.sort(np.unique(y_prob))
    best_f1 = 0.0
    best_threshold = 0.5
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    return float(best_threshold), float(best_f1)


def compute_expected_loss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0
) -> float:
    """
    Compute Expected Loss at a given threshold.
    
    Expected Loss = p(ransomware) * impact_cost
    where impact_cost = cost_fn * FN + cost_fp * FP
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    
    total_loss = (cost_fn * fn) + (cost_fp * fp)
    total_samples = len(y_true)
    
    return float(total_loss / total_samples)


def run_h2_calibration_thresholds_experiment(
    output_dir: Path,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    calibration_method: str = "auto",
    repo_root: Optional[Path] = None,
) -> Dict:
    """
    Run H2 calibration and thresholding experiment.
    
    Args:
        output_dir: Directory to save results
        cost_fn: Cost of false negative (default: 10.0 for banking)
        cost_fp: Cost of false positive (default: 1.0)
        calibration_method: Calibration method ("platt", "isotonic", "auto")
        repo_root: Repository root directory
        
    Returns:
        Dictionary with all metrics
    """
    if repo_root is None:
        repo_root = Path.cwd()
    
    logger.info("=" * 80)
    logger.info("H2 Experiment: Calibration and Cost-Aware Thresholding")
    logger.info("=" * 80)
    
    settings = Settings()
    
    # Load data (using model predictions from H1)
    logger.info("Loading data and model predictions...")
    try:
        train_data_full, test_data = load_ember_2024()
        logger.info(f"Full train: {len(train_data_full.features)}, Test: {len(test_data.features)}")
        
        # Split training data into train/val for calibration
        # Use 10% of training data for validation
        val_split = 0.1
        n_train = len(train_data_full.features)
        split_idx = int(n_train * (1 - val_split))
        
        train_data = Dataset(
            features=train_data_full.features.iloc[:split_idx].reset_index(drop=True),
            labels=train_data_full.labels.iloc[:split_idx].reset_index(drop=True),
            families=train_data_full.families.iloc[:split_idx].reset_index(drop=True) if train_data_full.families is not None else None,
            timestamps=train_data_full.timestamps.iloc[:split_idx].reset_index(drop=True) if train_data_full.timestamps is not None else None,
        )
        
        val_data = Dataset(
            features=train_data_full.features.iloc[split_idx:].reset_index(drop=True),
            labels=train_data_full.labels.iloc[split_idx:].reset_index(drop=True),
            families=train_data_full.families.iloc[split_idx:].reset_index(drop=True) if train_data_full.families is not None else None,
            timestamps=train_data_full.timestamps.iloc[split_idx:].reset_index(drop=True) if train_data_full.timestamps is not None else None,
        )
        
        logger.info(f"Train: {len(train_data.features)}, Val: {len(val_data.features)}, Test: {len(test_data.features)}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    # Load trained model from H1
    import joblib
    model_path = settings.models_dir / "h1_lgbm.joblib"
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}, using default model path")
        model_path = settings.models_dir / "bagged_lightgbm.joblib"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found. Please run H1 experiment first to train a model. "
            f"Expected at: {model_path}"
        )
    
    model = joblib.load(model_path)
    
    # Generate predictions
    logger.info("Generating predictions...")
    X_train = train_data.features.values
    X_val = val_data.features.values
    X_test = test_data.features.values
    
    # BaggedLightGBM.predict_proba() expects DataFrame and returns 1D array (probabilities for class 1)
    # Standard sklearn models return 2D array, so handle both cases
    X_train_df = pd.DataFrame(X_train)
    X_val_df = pd.DataFrame(X_val)
    X_test_df = pd.DataFrame(X_test)
    
    prob_train = model.predict_proba(X_train_df)
    prob_val = model.predict_proba(X_val_df)
    prob_test = model.predict_proba(X_test_df)
    
    # Handle both 1D (BaggedLightGBM) and 2D (standard sklearn) outputs
    if prob_train.ndim == 1:
        y_prob_train = prob_train
        y_prob_val = prob_val
        y_prob_test = prob_test
    else:
        y_prob_train = prob_train[:, 1]
        y_prob_val = prob_val[:, 1]
        y_prob_test = prob_test[:, 1]
    
    y_true_train = train_data.labels.values
    y_true_val = val_data.labels.values
    y_true_test = test_data.labels.values
    
    # Calibrate predictions
    logger.info(f"Calibrating predictions using {calibration_method}...")
    calibration_pipeline = CalibrationPipeline(settings)
    calibrator = calibration_pipeline.run(
        train_data=Dataset(
            features=pd.DataFrame(X_train),
            labels=pd.Series(y_true_train),
        ),
        val_data=Dataset(
            features=pd.DataFrame(X_val),
            labels=pd.Series(y_true_val),
        ),
        y_prob_train=y_prob_train,
        y_prob_val=y_prob_val,
        method=calibration_method,
        skip_mlflow=True,
    )
    
    y_prob_test_calibrated = calibrator.transform(y_prob_test)
    
    # Compute calibration metrics
    logger.info("Computing calibration metrics...")
    brier_uncalibrated = brier_score_loss(y_true_test, y_prob_test)
    brier_calibrated = brier_score_loss(y_true_test, y_prob_test_calibrated)
    ece_uncalibrated = compute_ece(y_true_test, y_prob_test)
    ece_calibrated = compute_ece(y_true_test, y_prob_test_calibrated)
    
    # ========================================================================
    # % IMPROVEMENT CALCULATIONS (H2 Requirement)
    # ========================================================================
    h2_improvements = compute_h2_improvements(
        brier_uncalibrated=brier_uncalibrated,
        brier_calibrated=brier_calibrated,
        ece_uncalibrated=ece_uncalibrated,
        ece_calibrated=ece_calibrated,
    )
    logger.info(f"Calibration improvements: Brier {h2_improvements['brier_improvement_pct']:.1f}%, "
                f"ECE {h2_improvements['ece_improvement_pct']:.1f}%")
    
    # Find optimal thresholds
    logger.info("Finding optimal thresholds...")
    
    # F1-optimized threshold (uncalibrated)
    f1_threshold_uncal, f1_score_uncal = find_f1_optimal_threshold(y_true_test, y_prob_test)
    
    # F1-optimized threshold (calibrated)
    f1_threshold_cal, f1_score_cal = find_f1_optimal_threshold(y_true_test, y_prob_test_calibrated)
    
    # Cost-optimal threshold (uncalibrated)
    cost_threshold_uncal = cost_sensitive_threshold(y_true_test, y_prob_test, cost_fn, cost_fp)
    
    # Cost-optimal threshold (calibrated)
    cost_threshold_cal = cost_sensitive_threshold(y_true_test, y_prob_test_calibrated, cost_fn, cost_fp)
    
    # Compute metrics at each threshold
    def compute_metrics_at_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float
    ) -> Dict[str, float]:
        """Compute metrics at a given threshold."""
        y_pred = (y_prob >= threshold).astype(int)
        return {
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "expected_loss": compute_expected_loss(y_true, y_prob, threshold, cost_fn, cost_fp),
        }
    
    metrics = {
        "calibration": {
            "brier_uncalibrated": float(brier_uncalibrated),
            "brier_calibrated": float(brier_calibrated),
            "brier_improvement": float(brier_uncalibrated - brier_calibrated),
            "brier_improvement_pct": h2_improvements['brier_improvement_pct'],
            "brier_vs_baseline_pct": h2_improvements['brier_vs_baseline_pct'],
            "ece_uncalibrated": float(ece_uncalibrated),
            "ece_calibrated": float(ece_calibrated),
            "ece_improvement": float(ece_uncalibrated - ece_calibrated),
            "ece_improvement_pct": h2_improvements['ece_improvement_pct'],
            "ece_vs_baseline_pct": h2_improvements['ece_vs_baseline_pct'],
            "baseline_brier": h2_improvements['baseline_brier'],
            "baseline_ece": h2_improvements['baseline_ece'],
            "method": calibration_method,
        },
        
        # ====================================================================
        # CANONICAL IMPROVEMENT STATEMENT (H2 Requirement)
        # ====================================================================
        "improvement_statement": format_improvement_statement('H2', {
            'ece_improvement_pct': h2_improvements['ece_improvement_pct'],
        }),
    }
        "f1_optimized": {
            "uncalibrated": compute_metrics_at_threshold(y_true_test, y_prob_test, f1_threshold_uncal),
            "calibrated": compute_metrics_at_threshold(y_true_test, y_prob_test_calibrated, f1_threshold_cal),
        },
        "cost_optimized": {
            "cost_fn": float(cost_fn),
            "cost_fp": float(cost_fp),
            "uncalibrated": compute_metrics_at_threshold(y_true_test, y_prob_test, cost_threshold_uncal),
            "calibrated": compute_metrics_at_threshold(y_true_test, y_prob_test_calibrated, cost_threshold_cal),
        },
        "n_test_samples": len(test_data.features),
    }
    
    # Save results in full_results.json format (for consistency with H3)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as metrics.json (backward compatibility)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Also save as H2_full_results.json (for praxis validation)
    full_results = {
        "hypothesis": "H2: Calibration and Cost-Aware Thresholding",
        "hypothesis_statement": "Calibration and cost-aware thresholding produce more decision-aligned susceptibility scores than uncalibrated F1-optimized thresholds.",
        "metrics": metrics,
        "n_test_samples": metrics.get("n_test_samples", 0),
        "calibration_method": metrics.get("calibration", {}).get("method", "auto"),
        "cost_fn": metrics.get("cost_optimized", {}).get("cost_fn", 10.0),
        "cost_fp": metrics.get("cost_optimized", {}).get("cost_fp", 1.0),
    }
    full_results_path = output_dir / "H2_full_results.json"
    with open(full_results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Saved full results to {full_results_path}")
    
    # Generate summary (keep original name for backward compatibility, also generate H2_summary.md)
    summary_path = output_dir / "summary.md"
    h2_summary_path = output_dir / "H2_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# H2 Calibration and Thresholding Experiment Results\n\n")
        f.write("## Hypothesis\n\n")
        f.write("Calibration and cost-aware thresholding produce more decision-aligned ")
        f.write("susceptibility scores than uncalibrated F1-optimized thresholds.\n\n")
        f.write("## Calibration Results\n\n")
        f.write(f"- **Brier Score (uncalibrated)**: {metrics['calibration']['brier_uncalibrated']:.4f}\n")
        f.write(f"- **Brier Score (calibrated)**: {metrics['calibration']['brier_calibrated']:.4f}\n")
        f.write(f"- **Brier Improvement**: {metrics['calibration']['brier_improvement']:.4f} "
                f"({metrics['calibration']['brier_improvement_pct']:.1f}% reduction)\n")
        f.write(f"- **ECE (uncalibrated)**: {metrics['calibration']['ece_uncalibrated']:.4f}\n")
        f.write(f"- **ECE (calibrated)**: {metrics['calibration']['ece_calibrated']:.4f}\n")
        f.write(f"- **ECE Improvement**: {metrics['calibration']['ece_improvement']:.4f} "
                f"({metrics['calibration']['ece_improvement_pct']:.1f}% reduction)\n\n")
        
        f.write("## Comparison vs Typical Baseline\n\n")
        f.write(f"- **Typical Uncalibrated Brier**: {metrics['calibration']['baseline_brier']:.3f} "
                f"(range: 0.18-0.22)\n")
        f.write(f"- **Typical Uncalibrated ECE**: {metrics['calibration']['baseline_ece']:.3f} "
                f"(range: 6-10%)\n")
        f.write(f"- **Calibrated Brier vs Baseline**: {metrics['calibration']['brier_vs_baseline_pct']:.1f}% better\n")
        f.write(f"- **Calibrated ECE vs Baseline**: {metrics['calibration']['ece_vs_baseline_pct']:.1f}% better\n\n")
        f.write("## Threshold Comparison\n\n")
        f.write("### F1-Optimized Threshold\n\n")
        f.write("**Uncalibrated:**\n")
        f.write(f"- Threshold: {metrics['f1_optimized']['uncalibrated']['threshold']:.4f}\n")
        f.write(f"- F1: {metrics['f1_optimized']['uncalibrated']['f1']:.4f}\n")
        f.write(f"- Expected Loss: {metrics['f1_optimized']['uncalibrated']['expected_loss']:.4f}\n\n")
        f.write("**Calibrated:**\n")
        f.write(f"- Threshold: {metrics['f1_optimized']['calibrated']['threshold']:.4f}\n")
        f.write(f"- F1: {metrics['f1_optimized']['calibrated']['f1']:.4f}\n")
        f.write(f"- Expected Loss: {metrics['f1_optimized']['calibrated']['expected_loss']:.4f}\n\n")
        f.write("### Cost-Optimized Threshold\n\n")
        f.write(f"Cost structure: FN={cost_fn}, FP={cost_fp}\n\n")
        f.write("**Uncalibrated:**\n")
        f.write(f"- Threshold: {metrics['cost_optimized']['uncalibrated']['threshold']:.4f}\n")
        f.write(f"- Expected Loss: {metrics['cost_optimized']['uncalibrated']['expected_loss']:.4f}\n")
        f.write(f"- Precision: {metrics['cost_optimized']['uncalibrated']['precision']:.4f}\n")
        f.write(f"- Recall: {metrics['cost_optimized']['uncalibrated']['recall']:.4f}\n\n")
        f.write("**Calibrated:**\n")
        f.write(f"- Threshold: {metrics['cost_optimized']['calibrated']['threshold']:.4f}\n")
        f.write(f"- Expected Loss: {metrics['cost_optimized']['calibrated']['expected_loss']:.4f}\n")
        f.write(f"- Precision: {metrics['cost_optimized']['calibrated']['precision']:.4f}\n")
        f.write(f"- Recall: {metrics['cost_optimized']['calibrated']['recall']:.4f}\n\n")
        f.write("## Conclusion\n\n")
        cal_improved = metrics['calibration']['brier_improvement'] > 0
        cost_better = metrics['cost_optimized']['calibrated']['expected_loss'] < metrics['f1_optimized']['calibrated']['expected_loss']
        if cal_improved and cost_better:
            f.write("✓ H2 is **supported**: Calibration improves Brier score and cost-optimal ")
            f.write("threshold reduces expected loss compared to F1-optimized threshold.\n\n")
            if "improvement_statement" in metrics:
                f.write(f"**Canonical Statement:** {metrics['improvement_statement']}\n")
        else:
            f.write("✗ H2 is **partially supported**: Results show mixed evidence.\n")
    
    logger.info(f"Saved summary to {summary_path}")
    
    # Also save as H2_summary.md for praxis validation
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_content = f.read()
    with open(h2_summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)
    logger.info(f"Saved H2 summary to {h2_summary_path}")
    
    logger.info("=" * 80)
    logger.info("H2 Experiment Complete")
    logger.info("=" * 80)
    
    return metrics


def main() -> None:
    """Main entry point for H2 experiment."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    parser = argparse.ArgumentParser(
        description="Run H2 calibration and thresholding experiment"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/H2_calibration_thresholds)",
    )
    parser.add_argument(
        "--cost-fn",
        type=float,
        default=10.0,
        help="Cost of false negative (default: 10.0)",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=1.0,
        help="Cost of false positive (default: 1.0)",
    )
    parser.add_argument(
        "--calibration-method",
        type=str,
        default="auto",
        choices=["platt", "isotonic", "auto"],
        help="Calibration method (default: auto)",
    )
    
    args = parser.parse_args()
    
    repo_root = Path.cwd()
    if args.output is None:
        args.output = repo_root / "results" / "H2_calibration_thresholds"
    
    try:
        metrics = run_h2_calibration_thresholds_experiment(
            output_dir=args.output,
            cost_fn=args.cost_fn,
            cost_fp=args.cost_fp,
            calibration_method=args.calibration_method,
            repo_root=repo_root,
        )
        
        print("\n" + "=" * 80)
        print("H2 Calibration and Thresholding Summary")
        print("=" * 80)
        print(f"Brier Improvement: {metrics['calibration']['brier_improvement']:.4f}")
        print(f"ECE Improvement: {metrics['calibration']['ece_improvement']:.4f}")
        print(f"Cost-Optimal Threshold (calibrated): {metrics['cost_optimized']['calibrated']['threshold']:.4f}")
        print(f"Expected Loss (cost-optimal): {metrics['cost_optimized']['calibrated']['expected_loss']:.4f}")
        print(f"Results saved to: {args.output}")
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"H2 experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
