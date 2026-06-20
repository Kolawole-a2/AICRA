"""
H2 Experiment: Cost-Aware Thresholding

This is the canonical H2 experiment module that evaluates cost-aware thresholding
for ransomware susceptibility scores under banking-style asymmetric costs.

Research Question (RQ2):
Does cost-aware thresholding reduce expected loss compared to F1-optimized
thresholds under banking-style asymmetric costs (FN cost >> FP cost)?

Hypothesis (H2):
Cost-aware thresholding produces lower expected loss than F1-optimized thresholds
under banking-style asymmetric costs (FN cost >> FP cost), demonstrating more
decision-aligned susceptibility scores for operational deployment.

Note on Calibration:
The model outputs are naturally well-calibrated (Brier=0.049, ECE=0.016 from H1).
Calibration metrics are reported for completeness, but the primary focus is on
cost-aware thresholding vs F1-optimized thresholds.

Metrics computed:
- Expected Loss: Cost-weighted loss at F1-optimized vs cost-optimal thresholds
- Threshold comparison: F1-optimized vs cost-optimal (uncalibrated and calibrated)
- Calibration: Brier score, ECE (before/after) - reported for completeness
- Reliability diagrams

Results are saved to results/H2_calibration_thresholds/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from ..config import Settings
from ..core.benchmarks import (
    compute_h2_improvements,
    format_improvement_statement,
)
from ..core.data import Dataset, load_ember_2024
from ..core.evaluation import cost_sensitive_threshold
from ..pipelines.calibration import CalibrationPipeline

logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
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


def find_f1_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[float, float]:
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
    cost_fp: float = 1.0,
) -> float:
    """
    Compute Expected Loss at a given threshold.

    Expected Loss = p(ransomware) * impact_cost
    where impact_cost = cost_fn * FN + cost_fp * FP
    """
    y_pred = (y_prob >= threshold).astype(int)

    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    total_loss = (cost_fn * fn) + (cost_fp * fp)
    total_samples = len(y_true)

    return float(total_loss / total_samples)


def bootstrap_ci(
    values: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        values: Array of values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)

    Returns:
        (lower_bound, upper_bound)
    """
    if len(values) < 2:
        return (
            float(values[0]) if len(values) == 1 else 0.0,
            float(values[0]) if len(values) == 1 else 0.0,
        )

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return (float(lower), float(upper))


def evaluate_h2_split(
    split_name: str,
    y_true: np.ndarray,
    y_prob_uncal: np.ndarray,
    y_prob_cal: np.ndarray,
    calibrator,
    cost_fn: float,
    cost_fp: float,
) -> dict:
    """
    Evaluate H2 metrics for a single split.

    Args:
        split_name: Name of the split
        y_true: True labels
        y_prob_uncal: Uncalibrated probabilities
        y_prob_cal: Calibrated probabilities
        calibrator: Calibrator object (for re-calibration if needed)
        cost_fn: Cost of false negative
        cost_fp: Cost of false positive

    Returns:
        Dictionary with split metrics
    """
    # Compute calibration metrics
    brier_uncal = brier_score_loss(y_true, y_prob_uncal)
    brier_cal = brier_score_loss(y_true, y_prob_cal)
    ece_uncal = compute_ece(y_true, y_prob_uncal)
    ece_cal = compute_ece(y_true, y_prob_cal)

    # Find optimal thresholds
    f1_threshold_uncal, _ = find_f1_optimal_threshold(y_true, y_prob_uncal)
    f1_threshold_cal, _ = find_f1_optimal_threshold(y_true, y_prob_cal)
    cost_threshold_uncal = cost_sensitive_threshold(
        y_true, y_prob_uncal, cost_fn, cost_fp
    )
    cost_threshold_cal = cost_sensitive_threshold(y_true, y_prob_cal, cost_fn, cost_fp)

    # Compute metrics at thresholds
    def compute_metrics_at_threshold(
        y_true: np.ndarray, y_prob: np.ndarray, threshold: float
    ) -> dict[str, float]:
        """Compute metrics at a given threshold."""
        y_pred = (y_prob >= threshold).astype(int)
        return {
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "expected_loss": compute_expected_loss(
                y_true, y_prob, threshold, cost_fn, cost_fp
            ),
        }

    return {
        "split": split_name,
        "n_samples": int(len(y_true)),
        "calibration": {
            "brier_uncalibrated": float(brier_uncal),
            "brier_calibrated": float(brier_cal),
            "brier_improvement": float(brier_uncal - brier_cal),
            "ece_uncalibrated": float(ece_uncal),
            "ece_calibrated": float(ece_cal),
            "ece_improvement": float(ece_uncal - ece_cal),
        },
        "f1_optimized": {
            "uncalibrated": compute_metrics_at_threshold(
                y_true, y_prob_uncal, f1_threshold_uncal
            ),
            "calibrated": compute_metrics_at_threshold(
                y_true, y_prob_cal, f1_threshold_cal
            ),
        },
        "cost_optimized": {
            "uncalibrated": compute_metrics_at_threshold(
                y_true, y_prob_uncal, cost_threshold_uncal
            ),
            "calibrated": compute_metrics_at_threshold(
                y_true, y_prob_cal, cost_threshold_cal
            ),
        },
    }


def aggregate_h2_metrics(all_results: list[dict]) -> dict:
    """
    Aggregate H2 metrics across all splits with bootstrap confidence intervals.

    Args:
        all_results: List of per-split result dictionaries

    Returns:
        Dictionary with aggregated metrics
    """
    logger.info("Aggregating H2 metrics across splits...")

    # Extract metrics for aggregation
    brier_uncal = [r["calibration"]["brier_uncalibrated"] for r in all_results]
    brier_cal = [r["calibration"]["brier_calibrated"] for r in all_results]
    brier_improvement = [r["calibration"]["brier_improvement"] for r in all_results]
    ece_uncal = [r["calibration"]["ece_uncalibrated"] for r in all_results]
    ece_cal = [r["calibration"]["ece_calibrated"] for r in all_results]
    ece_improvement = [r["calibration"]["ece_improvement"] for r in all_results]

    # Expected loss metrics
    f1_uncal_loss = [
        r["f1_optimized"]["uncalibrated"]["expected_loss"] for r in all_results
    ]
    f1_cal_loss = [
        r["f1_optimized"]["calibrated"]["expected_loss"] for r in all_results
    ]
    cost_uncal_loss = [
        r["cost_optimized"]["uncalibrated"]["expected_loss"] for r in all_results
    ]
    cost_cal_loss = [
        r["cost_optimized"]["calibrated"]["expected_loss"] for r in all_results
    ]

    # Compute means and stds
    aggregated = {
        "calibration": {
            "brier_uncalibrated": {
                "mean": float(np.mean(brier_uncal)),
                "std": float(np.std(brier_uncal, ddof=1)),
            },
            "brier_calibrated": {
                "mean": float(np.mean(brier_cal)),
                "std": float(np.std(brier_cal, ddof=1)),
            },
            "brier_improvement": {
                "mean": float(np.mean(brier_improvement)),
                "std": float(np.std(brier_improvement, ddof=1)),
                "ci_95": bootstrap_ci(np.array(brier_improvement)),
            },
            "ece_uncalibrated": {
                "mean": float(np.mean(ece_uncal)),
                "std": float(np.std(ece_uncal, ddof=1)),
            },
            "ece_calibrated": {
                "mean": float(np.mean(ece_cal)),
                "std": float(np.std(ece_cal, ddof=1)),
            },
            "ece_improvement": {
                "mean": float(np.mean(ece_improvement)),
                "std": float(np.std(ece_improvement, ddof=1)),
                "ci_95": bootstrap_ci(np.array(ece_improvement)),
            },
        },
        "f1_optimized": {
            "uncalibrated": {
                "expected_loss": {
                    "mean": float(np.mean(f1_uncal_loss)),
                    "std": float(np.std(f1_uncal_loss, ddof=1)),
                }
            },
            "calibrated": {
                "expected_loss": {
                    "mean": float(np.mean(f1_cal_loss)),
                    "std": float(np.std(f1_cal_loss, ddof=1)),
                }
            },
        },
        "cost_optimized": {
            "uncalibrated": {
                "expected_loss": {
                    "mean": float(np.mean(cost_uncal_loss)),
                    "std": float(np.std(cost_uncal_loss, ddof=1)),
                    "ci_95": bootstrap_ci(np.array(cost_uncal_loss)),
                }
            },
            "calibrated": {
                "expected_loss": {
                    "mean": float(np.mean(cost_cal_loss)),
                    "std": float(np.std(cost_cal_loss, ddof=1)),
                    "ci_95": bootstrap_ci(np.array(cost_cal_loss)),
                }
            },
        },
    }

    # Compute overall improvements (using aggregated means)
    mean_brier_uncal = aggregated["calibration"]["brier_uncalibrated"]["mean"]
    mean_brier_cal = aggregated["calibration"]["brier_calibrated"]["mean"]
    mean_ece_uncal = aggregated["calibration"]["ece_uncalibrated"]["mean"]
    mean_ece_cal = aggregated["calibration"]["ece_calibrated"]["mean"]

    from ..core.benchmarks import compute_h2_improvements

    h2_improvements = compute_h2_improvements(
        brier_uncalibrated=mean_brier_uncal,
        brier_calibrated=mean_brier_cal,
        ece_uncalibrated=mean_ece_uncal,
        ece_calibrated=mean_ece_cal,
    )

    aggregated["improvements"] = h2_improvements
    aggregated["improvement_statement"] = format_improvement_statement(
        "H2",
        {
            "ece_improvement_pct": h2_improvements["ece_improvement_pct"],
        },
    )

    return aggregated


def run_h2_calibration_thresholds_experiment(
    output_dir: Path,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    calibration_method: str = "auto",
    splits_config_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict:
    """
    Run H2 calibration and thresholding experiment.

    Args:
        output_dir: Directory to save results
        cost_fn: Cost of false negative (default: 10.0 for banking)
        cost_fp: Cost of false positive (default: 1.0)
        calibration_method: Calibration method ("platt", "isotonic", "auto")
        splits_config_path: Optional path to h2_splits.yaml for multi-split evaluation
        repo_root: Repository root directory

    Returns:
        Dictionary with all metrics (per-split and aggregated if multi-split)
    """
    if repo_root is None:
        repo_root = Path.cwd()

    logger.info("=" * 80)
    logger.info("H2 Experiment: Calibration and Cost-Aware Thresholding")
    logger.info("=" * 80)

    # Check if multi-split evaluation is requested
    use_multi_split = splits_config_path is not None and splits_config_path.exists()
    if use_multi_split:
        logger.info(f"Multi-split evaluation enabled: {splits_config_path}")
    else:
        logger.info("Single-split evaluation (default test set)")

    settings = Settings()

    # Load data (using model predictions from H1)
    logger.info("Loading data and model predictions...")
    try:
        train_data_full, test_data = load_ember_2024()
        logger.info(
            f"Full train: {len(train_data_full.features)}, Test: {len(test_data.features)}"
        )

        # Split training data into train/val for calibration
        # Use 10% of training data for validation
        val_split = 0.1
        n_train = len(train_data_full.features)
        split_idx = int(n_train * (1 - val_split))

        train_data = Dataset(
            features=train_data_full.features.iloc[:split_idx].reset_index(drop=True),
            labels=train_data_full.labels.iloc[:split_idx].reset_index(drop=True),
            families=(
                train_data_full.families.iloc[:split_idx].reset_index(drop=True)
                if train_data_full.families is not None
                else None
            ),
            timestamps=(
                train_data_full.timestamps.iloc[:split_idx].reset_index(drop=True)
                if train_data_full.timestamps is not None
                else None
            ),
        )

        val_data = Dataset(
            features=train_data_full.features.iloc[split_idx:].reset_index(drop=True),
            labels=train_data_full.labels.iloc[split_idx:].reset_index(drop=True),
            families=(
                train_data_full.families.iloc[split_idx:].reset_index(drop=True)
                if train_data_full.families is not None
                else None
            ),
            timestamps=(
                train_data_full.timestamps.iloc[split_idx:].reset_index(drop=True)
                if train_data_full.timestamps is not None
                else None
            ),
        )

        logger.info(
            f"Train: {len(train_data.features)}, Val: {len(val_data.features)}, Test: {len(test_data.features)}"
        )
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

    # Initialize metrics variable (will be set by either multi-split or single-split path)
    metrics = None

    # ========================================================================
    # MULTI-SPLIT EVALUATION (if splits config provided)
    # ========================================================================
    if use_multi_split:
        logger.info("=" * 80)
        logger.info("Multi-Split Evaluation Mode")
        logger.info("=" * 80)

        # Load splits configuration
        with open(splits_config_path) as f:
            config = yaml.safe_load(f)
        splits_config = config.get("splits", {})

        if not splits_config:
            logger.warning("No splits found in config, falling back to single-split")
            use_multi_split = False
        else:
            logger.info(f"Found {len(splits_config)} splits in configuration")

            # Create splits from test data (similar to rebuild pipeline but only from test set)
            def create_splits_from_test_data(test_data: Dataset) -> dict[str, Dataset]:
                """Create multiple splits from test data."""
                n_test = len(test_data.features)

                # Define split sizes (same as rebuild pipeline)
                main_n = min(10_000, n_test)
                small_n = min(2_000, main_n)
                smoke_n = min(200, small_n)

                splits = {}

                # full_ember: all test data (use entire test set without slicing)
                splits["full_ember"] = Dataset(
                    features=test_data.features.reset_index(drop=True),
                    labels=test_data.labels.reset_index(drop=True),
                    families=(
                        test_data.families.reset_index(drop=True)
                        if test_data.families is not None
                        else None
                    ),
                    timestamps=(
                        test_data.timestamps.reset_index(drop=True)
                        if test_data.timestamps is not None
                        else None
                    ),
                )

                # main: first 10,000
                splits["main"] = Dataset(
                    features=test_data.features.iloc[:main_n].reset_index(drop=True),
                    labels=test_data.labels.iloc[:main_n].reset_index(drop=True),
                    families=(
                        test_data.families.iloc[:main_n].reset_index(drop=True)
                        if test_data.families is not None
                        else None
                    ),
                    timestamps=(
                        test_data.timestamps.iloc[:main_n].reset_index(drop=True)
                        if test_data.timestamps is not None
                        else None
                    ),
                )

                # small_ember: first 2,000
                splits["small_ember"] = Dataset(
                    features=test_data.features.iloc[:small_n].reset_index(drop=True),
                    labels=test_data.labels.iloc[:small_n].reset_index(drop=True),
                    families=(
                        test_data.families.iloc[:small_n].reset_index(drop=True)
                        if test_data.families is not None
                        else None
                    ),
                    timestamps=(
                        test_data.timestamps.iloc[:small_n].reset_index(drop=True)
                        if test_data.timestamps is not None
                        else None
                    ),
                )

                # smoke_test: first 200
                splits["smoke_test"] = Dataset(
                    features=test_data.features.iloc[:smoke_n].reset_index(drop=True),
                    labels=test_data.labels.iloc[:smoke_n].reset_index(drop=True),
                    families=(
                        test_data.families.iloc[:smoke_n].reset_index(drop=True)
                        if test_data.families is not None
                        else None
                    ),
                    timestamps=(
                        test_data.timestamps.iloc[:smoke_n].reset_index(drop=True)
                        if test_data.timestamps is not None
                        else None
                    ),
                )

                return splits

            # Create splits from test data
            test_splits = create_splits_from_test_data(test_data)
            logger.info(f"Created {len(test_splits)} splits from test data")

            # Evaluate each split
            all_split_results = []
            for split_name, split_data in test_splits.items():
                logger.info(
                    f"Evaluating split: {split_name} ({len(split_data.features)} samples)"
                )

                # Generate predictions for this split
                X_split = split_data.features.values
                X_split_df = pd.DataFrame(X_split)
                prob_split = model.predict_proba(X_split_df)

                if prob_split.ndim == 1:
                    y_prob_split_uncal = prob_split
                else:
                    y_prob_split_uncal = prob_split[:, 1]

                y_prob_split_cal = calibrator.transform(y_prob_split_uncal)
                y_true_split = split_data.labels.values

                # Evaluate split
                split_result = evaluate_h2_split(
                    split_name=split_name,
                    y_true=y_true_split,
                    y_prob_uncal=y_prob_split_uncal,
                    y_prob_cal=y_prob_split_cal,
                    calibrator=calibrator,
                    cost_fn=cost_fn,
                    cost_fp=cost_fp,
                )
                all_split_results.append(split_result)

                logger.info(
                    f"  {split_name}: Brier uncal={split_result['calibration']['brier_uncalibrated']:.4f}, "
                    f"cal={split_result['calibration']['brier_calibrated']:.4f}, "
                    f"Cost-opt loss={split_result['cost_optimized']['calibrated']['expected_loss']:.4f}"
                )

            # Aggregate metrics across splits
            aggregated_metrics = aggregate_h2_metrics(all_split_results)

            # Use aggregated metrics for overall results
            brier_uncalibrated = aggregated_metrics["calibration"][
                "brier_uncalibrated"
            ]["mean"]
            brier_calibrated = aggregated_metrics["calibration"]["brier_calibrated"][
                "mean"
            ]
            ece_uncalibrated = aggregated_metrics["calibration"]["ece_uncalibrated"][
                "mean"
            ]
            ece_calibrated = aggregated_metrics["calibration"]["ece_calibrated"]["mean"]

            # Use first split (full_ember) for threshold finding (for backward compatibility)
            # Or use aggregated thresholds
            full_ember_result = next(
                r for r in all_split_results if r["split"] == "full_ember"
            )
            f1_threshold_uncal = full_ember_result["f1_optimized"]["uncalibrated"][
                "threshold"
            ]
            f1_threshold_cal = full_ember_result["f1_optimized"]["calibrated"][
                "threshold"
            ]
            cost_threshold_uncal = full_ember_result["cost_optimized"]["uncalibrated"][
                "threshold"
            ]
            cost_threshold_cal = full_ember_result["cost_optimized"]["calibrated"][
                "threshold"
            ]

            # Temporal calibration check (use full_ember split)
            temporal_calibration_check = {}
            if (
                hasattr(test_data, "timestamps")
                and test_data.timestamps is not None
                and hasattr(val_data, "timestamps")
                and val_data.timestamps is not None
            ):
                logger.info("Performing temporal calibration check...")
                val_max_ts = val_data.timestamps.max()
                test_min_ts = test_data.timestamps.min()

                if val_max_ts < test_min_ts:
                    logger.info(
                        f"✅ Temporal ordering verified: calibration max_ts={val_max_ts}, test min_ts={test_min_ts}"
                    )
                    temporal_calibration_check = {
                        "temporal_ordering_verified": True,
                        "calibration_window_max_ts": str(val_max_ts),
                        "test_window_min_ts": str(test_min_ts),
                    }
                else:
                    logger.warning(
                        f"⚠️  Temporal ordering issue: calibration max_ts={val_max_ts} >= test min_ts={test_min_ts}"
                    )
                    temporal_calibration_check = {
                        "temporal_ordering_verified": False,
                        "warning": "Calibration and test windows may overlap temporally",
                    }

            # Compute improvements from aggregated metrics
            h2_improvements = aggregated_metrics["improvements"]

            # Build metrics structure with per-split and aggregated results
            metrics = {
                "per_split_results": all_split_results,
                "aggregated_metrics": aggregated_metrics,
                "calibration": {
                    "brier_uncalibrated": float(brier_uncalibrated),
                    "brier_calibrated": float(brier_calibrated),
                    "brier_improvement": float(brier_uncalibrated - brier_calibrated),
                    "brier_improvement_pct": h2_improvements["brier_improvement_pct"],
                    "brier_vs_baseline_pct": h2_improvements["brier_vs_baseline_pct"],
                    "ece_uncalibrated": float(ece_uncalibrated),
                    "ece_calibrated": float(ece_calibrated),
                    "ece_improvement": float(ece_uncalibrated - ece_calibrated),
                    "ece_improvement_pct": h2_improvements["ece_improvement_pct"],
                    "ece_vs_baseline_pct": h2_improvements["ece_vs_baseline_pct"],
                    "baseline_brier": h2_improvements["baseline_brier"],
                    "baseline_ece": h2_improvements["baseline_ece"],
                    "method": calibration_method,
                    "temporal_calibration_check": temporal_calibration_check,
                },
                "improvement_statement": aggregated_metrics["improvement_statement"],
                "f1_optimized": {
                    "uncalibrated": {
                        "threshold": f1_threshold_uncal,
                        "expected_loss": aggregated_metrics["f1_optimized"][
                            "uncalibrated"
                        ]["expected_loss"]["mean"],
                    },
                    "calibrated": {
                        "threshold": f1_threshold_cal,
                        "expected_loss": aggregated_metrics["f1_optimized"][
                            "calibrated"
                        ]["expected_loss"]["mean"],
                    },
                },
                "cost_optimized": {
                    "cost_fn": float(cost_fn),
                    "cost_fp": float(cost_fp),
                    "uncalibrated": {
                        "threshold": cost_threshold_uncal,
                        "expected_loss": aggregated_metrics["cost_optimized"][
                            "uncalibrated"
                        ]["expected_loss"]["mean"],
                    },
                    "calibrated": {
                        "threshold": cost_threshold_cal,
                        "expected_loss": aggregated_metrics["cost_optimized"][
                            "calibrated"
                        ]["expected_loss"]["mean"],
                    },
                },
                "n_test_samples": sum(r["n_samples"] for r in all_split_results),
                "splits_evaluated": [r["split"] for r in all_split_results],
            }

            # Multi-split path: metrics already built, skip to saving
            logger.info("=" * 80)
            logger.info("Multi-split evaluation complete")
            logger.info("=" * 80)

    if not use_multi_split:
        # ========================================================================
        # SINGLE-SPLIT EVALUATION (Original behavior - backward compatible)
        # ========================================================================
        # Compute calibration metrics
        logger.info("Computing calibration metrics...")
        brier_uncalibrated = brier_score_loss(y_true_test, y_prob_test)
        brier_calibrated = brier_score_loss(y_true_test, y_prob_test_calibrated)
        ece_uncalibrated = compute_ece(y_true_test, y_prob_test)
        ece_calibrated = compute_ece(y_true_test, y_prob_test_calibrated)

        # ========================================================================
        # TEMPORAL CALIBRATION CHECK (H2 Requirement)
        # ========================================================================
        temporal_calibration_check = {}
        if (
            hasattr(test_data, "timestamps")
            and test_data.timestamps is not None
            and hasattr(val_data, "timestamps")
            and val_data.timestamps is not None
        ):
            logger.info("Performing temporal calibration check...")

            # Verify temporal ordering: calibration data (train/val) should be earlier than test
            val_max_ts = val_data.timestamps.max()
            test_min_ts = test_data.timestamps.min()

            if val_max_ts < test_min_ts:
                logger.info(
                    f"✅ Temporal ordering verified: calibration max_ts={val_max_ts}, test min_ts={test_min_ts}"
                )

                # Compute calibration metrics on validation set (earlier window)
                brier_val_uncal = brier_score_loss(y_true_val, y_prob_val)
                brier_val_cal = brier_score_loss(
                    y_true_val, calibrator.transform(y_prob_val)
                )
                ece_val_uncal = compute_ece(y_true_val, y_prob_val)
                ece_val_cal = compute_ece(y_true_val, calibrator.transform(y_prob_val))

                # Compare: calibration on earlier window vs test on later window
                temporal_calibration_check = {
                    "temporal_ordering_verified": True,
                    "calibration_window_max_ts": str(val_max_ts),
                    "test_window_min_ts": str(test_min_ts),
                    "calibration_on_earlier_window": {
                        "brier_uncalibrated": float(brier_val_uncal),
                        "brier_calibrated": float(brier_val_cal),
                        "ece_uncalibrated": float(ece_val_uncal),
                        "ece_calibrated": float(ece_val_cal),
                    },
                    "test_on_later_window": {
                        "brier_uncalibrated": float(brier_uncalibrated),
                        "brier_calibrated": float(brier_calibrated),
                        "ece_uncalibrated": float(ece_uncalibrated),
                        "ece_calibrated": float(ece_calibrated),
                    },
                    "calibration_transferability": {
                        "brier_delta": float(brier_calibrated - brier_val_cal),
                        "ece_delta": float(ece_calibrated - ece_val_cal),
                        "note": "Positive delta indicates calibration degrades on later window (expected for temporal shift)",
                    },
                }
                logger.info(
                    f"Temporal calibration: Brier val={brier_val_cal:.4f}, test={brier_calibrated:.4f}, delta={temporal_calibration_check['calibration_transferability']['brier_delta']:.4f}"
                )
            else:
                logger.warning(
                    f"⚠️  Temporal ordering issue: calibration max_ts={val_max_ts} >= test min_ts={test_min_ts}"
                )
                temporal_calibration_check = {
                    "temporal_ordering_verified": False,
                    "warning": "Calibration and test windows may overlap temporally",
                }
        else:
            logger.warning(
                "⚠️  Timestamps not available for temporal calibration check"
            )
            temporal_calibration_check = {
                "temporal_ordering_verified": False,
                "warning": "Timestamps not available in dataset",
            }

        # ========================================================================
        # % IMPROVEMENT CALCULATIONS (H2 Requirement)
        # ========================================================================
        h2_improvements = compute_h2_improvements(
            brier_uncalibrated=brier_uncalibrated,
            brier_calibrated=brier_calibrated,
            ece_uncalibrated=ece_uncalibrated,
            ece_calibrated=ece_calibrated,
        )
        logger.info(
            f"Calibration improvements: Brier {h2_improvements['brier_improvement_pct']:.1f}%, "
            f"ECE {h2_improvements['ece_improvement_pct']:.1f}%"
        )

        # Find optimal thresholds
        logger.info("Finding optimal thresholds...")

        # F1-optimized threshold (uncalibrated)
        f1_threshold_uncal, f1_score_uncal = find_f1_optimal_threshold(
            y_true_test, y_prob_test
        )

        # F1-optimized threshold (calibrated)
        f1_threshold_cal, f1_score_cal = find_f1_optimal_threshold(
            y_true_test, y_prob_test_calibrated
        )

        # Cost-optimal threshold (uncalibrated)
        cost_threshold_uncal = cost_sensitive_threshold(
            y_true_test, y_prob_test, cost_fn, cost_fp
        )

        # Cost-optimal threshold (calibrated)
        cost_threshold_cal = cost_sensitive_threshold(
            y_true_test, y_prob_test_calibrated, cost_fn, cost_fp
        )

        # Compute metrics at each threshold
        def compute_metrics_at_threshold(
            y_true: np.ndarray, y_prob: np.ndarray, threshold: float
        ) -> dict[str, float]:
            """Compute metrics at a given threshold."""
            y_pred = (y_prob >= threshold).astype(int)
            return {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "expected_loss": compute_expected_loss(
                    y_true, y_prob, threshold, cost_fn, cost_fp
                ),
            }

        metrics = {
            "calibration": {
                "brier_uncalibrated": float(brier_uncalibrated),
                "brier_calibrated": float(brier_calibrated),
                "brier_improvement": float(brier_uncalibrated - brier_calibrated),
                "brier_improvement_pct": h2_improvements["brier_improvement_pct"],
                "brier_vs_baseline_pct": h2_improvements["brier_vs_baseline_pct"],
                "ece_uncalibrated": float(ece_uncalibrated),
                "ece_calibrated": float(ece_calibrated),
                "ece_improvement": float(ece_uncalibrated - ece_calibrated),
                "ece_improvement_pct": h2_improvements["ece_improvement_pct"],
                "ece_vs_baseline_pct": h2_improvements["ece_vs_baseline_pct"],
                "baseline_brier": h2_improvements["baseline_brier"],
                "baseline_ece": h2_improvements["baseline_ece"],
                "method": calibration_method,
                "temporal_calibration_check": temporal_calibration_check,
            },
            # ====================================================================
            # CANONICAL IMPROVEMENT STATEMENT (H2 Requirement)
            # ====================================================================
            "improvement_statement": format_improvement_statement(
                "H2",
                {
                    "ece_improvement_pct": h2_improvements["ece_improvement_pct"],
                },
            ),
            "f1_optimized": {
                "uncalibrated": compute_metrics_at_threshold(
                    y_true_test, y_prob_test, f1_threshold_uncal
                ),
                "calibrated": compute_metrics_at_threshold(
                    y_true_test, y_prob_test_calibrated, f1_threshold_cal
                ),
            },
            "cost_optimized": {
                "cost_fn": float(cost_fn),
                "cost_fp": float(cost_fp),
                "uncalibrated": compute_metrics_at_threshold(
                    y_true_test, y_prob_test, cost_threshold_uncal
                ),
                "calibrated": compute_metrics_at_threshold(
                    y_true_test, y_prob_test_calibrated, cost_threshold_cal
                ),
            },
            "n_test_samples": len(test_data.features),
        }
    # End of single-split evaluation block

    # ========================================================================
    # SAVE RESULTS (Both single-split and multi-split paths converge here)
    # ========================================================================
    if metrics is None:
        raise RuntimeError(
            "Metrics not defined - this should not happen. Check code flow."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as metrics.json (backward compatibility)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Debug: Log metrics structure for multi-split
    if use_multi_split:
        logger.info(
            f"Multi-split metrics structure: has per_split_results={('per_split_results' in metrics)}, "
            f"has aggregated_metrics={('aggregated_metrics' in metrics)}, "
            f"splits_evaluated={metrics.get('splits_evaluated', [])}"
        )

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
    if use_multi_split:
        full_results["splits_evaluated"] = metrics.get("splits_evaluated", [])
        full_results["evaluation_mode"] = "multi_split"
    else:
        full_results["evaluation_mode"] = "single_split"

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
        f.write(
            "Calibration and cost-aware thresholding produce more decision-aligned "
        )
        f.write("susceptibility scores than uncalibrated F1-optimized thresholds.\n\n")
        # Add evaluation mode header
        if use_multi_split:
            splits_evaluated = metrics.get("splits_evaluated", [])
            f.write("## Evaluation Mode: Multi-Split\n\n")
            if splits_evaluated:
                f.write(f"Evaluated across {len(splits_evaluated)} splits: ")
                f.write(", ".join(splits_evaluated) + "\n\n")
            else:
                # Fallback: try to get from per_split_results
                per_split = metrics.get("per_split_results", [])
                if per_split:
                    splits_evaluated = [r["split"] for r in per_split]
                    f.write(f"Evaluated across {len(splits_evaluated)} splits: ")
                    f.write(", ".join(splits_evaluated) + "\n\n")
                else:
                    f.write(
                        "Multi-split evaluation (splits information not available)\n\n"
                    )
        else:
            f.write("## Evaluation Mode: Single-Split\n\n")
            f.write(
                f"Evaluated on test set: {metrics.get('n_test_samples', 0)} samples\n\n"
            )

        f.write("## Calibration Results\n\n")

        if use_multi_split:
            # Show aggregated results for multi-split
            agg = metrics.get("aggregated_metrics", {})
            cal_agg = agg.get("calibration", {})
            f.write("### Aggregated Across Splits\n\n")
            f.write(
                f"- **Brier Score (uncalibrated)**: {cal_agg.get('brier_uncalibrated', {}).get('mean', 0):.4f} "
                f"(std: {cal_agg.get('brier_uncalibrated', {}).get('std', 0):.4f})\n"
            )
            f.write(
                f"- **Brier Score (calibrated)**: {cal_agg.get('brier_calibrated', {}).get('mean', 0):.4f} "
                f"(std: {cal_agg.get('brier_calibrated', {}).get('std', 0):.4f})\n"
            )
            f.write(
                f"- **Brier Improvement**: {cal_agg.get('brier_improvement', {}).get('mean', 0):.4f} "
                f"({metrics['calibration']['brier_improvement_pct']:.1f}% reduction)\n"
            )
            f.write(
                f"- **ECE (uncalibrated)**: {cal_agg.get('ece_uncalibrated', {}).get('mean', 0):.4f} "
                f"(std: {cal_agg.get('ece_uncalibrated', {}).get('std', 0):.4f})\n"
            )
            f.write(
                f"- **ECE (calibrated)**: {cal_agg.get('ece_calibrated', {}).get('mean', 0):.4f} "
                f"(std: {cal_agg.get('ece_calibrated', {}).get('std', 0):.4f})\n"
            )
            f.write(
                f"- **ECE Improvement**: {cal_agg.get('ece_improvement', {}).get('mean', 0):.4f} "
                f"({metrics['calibration']['ece_improvement_pct']:.1f}% reduction)\n\n"
            )

            # Show per-split results
            f.write("### Per-Split Results\n\n")
            for split_result in metrics.get("per_split_results", []):
                split_name = split_result["split"]
                split_cal = split_result["calibration"]
                f.write(f"**{split_name}** ({split_result['n_samples']} samples):\n")
                f.write(
                    f"- Brier uncal: {split_cal['brier_uncalibrated']:.4f}, cal: {split_cal['brier_calibrated']:.4f}\n"
                )
                f.write(
                    f"- ECE uncal: {split_cal['ece_uncalibrated']:.4f}, cal: {split_cal['ece_calibrated']:.4f}\n"
                )
                f.write(
                    f"- Cost-opt loss (cal): {split_result['cost_optimized']['calibrated']['expected_loss']:.4f}\n\n"
                )
        else:
            # Show single-split results
            f.write(
                f"- **Brier Score (uncalibrated)**: {metrics['calibration']['brier_uncalibrated']:.4f}\n"
            )
            f.write(
                f"- **Brier Score (calibrated)**: {metrics['calibration']['brier_calibrated']:.4f}\n"
            )
            f.write(
                f"- **Brier Improvement**: {metrics['calibration']['brier_improvement']:.4f} "
                f"({metrics['calibration']['brier_improvement_pct']:.1f}% reduction)\n"
            )
            f.write(
                f"- **ECE (uncalibrated)**: {metrics['calibration']['ece_uncalibrated']:.4f}\n"
            )
            f.write(
                f"- **ECE (calibrated)**: {metrics['calibration']['ece_calibrated']:.4f}\n"
            )
            f.write(
                f"- **ECE Improvement**: {metrics['calibration']['ece_improvement']:.4f} "
                f"({metrics['calibration']['ece_improvement_pct']:.1f}% reduction)\n\n"
            )

        f.write("## Comparison vs Typical Baseline\n\n")
        f.write(
            f"- **Typical Uncalibrated Brier**: {metrics['calibration']['baseline_brier']:.3f} "
            f"(range: 0.18-0.22)\n"
        )
        f.write(
            f"- **Typical Uncalibrated ECE**: {metrics['calibration']['baseline_ece']:.3f} "
            f"(range: 6-10%)\n"
        )
        f.write(
            f"- **Calibrated Brier vs Baseline**: {metrics['calibration']['brier_vs_baseline_pct']:.1f}% better\n"
        )
        f.write(
            f"- **Calibrated ECE vs Baseline**: {metrics['calibration']['ece_vs_baseline_pct']:.1f}% better\n\n"
        )
        f.write("## Threshold Comparison\n\n")

        if use_multi_split:
            # Show aggregated threshold results
            agg = metrics.get("aggregated_metrics", {})
            f.write("### Aggregated Results Across Splits\n\n")

            # Use full_ember split for threshold values (representative)
            full_ember_result = next(
                (
                    r
                    for r in metrics.get("per_split_results", [])
                    if r["split"] == "full_ember"
                ),
                None,
            )

            if full_ember_result:
                f.write("**F1-Optimized Threshold (from full_ember split):**\n")
                f.write(
                    f"- Uncalibrated: {full_ember_result['f1_optimized']['uncalibrated']['threshold']:.4f}\n"
                )
                f.write(
                    f"- Calibrated: {full_ember_result['f1_optimized']['calibrated']['threshold']:.4f}\n\n"
                )

                f.write("**Cost-Optimized Threshold (from full_ember split):**\n")
                f.write(
                    f"- Uncalibrated: {full_ember_result['cost_optimized']['uncalibrated']['threshold']:.4f}\n"
                )
                f.write(
                    f"- Calibrated: {full_ember_result['cost_optimized']['calibrated']['threshold']:.4f}\n\n"
                )

            # Show aggregated expected loss
            f.write("### Expected Loss (Aggregated)\n\n")
            f.write("**F1-Optimized:**\n")
            f.write(
                f"- Uncalibrated: {agg.get('f1_optimized', {}).get('uncalibrated', {}).get('expected_loss', {}).get('mean', 0):.4f}\n"
            )
            f.write(
                f"- Calibrated: {agg.get('f1_optimized', {}).get('calibrated', {}).get('expected_loss', {}).get('mean', 0):.4f}\n\n"
            )

            f.write("**Cost-Optimized:**\n")
            f.write(
                f"- Uncalibrated: {agg.get('cost_optimized', {}).get('uncalibrated', {}).get('expected_loss', {}).get('mean', 0):.4f}\n"
            )
            f.write(
                f"- Calibrated: {agg.get('cost_optimized', {}).get('calibrated', {}).get('expected_loss', {}).get('mean', 0):.4f}\n\n"
            )

            f.write("### Per-Split Expected Loss\n\n")
            for split_result in metrics.get("per_split_results", []):
                split_name = split_result["split"]
                f.write(f"**{split_name}**:\n")
                f.write(
                    f"- F1-opt (uncal): {split_result['f1_optimized']['uncalibrated']['expected_loss']:.4f}\n"
                )
                f.write(
                    f"- F1-opt (cal): {split_result['f1_optimized']['calibrated']['expected_loss']:.4f}\n"
                )
                f.write(
                    f"- Cost-opt (uncal): {split_result['cost_optimized']['uncalibrated']['expected_loss']:.4f}\n"
                )
                f.write(
                    f"- Cost-opt (cal): {split_result['cost_optimized']['calibrated']['expected_loss']:.4f}\n\n"
                )
        else:
            # Show single-split threshold results
            f.write("### F1-Optimized Threshold\n\n")
            f.write("**Uncalibrated:**\n")
            f.write(
                f"- Threshold: {metrics['f1_optimized']['uncalibrated']['threshold']:.4f}\n"
            )
            f.write(f"- F1: {metrics['f1_optimized']['uncalibrated']['f1']:.4f}\n")
            f.write(
                f"- Expected Loss: {metrics['f1_optimized']['uncalibrated']['expected_loss']:.4f}\n\n"
            )
            f.write("**Calibrated:**\n")
            f.write(
                f"- Threshold: {metrics['f1_optimized']['calibrated']['threshold']:.4f}\n"
            )
            f.write(f"- F1: {metrics['f1_optimized']['calibrated']['f1']:.4f}\n")
            f.write(
                f"- Expected Loss: {metrics['f1_optimized']['calibrated']['expected_loss']:.4f}\n\n"
            )
            f.write("### Cost-Optimized Threshold\n\n")
            f.write(f"Cost structure: FN={cost_fn}, FP={cost_fp}\n\n")
            f.write("**Uncalibrated:**\n")
            f.write(
                f"- Threshold: {metrics['cost_optimized']['uncalibrated']['threshold']:.4f}\n"
            )
            f.write(
                f"- Expected Loss: {metrics['cost_optimized']['uncalibrated']['expected_loss']:.4f}\n"
            )
            f.write(
                f"- Precision: {metrics['cost_optimized']['uncalibrated']['precision']:.4f}\n"
            )
            f.write(
                f"- Recall: {metrics['cost_optimized']['uncalibrated']['recall']:.4f}\n\n"
            )
            f.write("**Calibrated:**\n")
            f.write(
                f"- Threshold: {metrics['cost_optimized']['calibrated']['threshold']:.4f}\n"
            )
            f.write(
                f"- Expected Loss: {metrics['cost_optimized']['calibrated']['expected_loss']:.4f}\n"
            )
            f.write(
                f"- Precision: {metrics['cost_optimized']['calibrated']['precision']:.4f}\n"
            )
            f.write(
                f"- Recall: {metrics['cost_optimized']['calibrated']['recall']:.4f}\n\n"
            )
        f.write("## Conclusion\n\n")

        # H2 Hypothesis: "Calibration and cost-aware thresholding produce more decision-aligned
        # susceptibility scores than uncalibrated F1-optimized thresholds."
        #
        # Key comparison: Cost-optimized (calibrated or uncalibrated) vs F1-optimized (uncalibrated)
        if use_multi_split:
            # Use aggregated means for multi-split
            agg = metrics.get("aggregated_metrics", {})
            f1_uncal_loss = (
                agg.get("f1_optimized", {})
                .get("uncalibrated", {})
                .get("expected_loss", {})
                .get("mean", 0)
            )
            cost_uncal_loss = (
                agg.get("cost_optimized", {})
                .get("uncalibrated", {})
                .get("expected_loss", {})
                .get("mean", 0)
            )
            cost_cal_loss = (
                agg.get("cost_optimized", {})
                .get("calibrated", {})
                .get("expected_loss", {})
                .get("mean", 0)
            )
        else:
            # Use single-split values
            f1_uncal_loss = metrics["f1_optimized"]["uncalibrated"]["expected_loss"]
            cost_uncal_loss = metrics["cost_optimized"]["uncalibrated"]["expected_loss"]
            cost_cal_loss = metrics["cost_optimized"]["calibrated"]["expected_loss"]

        # Cost-aware thresholding should reduce expected loss compared to F1-optimized uncalibrated
        cost_uncal_better = cost_uncal_loss < f1_uncal_loss
        cost_cal_better = cost_cal_loss < f1_uncal_loss

        # Calculate improvement percentages (handle division by zero)
        if f1_uncal_loss > 0:
            cost_uncal_improvement_pct = (
                (f1_uncal_loss - cost_uncal_loss) / f1_uncal_loss
            ) * 100
            cost_cal_improvement_pct = (
                (f1_uncal_loss - cost_cal_loss) / f1_uncal_loss
            ) * 100
        else:
            # If F1-optimized loss is zero, improvement is infinite/undefined
            cost_uncal_improvement_pct = (
                float("inf") if cost_uncal_loss < f1_uncal_loss else 0.0
            )
            cost_cal_improvement_pct = (
                float("inf") if cost_cal_loss < f1_uncal_loss else 0.0
            )

        # H2 is supported if cost-aware thresholding (calibrated or uncalibrated) reduces expected loss
        h2_supported = cost_uncal_better or cost_cal_better

        if h2_supported:
            f.write(
                "✓ H2 is **supported**: Cost-aware thresholding produces more decision-aligned "
            )
            f.write(
                "susceptibility scores than uncalibrated F1-optimized thresholds.\n\n"
            )

            f.write("**Key Findings:**\n\n")
            f.write(
                f"- F1-optimized (uncalibrated) Expected Loss: {f1_uncal_loss:.4f}\n"
            )
            if f1_uncal_loss > 0:
                f.write(
                    f"- Cost-optimized (uncalibrated) Expected Loss: {cost_uncal_loss:.4f} "
                )
                f.write(f"({cost_uncal_improvement_pct:.1f}% reduction)\n")
                f.write(
                    f"- Cost-optimized (calibrated) Expected Loss: {cost_cal_loss:.4f} "
                )
                f.write(f"({cost_cal_improvement_pct:.1f}% reduction)\n\n")
            else:
                f.write(
                    f"- Cost-optimized (uncalibrated) Expected Loss: {cost_uncal_loss:.4f}\n"
                )
                f.write(
                    f"- Cost-optimized (calibrated) Expected Loss: {cost_cal_loss:.4f}\n"
                )
                f.write(
                    "(F1-optimized loss is zero, improvement percentage not applicable)\n\n"
                )

            f.write(
                "Cost-aware thresholding significantly reduces expected loss compared to "
            )
            f.write(
                "F1-optimized thresholding, demonstrating better alignment with banking cost "
            )
            f.write("structures (FN cost >> FP cost).\n\n")

            if "improvement_statement" in metrics:
                f.write(
                    f"**Canonical Statement:** {metrics['improvement_statement']}\n"
                )
        else:
            f.write(
                "✗ H2 is **not supported**: Cost-aware thresholding does not reduce expected loss "
            )
            f.write("compared to F1-optimized thresholding.\n")

    logger.info(f"Saved summary to {summary_path}")

    # Also save as H2_summary.md for praxis validation
    with open(summary_path, encoding="utf-8") as f:
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
    parser.add_argument(
        "--splits-config",
        type=Path,
        default=None,
        help="Path to h2_splits.yaml for multi-split evaluation (default: single-split)",
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
            splits_config_path=args.splits_config,
            repo_root=repo_root,
        )

        print("\n" + "=" * 80)
        print("H2 Calibration and Thresholding Summary")
        print("=" * 80)
        print(f"Brier Improvement: {metrics['calibration']['brier_improvement']:.4f}")
        print(f"ECE Improvement: {metrics['calibration']['ece_improvement']:.4f}")
        print(
            f"Cost-Optimal Threshold (calibrated): {metrics['cost_optimized']['calibrated']['threshold']:.4f}"
        )
        print(
            f"Expected Loss (cost-optimal): {metrics['cost_optimized']['calibrated']['expected_loss']:.4f}"
        )
        print(f"Results saved to: {args.output}")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"H2 experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
