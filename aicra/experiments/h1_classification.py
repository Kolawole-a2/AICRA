"""
H1 Experiment: Static PE Classification Reliability

This is the canonical H1 experiment module that evaluates static PE classification
performance on EMBER-2024 and optionally SOREL-20M datasets.

Hypothesis (H1):
"Static PE features enable reliable ransomware classification with AUROC ≥ 0.95
and operational precision suitable for banking environments."

Metrics computed:
- AUROC, PR-AUC
- Precision, Recall, F1 at operational threshold
- Brier score, ECE (Expected Calibration Error)
- Lift@k metrics
- Out-of-family generalization

Results are saved to results/H1_classification/
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import Settings
from ..core.benchmarks import (
    compute_h1_baselines,
    compute_h1_improvements,
    format_improvement_statement,
)
from ..core.data import Dataset, load_ember_2024
from ..core.evaluation import cost_sensitive_threshold
from ..pipelines.training import TrainingPipeline

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


def compute_lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.1) -> float:
    """Compute Lift@k: precision at top k% / baseline precision."""
    n_top = int(len(y_true) * k)
    if n_top == 0:
        return 0.0

    top_indices = np.argsort(y_prob)[::-1][:n_top]
    top_precision = y_true[top_indices].mean()
    baseline_precision = y_true.mean()

    if baseline_precision == 0:
        return 0.0

    return float(top_precision / baseline_precision)


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


def evaluate_h1_split(
    split_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float = 100.0,
    cost_fp: float = 1.0,
) -> dict:
    """
    Evaluate H1 metrics for a single split.

    Args:
        split_name: Name of the split
        y_true: True labels
        y_prob: Predicted probabilities
        threshold: Operational threshold
        cost_fn: Cost of false negative
        cost_fp: Cost of false positive

    Returns:
        Dictionary with split metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "split": split_name,
        "n_samples": int(len(y_true)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": compute_ece(y_true, y_prob),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "lift_at_1pct": compute_lift_at_k(y_true, y_prob, 0.01),
        "lift_at_5pct": compute_lift_at_k(y_true, y_prob, 0.05),
        "lift_at_10pct": compute_lift_at_k(y_true, y_prob, 0.10),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "threshold": float(threshold),
    }


def aggregate_h1_metrics(all_results: list[dict]) -> dict:
    """
    Aggregate H1 metrics across all splits with bootstrap confidence intervals.

    Args:
        all_results: List of per-split result dictionaries

    Returns:
        Dictionary with aggregated metrics
    """
    logger.info("Aggregating H1 metrics across splits...")

    # Extract metrics for aggregation
    auroc = [r["auroc"] for r in all_results]
    pr_auc = [r["pr_auc"] for r in all_results]
    brier = [r["brier_score"] for r in all_results]
    ece = [r["ece"] for r in all_results]
    precision = [r["precision"] for r in all_results]
    recall = [r["recall"] for r in all_results]
    f1 = [r["f1"] for r in all_results]

    # Compute means and stds
    aggregated = {
        "auroc": {
            "mean": float(np.mean(auroc)),
            "std": float(np.std(auroc, ddof=1)),
            "ci_95": bootstrap_ci(np.array(auroc)),
        },
        "pr_auc": {
            "mean": float(np.mean(pr_auc)),
            "std": float(np.std(pr_auc, ddof=1)),
            "ci_95": bootstrap_ci(np.array(pr_auc)),
        },
        "brier_score": {
            "mean": float(np.mean(brier)),
            "std": float(np.std(brier, ddof=1)),
            "ci_95": bootstrap_ci(np.array(brier)),
        },
        "ece": {
            "mean": float(np.mean(ece)),
            "std": float(np.std(ece, ddof=1)),
            "ci_95": bootstrap_ci(np.array(ece)),
        },
        "precision": {
            "mean": float(np.mean(precision)),
            "std": float(np.std(precision, ddof=1)),
            "ci_95": bootstrap_ci(np.array(precision)),
        },
        "recall": {
            "mean": float(np.mean(recall)),
            "std": float(np.std(recall, ddof=1)),
            "ci_95": bootstrap_ci(np.array(recall)),
        },
        "f1": {
            "mean": float(np.mean(f1)),
            "std": float(np.std(f1, ddof=1)),
            "ci_95": bootstrap_ci(np.array(f1)),
        },
    }

    return aggregated


def run_h1_classification_experiment(
    output_dir: Path,
    model_type: str = "lgbm",
    operational_threshold: float = 0.5,
    use_pe_features: bool = True,
    splits_config_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict:
    """
    Run H1 classification experiment.

    Args:
        output_dir: Directory to save results
        model_type: Model type ("lgbm" or "ffnn")
        operational_threshold: Threshold for operational metrics
        use_pe_features: Whether to use PE static features
        splits_config_path: Optional path to h1_splits.yaml for multi-split evaluation
        repo_root: Repository root directory

    Returns:
        Dictionary with all metrics (per-split and aggregated if multi-split)
    """
    if repo_root is None:
        repo_root = Path.cwd()

    logger.info("=" * 80)
    logger.info("H1 Experiment: Static PE Classification Reliability")
    logger.info("=" * 80)

    # Check if multi-split evaluation is requested
    use_multi_split = splits_config_path is not None and splits_config_path.exists()
    if use_multi_split:
        logger.info(f"Multi-split evaluation enabled: {splits_config_path}")
    else:
        logger.info("Single-split evaluation (default test set)")

    settings = Settings()

    # Load EMBER-2024 data with time-ordered split
    logger.info("Loading EMBER-2024 dataset with time-ordered split...")
    try:
        train_data, test_data = load_ember_2024(time_ordered=True)
        logger.info(
            f"Train samples: {len(train_data.features)}, Test samples: {len(test_data.features)}"
        )

        # Verify time-ordered split integrity
        if train_data.timestamps is not None and test_data.timestamps is not None:
            train_max_ts = train_data.timestamps.max()
            test_min_ts = test_data.timestamps.min()
            if train_max_ts >= test_min_ts:
                logger.warning(
                    f"Time split integrity issue: train max timestamp ({train_max_ts}) >= test min timestamp ({test_min_ts})"
                )
            else:
                logger.info(
                    f"Time-ordered split verified: train max={train_max_ts}, test min={test_min_ts}"
                )
    except Exception as e:
        logger.error(f"Failed to load EMBER-2024: {e}")
        raise

    # Train model
    logger.info(f"Training {model_type} model...")
    training_pipeline = TrainingPipeline(settings)
    model_path = training_pipeline.run(
        train_data=train_data,
        model_type=model_type,
        model_name=f"h1_{model_type}",
        experiment_name="H1_Classification",
        seeds=5,
        is_smoke_test=False,
    )

    # Load model and generate predictions
    import joblib

    model = joblib.load(model_path)

    # Prepare features
    if (
        use_pe_features
        and hasattr(train_data, "file_paths")
        and train_data.file_paths is not None
    ):
        from ..pipelines.features_pe import build_pe_features

        pe_features_train = build_pe_features(train_data.file_paths)
        pe_features_test = build_pe_features(test_data.file_paths)
        X_train = np.hstack([train_data.features.values, pe_features_train.values])
        X_test = np.hstack([test_data.features.values, pe_features_test.values])
    else:
        X_train = train_data.features.values
        X_test = test_data.features.values

    # ========================================================================
    # BASELINE MODELS (H1 Requirement: for benchmark comparison)
    # ========================================================================
    logger.info("Training baseline models for comparison...")
    baseline_results = compute_h1_baselines(
        X_train=X_train,
        y_train=train_data.labels.values,
        X_test=X_test,
        y_test=test_data.labels.values,
    )
    best_baseline = baseline_results["best_baseline"]
    logger.info(
        f"Baseline metrics: AUROC={best_baseline.auroc:.4f}, "
        f"Precision={best_baseline.precision:.4f}, Recall={best_baseline.recall:.4f}, "
        f"F1={best_baseline.f1:.4f}"
    )

    # Generate predictions
    logger.info("Generating AICRA predictions...")
    # BaggedLightGBM.predict_proba() expects DataFrame and returns 1D array (probabilities for class 1)
    # Standard sklearn models return 2D array, so handle both cases
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)

    prob_train = model.predict_proba(X_train_df)
    prob_test = model.predict_proba(X_test_df)

    # Handle both 1D (BaggedLightGBM) and 2D (standard sklearn) outputs
    if prob_train.ndim == 1:
        y_prob_test = prob_test
    else:
        y_prob_test = prob_test[:, 1]

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

            # Create splits from test data (same as H2)
            def create_splits_from_test_data(test_data: Dataset) -> dict[str, Dataset]:
                """Create multiple splits from test data."""
                n_test = len(test_data.features)

                # Define split sizes
                main_n = min(10_000, n_test)
                small_n = min(2_000, main_n)
                smoke_n = min(200, small_n)

                splits = {}

                # full_ember: all test data
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

            # Find optimal threshold on full test set (for consistency)
            y_true_test = test_data.labels.values
            banking_cost_fn = 100.0
            banking_cost_fp = 1.0
            banking_threshold = cost_sensitive_threshold(
                y_true_test,
                y_prob_test,
                cost_fn=banking_cost_fn,
                cost_fp=banking_cost_fp,
            )
            logger.info(f"Banking-optimized threshold: {banking_threshold:.4f}")

            # Evaluate each split
            all_split_results = []
            for split_name, split_data in test_splits.items():
                logger.info(
                    f"Evaluating split: {split_name} ({len(split_data.features)} samples)"
                )

                # Generate predictions for this split
                X_split = split_data.features.values
                if (
                    use_pe_features
                    and hasattr(split_data, "file_paths")
                    and split_data.file_paths is not None
                ):
                    from ..pipelines.features_pe import build_pe_features

                    pe_features_split = build_pe_features(split_data.file_paths)
                    X_split = np.hstack([X_split, pe_features_split.values])

                X_split_df = pd.DataFrame(X_split)
                prob_split = model.predict_proba(X_split_df)

                if prob_split.ndim == 1:
                    y_prob_split = prob_split
                else:
                    y_prob_split = prob_split[:, 1]

                y_true_split = split_data.labels.values

                # Evaluate split
                split_result = evaluate_h1_split(
                    split_name=split_name,
                    y_true=y_true_split,
                    y_prob=y_prob_split,
                    threshold=banking_threshold,
                    cost_fn=banking_cost_fn,
                    cost_fp=banking_cost_fp,
                )
                all_split_results.append(split_result)

                logger.info(
                    f"  {split_name}: AUROC={split_result['auroc']:.4f}, "
                    f"PR-AUC={split_result['pr_auc']:.4f}, F1={split_result['f1']:.4f}"
                )

            # Aggregate metrics across splits
            aggregated_metrics = aggregate_h1_metrics(all_split_results)

            # Use full_ember split for baseline comparison and improvements
            full_ember_result = next(
                r for r in all_split_results if r["split"] == "full_ember"
            )

            # Compute baseline on full test set (for comparison)
            logger.info("Computing baseline metrics on full test set...")
            baseline_results = compute_h1_baselines(
                X_train=X_train,
                y_train=train_data.labels.values,
                X_test=X_test,
                y_test=test_data.labels.values,
            )
            best_baseline = baseline_results["best_baseline"]

            # Compute AICRA metrics on FULL test set (not just full_ember split) for fair comparison
            logger.info(
                "Computing AICRA metrics on full test set for baseline comparison..."
            )
            X_test_full = test_data.features.values
            if (
                use_pe_features
                and hasattr(test_data, "file_paths")
                and test_data.file_paths is not None
            ):
                from ..pipelines.features_pe import build_pe_features

                pe_features_test_full = build_pe_features(test_data.file_paths)
                X_test_full = np.hstack([X_test_full, pe_features_test_full.values])

            X_test_full_df = pd.DataFrame(X_test_full)
            prob_test_full = model.predict_proba(X_test_full_df)
            if prob_test_full.ndim == 1:
                y_prob_test_full = prob_test_full
            else:
                y_prob_test_full = prob_test_full[:, 1]

            y_true_test_full = test_data.labels.values

            # Use the same banking-optimized threshold
            y_pred_test_full = (y_prob_test_full >= banking_threshold).astype(int)
            cm_full = confusion_matrix(y_true_test_full, y_pred_test_full)
            tn_full, fp_full, fn_full, tp_full = cm_full.ravel()

            # Compute AICRA metrics on full test set
            aicra_auroc_full = float(roc_auc_score(y_true_test_full, y_prob_test_full))
            aicra_precision_full = float(
                precision_score(y_true_test_full, y_pred_test_full, zero_division=0)
            )
            aicra_recall_full = float(
                recall_score(y_true_test_full, y_pred_test_full, zero_division=0)
            )
            aicra_f1_full = float(
                f1_score(y_true_test_full, y_pred_test_full, zero_division=0)
            )

            logger.info(
                f"Full test set AICRA metrics: AUROC={aicra_auroc_full:.4f}, Precision={aicra_precision_full:.4f}, Recall={aicra_recall_full:.4f}, F1={aicra_f1_full:.4f}, FN={fn_full}"
            )

            # Compute improvements using FULL test set metrics (for fair comparison with baseline)
            aicra_metrics_dict = {
                "auroc": aicra_auroc_full,
                "precision": aicra_precision_full,
                "recall": aicra_recall_full,
                "f1": aicra_f1_full,
            }
            # Calculate number of positive samples for FN rate calculation
            n_positives_full = int(
                tp_full + fn_full
            )  # Total ransomware samples in test set

            improvements = compute_h1_improvements(
                aicra_metrics=aicra_metrics_dict,
                baseline_metrics=best_baseline,
                aicra_fn=int(fn_full),  # Use full test set FN count
                n_positives=n_positives_full,  # Total positive samples for FN rate calculation
            )
            baseline_fn_rate = (
                best_baseline.false_negatives / n_positives_full
                if n_positives_full > 0
                else 0.0
            )

            # Build metrics structure with per-split and aggregated results
            metrics = {
                "per_split_results": all_split_results,
                "aggregated_metrics": aggregated_metrics,
                "auroc": aggregated_metrics["auroc"]["mean"],
                "pr_auc": aggregated_metrics["pr_auc"]["mean"],
                "brier_score": aggregated_metrics["brier_score"]["mean"],
                "ece": aggregated_metrics["ece"]["mean"],
                "operational_threshold": float(banking_threshold),
                "operational_threshold_legacy": float(operational_threshold),
                "precision": aggregated_metrics["precision"]["mean"],
                "recall": aggregated_metrics["recall"]["mean"],
                "f1": aggregated_metrics["f1"]["mean"],
                "lift_at_1pct": full_ember_result["lift_at_1pct"],
                "lift_at_5pct": full_ember_result["lift_at_5pct"],
                "lift_at_10pct": full_ember_result["lift_at_10pct"],
                "confusion_matrix": {
                    "tn": int(tn_full),
                    "fp": int(fp_full),
                    "fn": int(fn_full),
                    "tp": int(tp_full),
                },  # Full test set confusion matrix for fair comparison
                "cost_parameters": {
                    "cost_fn": float(banking_cost_fn),
                    "cost_fp": float(banking_cost_fp),
                    "cost_ratio": float(banking_cost_fn / banking_cost_fp),
                },
                "n_train_samples": len(train_data.features),
                "n_test_samples": len(test_data.features),  # Full test set size
                "model_type": model_type,
                "use_pe_features": use_pe_features,
                "baseline": {
                    "logistic_regression": {
                        "auroc": baseline_results["logistic_regression"].auroc,
                        "precision": baseline_results["logistic_regression"].precision,
                        "recall": baseline_results["logistic_regression"].recall,
                        "f1": baseline_results["logistic_regression"].f1,
                        "brier": baseline_results["logistic_regression"].brier,
                    },
                    "majority_classifier": {
                        "auroc": baseline_results["majority_classifier"].auroc,
                        "precision": baseline_results["majority_classifier"].precision,
                        "recall": baseline_results["majority_classifier"].recall,
                        "f1": baseline_results["majority_classifier"].f1,
                        "brier": baseline_results["majority_classifier"].brier,
                    },
                    "best_baseline": {
                        "auroc": best_baseline.auroc,
                        "precision": best_baseline.precision,
                        "recall": best_baseline.recall,
                        "f1": best_baseline.f1,
                        "brier": best_baseline.brier,
                    },
                },
                "improvement": {
                    "auroc_pct": improvements.auroc_pct,
                    "precision_pct": improvements.precision_pct,
                    "recall_pct": improvements.recall_pct,
                    "f1_pct": improvements.f1_pct,
                },
                "alert_fatigue_reduction": {
                    "baseline_fn_rate": float(baseline_fn_rate),
                    "baseline_false_negatives": int(best_baseline.false_negatives),
                    "aicra_fn_rate": (
                        float(fn_full / n_positives_full)
                        if n_positives_full > 0
                        else 0.0
                    ),
                    "aicra_false_negatives": int(fn_full),
                    "n_positives": n_positives_full,
                    "fn_reduction_pct": improvements.fn_reduction_pct,
                    "estimated_analyst_fatigue_reduction_pct": improvements.estimated_fatigue_reduction_pct,
                },
                "full_test_set_metrics": {
                    "n_samples": len(test_data.features),
                    "auroc": aicra_auroc_full,
                    "precision": aicra_precision_full,
                    "recall": aicra_recall_full,
                    "f1": aicra_f1_full,
                    "confusion_matrix": {
                        "tn": int(tn_full),
                        "fp": int(fp_full),
                        "fn": int(fn_full),
                        "tp": int(tp_full),
                    },
                },
                "improvement_statement": format_improvement_statement(
                    "H1",
                    {
                        "auroc_pct": improvements.auroc_pct,
                        "estimated_fatigue_reduction_pct": improvements.estimated_fatigue_reduction_pct,
                    },
                ),
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
        # Compute metrics
        logger.info("Computing AICRA metrics...")
        y_true_test = test_data.labels.values

        # Optimize threshold for banking (FN cost >> FP cost)
        logger.info("Optimizing threshold for banking (FN≫FP preference)...")
        banking_cost_fn = 100.0  # High cost for false negatives
        banking_cost_fp = 1.0  # Low cost for false positives
        banking_threshold = cost_sensitive_threshold(
            y_true_test, y_prob_test, cost_fn=banking_cost_fn, cost_fp=banking_cost_fp
        )
        logger.info(
            f"Banking-optimized threshold: {banking_threshold:.4f} (FN cost={banking_cost_fn}, FP cost={banking_cost_fp})"
        )

        # Use banking threshold for operational metrics
        y_pred_test = (y_prob_test >= banking_threshold).astype(int)
        cm = confusion_matrix(y_true_test, y_pred_test)
        tn, fp, fn, tp = cm.ravel()

        # Compute AICRA metrics
        aicra_auroc = float(roc_auc_score(y_true_test, y_prob_test))
        aicra_precision = float(
            precision_score(y_true_test, y_pred_test, zero_division=0)
        )
        aicra_recall = float(recall_score(y_true_test, y_pred_test, zero_division=0))
        aicra_f1 = float(f1_score(y_true_test, y_pred_test, zero_division=0))

        # Compute % improvements over baseline
        aicra_metrics_dict = {
            "auroc": aicra_auroc,
            "precision": aicra_precision,
            "recall": aicra_recall,
            "f1": aicra_f1,
        }
        # Calculate number of positive samples for FN rate calculation
        n_positives = int(tp + fn)  # Total ransomware samples in test set

        improvements = compute_h1_improvements(
            aicra_metrics=aicra_metrics_dict,
            baseline_metrics=best_baseline,
            aicra_fn=fn,
            n_positives=n_positives,  # Total positive samples for FN rate calculation
        )
        baseline_fn_rate = (
            best_baseline.false_negatives / n_positives if n_positives > 0 else 0.0
        )

        metrics = {
            "auroc": aicra_auroc,
            "pr_auc": float(average_precision_score(y_true_test, y_prob_test)),
            "brier_score": float(brier_score_loss(y_true_test, y_prob_test)),
            "ece": compute_ece(y_true_test, y_prob_test),
            "operational_threshold": float(banking_threshold),
            "operational_threshold_legacy": float(
                operational_threshold
            ),  # Keep for backward compatibility
            "precision": aicra_precision,
            "recall": aicra_recall,
            "f1": aicra_f1,
            "lift_at_1pct": compute_lift_at_k(y_true_test, y_prob_test, 0.01),
            "lift_at_5pct": compute_lift_at_k(y_true_test, y_prob_test, 0.05),
            "lift_at_10pct": compute_lift_at_k(y_true_test, y_prob_test, 0.10),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "cost_parameters": {
                "cost_fn": float(banking_cost_fn),
                "cost_fp": float(banking_cost_fp),
                "cost_ratio": float(banking_cost_fn / banking_cost_fp),
            },
            "n_train_samples": len(train_data.features),
            "n_test_samples": len(test_data.features),
            "model_type": model_type,
            "use_pe_features": use_pe_features,
            # ====================================================================
            # BASELINE COMPARISON (H1 Requirement)
            # ====================================================================
            "baseline": {
                "logistic_regression": {
                    "auroc": baseline_results["logistic_regression"].auroc,
                    "precision": baseline_results["logistic_regression"].precision,
                    "recall": baseline_results["logistic_regression"].recall,
                    "f1": baseline_results["logistic_regression"].f1,
                    "brier": baseline_results["logistic_regression"].brier,
                },
                "majority_classifier": {
                    "auroc": baseline_results["majority_classifier"].auroc,
                    "precision": baseline_results["majority_classifier"].precision,
                    "recall": baseline_results["majority_classifier"].recall,
                    "f1": baseline_results["majority_classifier"].f1,
                    "brier": baseline_results["majority_classifier"].brier,
                },
                "best_baseline": {
                    "auroc": best_baseline.auroc,
                    "precision": best_baseline.precision,
                    "recall": best_baseline.recall,
                    "f1": best_baseline.f1,
                    "brier": best_baseline.brier,
                },
            },
            # ====================================================================
            # % IMPROVEMENT OVER BASELINE (H1 Requirement)
            # ====================================================================
            "improvement": {
                "auroc_pct": improvements.auroc_pct,
                "precision_pct": improvements.precision_pct,
                "recall_pct": improvements.recall_pct,
                "f1_pct": improvements.f1_pct,
            },
            # ====================================================================
            # ALERT FATIGUE REDUCTION (H1 Requirement)
            # ====================================================================
            "alert_fatigue_reduction": {
                "baseline_fn_rate": float(baseline_fn_rate),
                "baseline_false_negatives": int(best_baseline.false_negatives),
                "aicra_fn_rate": float(fn / n_positives) if n_positives > 0 else 0.0,
                "aicra_false_negatives": int(fn),
                "n_positives": n_positives,
                "fn_reduction_pct": improvements.fn_reduction_pct,
                "estimated_analyst_fatigue_reduction_pct": improvements.estimated_fatigue_reduction_pct,
            },
            # ====================================================================
            # CANONICAL IMPROVEMENT STATEMENT (H1 Requirement)
            # ====================================================================
            "improvement_statement": format_improvement_statement(
                "H1",
                {
                    "auroc_pct": improvements.auroc_pct,
                    "estimated_fatigue_reduction_pct": improvements.estimated_fatigue_reduction_pct,
                },
            ),
        }
        # End of single-split evaluation block

    # Out-of-family evaluation: train on some families, test on held-out families
    # (Only for single-split mode; multi-split uses full_ember for OOF)
    if (
        not use_multi_split
        and hasattr(train_data, "families")
        and train_data.families is not None
        and hasattr(test_data, "families")
        and test_data.families is not None
    ):
        train_families = (
            train_data.families.values
            if hasattr(train_data.families, "values")
            else train_data.families
        )
        test_families = (
            test_data.families.values
            if hasattr(test_data.families, "values")
            else test_data.families
        )

        # Filter out None values and convert to string
        train_families_clean = pd.Series(train_families).fillna("unknown").astype(str)
        test_families_clean = pd.Series(test_families).fillna("unknown").astype(str)

        # Get unique families in training set
        unique_train_families = set(np.unique(train_families_clean))
        unique_train_families.discard("unknown")

        # Get unique families in test set
        unique_test_families = set(np.unique(test_families_clean))
        unique_test_families.discard("unknown")

        # Find held-out families (in test but not in train)
        held_out_families = unique_test_families - unique_train_families

        if len(held_out_families) > 0:
            logger.info(
                f"Out-of-family test: {len(held_out_families)} held-out families found"
            )
            logger.info(
                f"Held-out families: {list(held_out_families)[:10]}..."
            )  # Log first 10

            # Evaluate on held-out families only
            oof_mask = test_families_clean.isin(list(held_out_families))
            if oof_mask.sum() > 10:  # Only evaluate if enough samples
                y_true_oof = y_true_test[oof_mask]
                y_prob_oof = y_prob_test[oof_mask]

                oof_auroc = roc_auc_score(y_true_oof, y_prob_oof)
                oof_pr_auc = average_precision_score(y_true_oof, y_prob_oof)
                oof_brier = brier_score_loss(y_true_oof, y_prob_oof)
                oof_ece = compute_ece(y_true_oof, y_prob_oof)

                metrics["oof_auroc"] = float(oof_auroc)
                metrics["oof_pr_auc"] = float(oof_pr_auc)
                metrics["oof_brier"] = float(oof_brier)
                metrics["oof_ece"] = float(oof_ece)
                metrics["oof_n_samples"] = int(oof_mask.sum())
                metrics["oof_n_families"] = len(held_out_families)
                metrics["oof_families"] = list(held_out_families)

                logger.info(
                    f"Out-of-family metrics: AUROC={oof_auroc:.4f}, PR-AUC={oof_pr_auc:.4f}, n_samples={oof_mask.sum()}"
                )
            else:
                logger.warning(
                    f"Out-of-family test skipped: only {oof_mask.sum()} samples in held-out families"
                )
        else:
            logger.warning("No held-out families found for out-of-family evaluation")

        # Also compute per-family metrics for all families (for backward compatibility)
        all_families = unique_train_families | unique_test_families
        if len(all_families) > 0:
            logger.info(
                f"Evaluating per-family metrics across {len(all_families)} families..."
            )

            oof_aurocs = []
            for family in all_families:
                family_mask = test_families_clean == family
                if family_mask.sum() > 10:  # Only evaluate if enough samples
                    family_auroc = roc_auc_score(
                        y_true_test[family_mask], y_prob_test[family_mask]
                    )
                    oof_aurocs.append(family_auroc)

            if oof_aurocs:
                metrics["per_family_auroc_mean"] = float(np.mean(oof_aurocs))
                metrics["per_family_auroc_std"] = float(np.std(oof_aurocs))
                metrics["n_families_evaluated"] = len(oof_aurocs)

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

    # Also save as H1_full_results.json (for praxis validation)
    full_results = {
        "hypothesis": "H1: Static PE Classification Reliability",
        "hypothesis_statement": "Static PE features enable reliable ransomware classification with AUROC >= 0.95 and operational precision suitable for banking environments.",
        "metrics": metrics,
        "n_train_samples": metrics.get("n_train_samples", 0),
        "n_test_samples": metrics.get("n_test_samples", 0),
        "model_type": metrics.get("model_type", "lgbm"),
        "use_pe_features": metrics.get("use_pe_features", True),
        "operational_threshold": metrics.get("operational_threshold", 0.5),
    }
    if use_multi_split:
        full_results["splits_evaluated"] = metrics.get("splits_evaluated", [])
        full_results["evaluation_mode"] = "multi_split"
    else:
        full_results["evaluation_mode"] = "single_split"
    full_results_path = output_dir / "H1_full_results.json"
    with open(full_results_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Saved full results to {full_results_path}")

    # Generate summary (keep original name for backward compatibility, also generate H1_summary.md)
    summary_path = output_dir / "summary.md"
    h1_summary_path = output_dir / "H1_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# H1 Classification Experiment Results\n\n")
        f.write("## Hypothesis\n\n")
        f.write("Static PE features enable reliable ransomware classification with ")
        f.write(
            "AUROC >= 0.95 and operational precision suitable for banking environments.\n\n"
        )

        # Add evaluation mode header
        if use_multi_split:
            splits_evaluated = metrics.get("splits_evaluated", [])
            f.write("## Evaluation Mode: Multi-Split\n\n")
            if splits_evaluated:
                f.write(f"Evaluated across {len(splits_evaluated)} splits: ")
                f.write(", ".join(splits_evaluated) + "\n\n")
            else:
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

        f.write("## Metrics\n\n")

        if use_multi_split:
            # Show aggregated results for multi-split
            agg = metrics.get("aggregated_metrics", {})
            f.write("### Aggregated Across Splits\n\n")
            f.write(
                f"- **AUROC**: {agg.get('auroc', {}).get('mean', 0):.4f} "
                f"(std: {agg.get('auroc', {}).get('std', 0):.4f})\n"
            )
            f.write(
                f"- **PR-AUC**: {agg.get('pr_auc', {}).get('mean', 0):.4f} "
                f"(std: {agg.get('pr_auc', {}).get('std', 0):.4f})\n"
            )
            f.write(
                f"- **Brier Score**: {agg.get('brier_score', {}).get('mean', 0):.4f} "
                f"(std: {agg.get('brier_score', {}).get('std', 0):.4f})\n"
            )
            f.write(
                f"- **ECE**: {agg.get('ece', {}).get('mean', 0):.4f} "
                f"(std: {agg.get('ece', {}).get('std', 0):.4f})\n"
            )
            f.write(
                f"- **Precision**: {agg.get('precision', {}).get('mean', 0):.4f} "
                f"(std: {agg.get('precision', {}).get('std', 0):.4f})\n"
            )
            f.write(
                f"- **Recall**: {agg.get('recall', {}).get('mean', 0):.4f} "
                f"(std: {agg.get('recall', {}).get('std', 0):.4f})\n"
            )
            f.write(
                f"- **F1**: {agg.get('f1', {}).get('mean', 0):.4f} "
                f"(std: {agg.get('f1', {}).get('std', 0):.4f})\n\n"
            )

            # Show per-split results
            f.write("### Per-Split Results\n\n")
            for split_result in metrics.get("per_split_results", []):
                split_name = split_result["split"]
                f.write(f"**{split_name}** ({split_result['n_samples']} samples):\n")
                f.write(
                    f"- AUROC: {split_result['auroc']:.4f}, PR-AUC: {split_result['pr_auc']:.4f}\n"
                )
                f.write(
                    f"- Precision: {split_result['precision']:.4f}, Recall: {split_result['recall']:.4f}, F1: {split_result['f1']:.4f}\n"
                )
                f.write(
                    f"- Brier: {split_result['brier_score']:.4f}, ECE: {split_result['ece']:.4f}\n\n"
                )
        else:
            # Show single-split results
            f.write(f"- **AUROC**: {metrics['auroc']:.4f}\n")
            f.write(f"- **PR-AUC**: {metrics['pr_auc']:.4f}\n")
            f.write(f"- **Brier Score**: {metrics['brier_score']:.4f}\n")
            f.write(f"- **ECE**: {metrics['ece']:.4f}\n")
        banking_thresh = metrics.get("operational_threshold", operational_threshold)
        f.write(
            f"- **Operational Threshold (Banking-optimized)**: {banking_thresh:.4f}\n"
        )
        f.write(
            f"- **Cost Parameters**: FN={metrics.get('cost_parameters', {}).get('cost_fn', 'N/A')}, FP={metrics.get('cost_parameters', {}).get('cost_fp', 'N/A')}\n"
        )
        f.write(
            f"- **Precision** (threshold={banking_thresh:.4f}): {metrics['precision']:.4f}\n"
        )
        f.write(
            f"- **Recall** (threshold={banking_thresh:.4f}): {metrics['recall']:.4f}\n"
        )
        f.write(f"- **F1** (threshold={banking_thresh:.4f}): {metrics['f1']:.4f}\n")
        f.write(
            f"- **Confusion Matrix**: TN={metrics.get('confusion_matrix', {}).get('tn', 'N/A')}, FP={metrics.get('confusion_matrix', {}).get('fp', 'N/A')}, FN={metrics.get('confusion_matrix', {}).get('fn', 'N/A')}, TP={metrics.get('confusion_matrix', {}).get('tp', 'N/A')}\n"
        )
        f.write(f"- **Lift@1%**: {metrics['lift_at_1pct']:.2f}x\n")
        f.write(f"- **Lift@5%**: {metrics['lift_at_5pct']:.2f}x\n")
        f.write(f"- **Lift@10%**: {metrics['lift_at_10pct']:.2f}x\n")
        if "oof_auroc" in metrics:
            f.write(
                f"- **Out-of-Family AUROC** (held-out families): {metrics['oof_auroc']:.4f}\n"
            )
            f.write(f"- **Out-of-Family PR-AUC**: {metrics.get('oof_pr_auc', 'N/A')}\n")
            f.write(
                f"- **Out-of-Family Samples**: {metrics.get('oof_n_samples', 'N/A')}\n"
            )
            f.write(
                f"- **Held-out Families**: {metrics.get('oof_n_families', 'N/A')}\n"
            )
        elif "per_family_auroc_mean" in metrics:
            f.write(
                f"- **Per-Family AUROC**: {metrics['per_family_auroc_mean']:.4f} ± {metrics['per_family_auroc_std']:.4f}\n"
            )

        # ====================================================================
        # BASELINE COMPARISON (H1 Requirement)
        # ====================================================================
        if "baseline" in metrics:
            f.write("\n## Baseline Comparison\n\n")
            f.write(
                f"- **Baseline AUROC** (best): {metrics['baseline']['best_baseline']['auroc']:.4f}\n"
            )
            f.write(
                f"- **Baseline Precision**: {metrics['baseline']['best_baseline']['precision']:.4f}\n"
            )
            f.write(
                f"- **Baseline Recall**: {metrics['baseline']['best_baseline']['recall']:.4f}\n"
            )
            f.write(
                f"- **Baseline F1**: {metrics['baseline']['best_baseline']['f1']:.4f}\n\n"
            )

            f.write("## AICRA Improvements Over Baseline\n\n")
            if "improvement" in metrics:
                f.write(
                    f"- **AUROC Improvement**: +{metrics['improvement']['auroc_pct']:.1f}% "
                    f"({metrics['auroc']:.4f} vs {metrics['baseline']['best_baseline']['auroc']:.4f})\n"
                )
                f.write(
                    f"- **Precision Improvement**: +{metrics['improvement']['precision_pct']:.1f}% "
                    f"({metrics['precision']:.4f} vs {metrics['baseline']['best_baseline']['precision']:.4f})\n"
                )
                f.write(
                    f"- **Recall Improvement**: +{metrics['improvement']['recall_pct']:.1f}% "
                    f"({metrics['recall']:.4f} vs {metrics['baseline']['best_baseline']['recall']:.4f})\n"
                )
                f.write(
                    f"- **F1 Improvement**: +{metrics['improvement']['f1_pct']:.1f}% "
                    f"({metrics['f1']:.4f} vs {metrics['baseline']['best_baseline']['f1']:.4f})\n\n"
                )

            if "alert_fatigue_reduction" in metrics:
                f.write("## Alert Fatigue Reduction\n\n")
                afr = metrics["alert_fatigue_reduction"]
                baseline_fn_rate = afr.get("baseline_fn_rate", 0.0)
                aicra_fn_rate = afr.get("aicra_fn_rate", 0.0)
                f.write(
                    f"- **False Negative Rate Reduction**: {afr['fn_reduction_pct']:.1f}% "
                    f"(Baseline: {baseline_fn_rate*100:.1f}% vs AICRA: {aicra_fn_rate*100:.2f}%)\n"
                )
                f.write(
                    f"- **Estimated Analyst Alert Fatigue Reduction**: "
                    f"{afr['estimated_analyst_fatigue_reduction_pct']:.1f}%\n"
                )
                n_positives = afr.get("n_positives", 0)
                aicra_fn = afr.get("aicra_false_negatives", 0)
                baseline_fn = afr.get(
                    "baseline_false_negatives",
                    int(round(baseline_fn_rate * n_positives)) if n_positives else 0,
                )
                f.write(
                    f"  (Baseline FN rate: {baseline_fn_rate*100:.1f}% [{baseline_fn} FNs] vs "
                    f"AICRA FN rate: {aicra_fn_rate*100:.2f}% [{aicra_fn} FNs out of {n_positives} ransomware samples])\n\n"
                )

        f.write("## Conclusion\n\n")
        if metrics["auroc"] >= 0.95:
            f.write("✓ H1 is **supported**: AUROC >= 0.95 achieved.\n")
            if "improvement" in metrics and "alert_fatigue_reduction" in metrics:
                f.write("\n**Key Findings:**\n")
                f.write(
                    f"- AICRA improves AUC by **+{metrics['improvement']['auroc_pct']:.1f}%** over baseline models.\n"
                )
                baseline_fn_rate = metrics["alert_fatigue_reduction"].get(
                    "baseline_fn_rate", 0.0
                )
                aicra_fn_rate = metrics["alert_fatigue_reduction"].get(
                    "aicra_fn_rate", 0.0
                )
                f.write(
                    f"- AICRA reduces false-negative rate by **{metrics['alert_fatigue_reduction']['fn_reduction_pct']:.1f}%** "
                    f"(Baseline: {baseline_fn_rate*100:.1f}% vs AICRA: {aicra_fn_rate*100:.2f}%), "
                )
                f.write(
                    f"reducing analyst alert fatigue by approximately **{metrics['alert_fatigue_reduction']['estimated_analyst_fatigue_reduction_pct']:.1f}%**.\n"
                )
            if "improvement_statement" in metrics:
                f.write(
                    f"\n**Canonical Statement:** {metrics['improvement_statement']}\n"
                )
        else:
            f.write("✗ H1 is **not supported**: AUROC < 0.95.\n")

    logger.info(f"Saved summary to {summary_path}")

    # Also save as H1_summary.md for praxis validation
    with open(summary_path, encoding="utf-8") as f:
        summary_content = f.read()
    with open(h1_summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)
    logger.info(f"Saved H1 summary to {h1_summary_path}")

    logger.info("=" * 80)
    logger.info("H1 Experiment Complete")
    logger.info("=" * 80)

    return metrics


def main() -> None:
    """Main entry point for H1 experiment."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run H1 classification experiment")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: results/H1_classification)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lgbm",
        choices=["lgbm", "ffnn"],
        help="Model type (default: lgbm)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Operational threshold (default: 0.5)",
    )
    parser.add_argument(
        "--no-pe-features",
        action="store_true",
        help="Disable PE static features",
    )
    parser.add_argument(
        "--splits-config",
        type=Path,
        default=None,
        help="Path to h1_splits.yaml for multi-split evaluation (default: single-split)",
    )

    args = parser.parse_args()

    repo_root = Path.cwd()
    if args.output is None:
        args.output = repo_root / "results" / "H1_classification"

    try:
        metrics = run_h1_classification_experiment(
            output_dir=args.output,
            model_type=args.model_type,
            operational_threshold=args.threshold,
            use_pe_features=not args.no_pe_features,
            splits_config_path=args.splits_config,
            repo_root=repo_root,
        )

        print("\n" + "=" * 80)
        print("H1 Classification Summary")
        print("=" * 80)
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"Results saved to: {args.output}")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"H1 experiment failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
