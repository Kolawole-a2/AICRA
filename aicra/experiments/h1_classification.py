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
from ..core.data import load_ember_2024
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


def run_h1_classification_experiment(
    output_dir: Path,
    model_type: str = "lgbm",
    operational_threshold: float = 0.5,
    use_pe_features: bool = True,
    repo_root: Path | None = None,
) -> dict:
    """
    Run H1 classification experiment.

    Args:
        output_dir: Directory to save results
        model_type: Model type ("lgbm" or "ffnn")
        operational_threshold: Threshold for operational metrics
        use_pe_features: Whether to use PE static features
        repo_root: Repository root directory

    Returns:
        Dictionary with all metrics
    """
    if repo_root is None:
        repo_root = Path.cwd()

    logger.info("=" * 80)
    logger.info("H1 Experiment: Static PE Classification Reliability")
    logger.info("=" * 80)

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
    aicra_precision = float(precision_score(y_true_test, y_pred_test, zero_division=0))
    aicra_recall = float(recall_score(y_true_test, y_pred_test, zero_division=0))
    aicra_f1 = float(f1_score(y_true_test, y_pred_test, zero_division=0))

    # Compute % improvements over baseline
    aicra_metrics_dict = {
        "auroc": aicra_auroc,
        "precision": aicra_precision,
        "recall": aicra_recall,
        "f1": aicra_f1,
    }
    improvements = compute_h1_improvements(
        aicra_metrics=aicra_metrics_dict,
        baseline_metrics=best_baseline,
        aicra_fn=fn,
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
                "false_negatives": baseline_results[
                    "logistic_regression"
                ].false_negatives,
            },
            "majority_classifier": {
                "auroc": baseline_results["majority_classifier"].auroc,
                "precision": baseline_results["majority_classifier"].precision,
                "recall": baseline_results["majority_classifier"].recall,
                "f1": baseline_results["majority_classifier"].f1,
                "brier": baseline_results["majority_classifier"].brier,
                "false_negatives": baseline_results[
                    "majority_classifier"
                ].false_negatives,
            },
            "best_baseline": {
                "auroc": best_baseline.auroc,
                "precision": best_baseline.precision,
                "recall": best_baseline.recall,
                "f1": best_baseline.f1,
                "brier": best_baseline.brier,
                "false_negatives": best_baseline.false_negatives,
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
            "baseline_false_negatives": best_baseline.false_negatives,
            "aicra_false_negatives": int(fn),
            "fn_reduction_absolute": int(best_baseline.false_negatives - fn)
            if best_baseline.false_negatives is not None
            else 0,
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

    # Out-of-family evaluation: train on some families, test on held-out families
    if (
        hasattr(train_data, "families")
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

    # Save results in full_results.json format (for consistency with H3)
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
        f.write("## Metrics\n\n")
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
                f.write(
                    f"- **False Negatives Reduced**: {metrics['alert_fatigue_reduction']['fn_reduction_absolute']} "
                    f"({metrics['alert_fatigue_reduction']['fn_reduction_pct']:.1f}% reduction)\n"
                )
                f.write(
                    f"- **Estimated Analyst Alert Fatigue Reduction**: "
                    f"{metrics['alert_fatigue_reduction']['estimated_analyst_fatigue_reduction_pct']:.1f}%\n"
                )
                f.write(
                    f"  (Based on {metrics['alert_fatigue_reduction']['baseline_false_negatives']} baseline FNs "
                    f"vs {metrics['alert_fatigue_reduction']['aicra_false_negatives']} AICRA FNs)\n\n"
                )

        f.write("## Conclusion\n\n")
        if metrics["auroc"] >= 0.95:
            f.write("✓ H1 is **supported**: AUROC >= 0.95 achieved.\n")
            if "improvement" in metrics and "alert_fatigue_reduction" in metrics:
                f.write("\n**Key Findings:**\n")
                f.write(
                    f"- AICRA improves AUC by **+{metrics['improvement']['auroc_pct']:.1f}%** over baseline models.\n"
                )
                f.write(
                    f"- AICRA reduces false-negatives by **{metrics['alert_fatigue_reduction']['fn_reduction_pct']:.1f}%**, "
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
