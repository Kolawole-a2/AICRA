"""
Statistical Analysis for H3 DAC Validation

Performs statistical tests to validate that:
1. Deterministic mapping significantly outperforms learned mapping
2. Higher DAC values correlate with better operational performance
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon

logger = logging.getLogger(__name__)


def perform_statistical_tests(results_df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Perform statistical tests comparing deterministic vs learned mappings.

    Tests performed:
    1. Paired t-test for precision differences
    2. Wilcoxon signed-rank test for precision (non-parametric)
    3. Paired t-test for variance reduction differences
    4. Spearman correlation: DAC vs precision
    5. Spearman correlation: DAC vs variance reduction

    Args:
        results_df: DataFrame with metrics for all splits.
        output_dir: Directory to save results and plots.

    Returns:
        Dictionary with all statistical test results.
    """
    logger.info("Performing statistical tests...")

    stats_results = {}

    # Test 1: Paired t-test for precision (deterministic vs learned)
    if len(results_df) > 1:
        precision_det = results_df["precision_deterministic"].values
        precision_learned = results_df["precision_learned"].values

        t_stat_precision, p_value_precision = ttest_rel(
            precision_det, precision_learned
        )
        stats_results["precision_ttest"] = {
            "statistic": float(t_stat_precision),
            "pvalue": float(p_value_precision),
            "significant": p_value_precision < 0.05,
            "interpretation": (
                "Deterministic mapping significantly outperforms learned mapping in precision"
                if p_value_precision < 0.05
                else "No significant difference in precision (may need more data)"
            ),
        }
        logger.info(
            f"Precision t-test: t={t_stat_precision:.4f}, p={p_value_precision:.4f}, "
            f"significant={p_value_precision < 0.05}"
        )

        # Test 2: Wilcoxon signed-rank test (non-parametric)
        try:
            w_stat, w_pvalue = wilcoxon(
                precision_det, precision_learned, alternative="greater"
            )
            stats_results["precision_wilcoxon"] = {
                "statistic": float(w_stat),
                "pvalue": float(w_pvalue),
                "significant": w_pvalue < 0.05,
                "interpretation": (
                    "Deterministic mapping significantly outperforms learned mapping (non-parametric)"
                    if w_pvalue < 0.05
                    else "No significant difference (non-parametric test)"
                ),
            }
            logger.info(
                f"Precision Wilcoxon: W={w_stat:.4f}, p={w_pvalue:.4f}, "
                f"significant={w_pvalue < 0.05}"
            )
        except Exception as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            stats_results["precision_wilcoxon"] = {"error": str(e)}
    else:
        logger.warning("Insufficient data for paired tests (need >1 split)")
        stats_results["precision_ttest"] = {"error": "Insufficient data"}
        stats_results["precision_wilcoxon"] = {"error": "Insufficient data"}

    # Test 3: Paired t-test for variance reduction
    if len(results_df) > 1:
        var_red_det = results_df["variance_reduction_deterministic"].values
        var_red_learned = results_df["variance_reduction_learned"].values

        t_stat_var, p_value_var = ttest_rel(var_red_det, var_red_learned)
        stats_results["variance_reduction_ttest"] = {
            "statistic": float(t_stat_var),
            "pvalue": float(p_value_var),
            "significant": p_value_var < 0.05,
            "interpretation": (
                "Deterministic mapping provides significantly better variance reduction"
                if p_value_var < 0.05
                else "No significant difference in variance reduction"
            ),
        }
        logger.info(
            f"Variance reduction t-test: t={t_stat_var:.4f}, p={p_value_var:.4f}, "
            f"significant={p_value_var < 0.05}"
        )
    else:
        stats_results["variance_reduction_ttest"] = {"error": "Insufficient data"}

    # Test 4: Spearman correlation - DAC vs Precision
    # Combine deterministic and learned data
    dac_values = []
    precision_values = []

    for _, row in results_df.iterrows():
        dac_values.append(row["dac_deterministic"])
        precision_values.append(row["precision_deterministic"])
        dac_values.append(row["dac_learned"])
        precision_values.append(row["precision_learned"])

    if len(dac_values) > 2:
        corr_dac_precision, p_corr_precision = spearmanr(dac_values, precision_values)
        stats_results["dac_vs_precision_correlation"] = {
            "correlation": float(corr_dac_precision),
            "pvalue": float(p_corr_precision),
            "significant": p_corr_precision < 0.05,
            "interpretation": (
                "Strong positive correlation: Higher DAC predicts better precision"
                if corr_dac_precision > 0.5 and p_corr_precision < 0.05
                else "Weak or non-significant correlation"
            ),
        }
        logger.info(
            f"DAC vs Precision correlation: rho={corr_dac_precision:.4f}, "
            f"p={p_corr_precision:.4f}"
        )
    else:
        stats_results["dac_vs_precision_correlation"] = {"error": "Insufficient data"}

    # Test 5: Spearman correlation - DAC vs Variance Reduction
    dac_values_var = []
    var_red_values = []

    for _, row in results_df.iterrows():
        dac_values_var.append(row["dac_deterministic"])
        var_red_values.append(row["variance_reduction_deterministic"])
        dac_values_var.append(row["dac_learned"])
        var_red_values.append(row["variance_reduction_learned"])

    if len(dac_values_var) > 2:
        corr_dac_var, p_corr_var = spearmanr(dac_values_var, var_red_values)
        stats_results["dac_vs_variance_reduction_correlation"] = {
            "correlation": float(corr_dac_var),
            "pvalue": float(p_corr_var),
            "significant": p_corr_var < 0.05,
            "interpretation": (
                "Strong positive correlation: Higher DAC predicts better variance reduction"
                if corr_dac_var > 0.5 and p_corr_var < 0.05
                else "Weak or non-significant correlation"
            ),
        }
        logger.info(
            f"DAC vs Variance Reduction correlation: rho={corr_dac_var:.4f}, "
            f"p={p_corr_var:.4f}"
        )
    else:
        stats_results["dac_vs_variance_reduction_correlation"] = {
            "error": "Insufficient data"
        }

    # Save summary
    summary_rows = []
    for test_name, test_result in stats_results.items():
        if "error" not in test_result:
            summary_rows.append(
                {
                    "test": test_name,
                    "statistic": test_result.get("statistic")
                    or test_result.get("correlation"),
                    "pvalue": test_result.get("pvalue"),
                    "significant": test_result.get("significant"),
                    "interpretation": test_result.get("interpretation", ""),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "h3_dac_stats_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved statistical summary to {summary_path}")

    return stats_results


def create_correlation_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create correlation plots for DAC vs operational performance metrics.

    Args:
        results_df: DataFrame with metrics for all splits.
        output_dir: Directory to save plots.
    """
    logger.info("Creating correlation plots...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data: combine deterministic and learned
    dac_det = results_df["dac_deterministic"].values
    precision_det = results_df["precision_deterministic"].values
    var_red_det = results_df["variance_reduction_deterministic"].values

    dac_learned = results_df["dac_learned"].values
    precision_learned = results_df["precision_learned"].values
    var_red_learned = results_df["variance_reduction_learned"].values

    # Plot 1: DAC vs Precision
    plt.figure(figsize=(10, 6))
    plt.scatter(
        dac_det, precision_det, label="Deterministic", color="#2e7d32", s=100, alpha=0.7
    )
    plt.scatter(
        dac_learned,
        precision_learned,
        label="Learned",
        color="#1976d2",
        s=100,
        alpha=0.7,
    )
    plt.xlabel("DAC (Defense-Attack Consistency)", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(
        "DAC vs Precision: Higher DAC Predicts Better Precision",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)

    # Add correlation line
    all_dac = np.concatenate([dac_det, dac_learned])
    all_precision = np.concatenate([precision_det, precision_learned])
    if len(all_dac) > 1:
        z = np.polyfit(all_dac, all_precision, 1)
        p = np.poly1d(z)
        plt.plot(all_dac, p(all_dac), "r--", alpha=0.5, label="Trend line")

    plt.tight_layout()
    plot_path = output_dir / "h3_dac_vs_precision.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {plot_path}")

    # Plot 2: DAC vs Variance Reduction
    plt.figure(figsize=(10, 6))
    plt.scatter(
        dac_det, var_red_det, label="Deterministic", color="#2e7d32", s=100, alpha=0.7
    )
    plt.scatter(
        dac_learned, var_red_learned, label="Learned", color="#1976d2", s=100, alpha=0.7
    )
    plt.xlabel("DAC (Defense-Attack Consistency)", fontsize=12)
    plt.ylabel("Variance Reduction", fontsize=12)
    plt.title(
        "DAC vs Variance Reduction: Higher DAC Predicts Better Stability",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)

    # Add correlation line
    all_var_red = np.concatenate([var_red_det, var_red_learned])
    if len(all_dac) > 1 and len(all_var_red) > 1:
        z = np.polyfit(all_dac, all_var_red, 1)
        p = np.poly1d(z)
        plt.plot(all_dac, p(all_dac), "r--", alpha=0.5, label="Trend line")

    plt.tight_layout()
    plot_path = output_dir / "h3_dac_vs_variance_reduction.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot to {plot_path}")


def analyze_h3_results(results_path: Path, output_dir: Path) -> dict:
    """
    Main analysis function: load results and perform all statistical tests.

    Args:
        results_path: Path to h3_dac_metrics_by_split.csv.
        output_dir: Directory to save analysis results.

    Returns:
        Dictionary with all statistical test results.
    """
    logger.info("Loading H3 results for statistical analysis...")

    results_df = pd.read_csv(results_path)
    logger.info(f"Loaded results for {len(results_df)} splits")

    # Perform statistical tests
    stats_results = perform_statistical_tests(results_df, output_dir)

    # Create plots
    create_correlation_plots(results_df, output_dir)

    return stats_results

