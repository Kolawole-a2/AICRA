"""
Benchmark Reporter: Generates consolidated benchmark improvement reports.

This module generates:
- artifacts/benchmark_improvements.csv - Machine-readable table
- artifacts/benchmark_improvements.md - Human-readable summary

All experiments (H1, H2, H3) should call this module to record their improvements.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_h1_results(results_dir: Path) -> dict | None:
    """Load H1 experiment results."""
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        logger.warning(f"H1 metrics not found: {metrics_path}")
        return None

    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def load_h2_results(results_dir: Path) -> dict | None:
    """Load H2 experiment results."""
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        logger.warning(f"H2 metrics not found: {metrics_path}")
        return None

    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


def load_h3_results(results_dir: Path) -> dict | None:
    """Load H3 experiment results."""
    results_path = results_dir / "H3_full_results.json"
    if not results_path.exists():
        logger.warning(f"H3 results not found: {results_path}")
        return None

    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


def generate_benchmark_improvements_table(
    h1_results_dir: Path | None = None,
    h2_results_dir: Path | None = None,
    h3_results_dir: Path | None = None,
    output_dir: Path = Path("artifacts"),
) -> None:
    """
    Generate consolidated benchmark improvements table.

    Args:
        h1_results_dir: Path to H1 results directory (default: artifacts/H1_classification)
        h2_results_dir: Path to H2 results directory (default: artifacts/H2_calibration_thresholds)
        h3_results_dir: Path to H3 results directory (default: artifacts/H3_full_evaluation)
        output_dir: Directory to save output files (default: artifacts)
    """
    # Default paths
    if h1_results_dir is None:
        h1_results_dir = output_dir / "H1_classification"
    if h2_results_dir is None:
        h2_results_dir = output_dir / "H2_calibration_thresholds"
    if h3_results_dir is None:
        h3_results_dir = output_dir / "H3_full_evaluation"

    # Load results
    h1_results = load_h1_results(h1_results_dir) if h1_results_dir.exists() else None
    h2_results = load_h2_results(h2_results_dir) if h2_results_dir.exists() else None
    h3_results = load_h3_results(h3_results_dir) if h3_results_dir.exists() else None

    # Build table rows
    rows = []

    # H1 improvements
    if h1_results and "improvement" in h1_results:
        imp = h1_results["improvement"]
        rows.append(
            {
                "hypothesis": "H1",
                "metric": "AUROC",
                "baseline_value": h1_results.get("baseline", {})
                .get("best_baseline", {})
                .get("auroc", "N/A"),
                "aicra_value": h1_results.get("auroc", "N/A"),
                "improvement_pct": imp.get("auroc_pct", "N/A"),
                "description": "Static PE classification AUROC improvement over baseline",
            }
        )
        rows.append(
            {
                "hypothesis": "H1",
                "metric": "Precision",
                "baseline_value": h1_results.get("baseline", {})
                .get("best_baseline", {})
                .get("precision", "N/A"),
                "aicra_value": h1_results.get("precision", "N/A"),
                "improvement_pct": imp.get("precision_pct", "N/A"),
                "description": "Precision improvement over baseline",
            }
        )
        rows.append(
            {
                "hypothesis": "H1",
                "metric": "Recall",
                "baseline_value": h1_results.get("baseline", {})
                .get("best_baseline", {})
                .get("recall", "N/A"),
                "aicra_value": h1_results.get("recall", "N/A"),
                "improvement_pct": imp.get("recall_pct", "N/A"),
                "description": "Recall improvement over baseline",
            }
        )
        rows.append(
            {
                "hypothesis": "H1",
                "metric": "F1",
                "baseline_value": h1_results.get("baseline", {})
                .get("best_baseline", {})
                .get("f1", "N/A"),
                "aicra_value": h1_results.get("f1", "N/A"),
                "improvement_pct": imp.get("f1_pct", "N/A"),
                "description": "F1 score improvement over baseline",
            }
        )
        if "alert_fatigue_reduction" in h1_results:
            afr = h1_results["alert_fatigue_reduction"]
            rows.append(
                {
                    "hypothesis": "H1",
                    "metric": "Alert Fatigue Reduction",
                    "baseline_value": afr.get("baseline_false_negatives", "N/A"),
                    "aicra_value": afr.get("aicra_false_negatives", "N/A"),
                    "improvement_pct": afr.get(
                        "estimated_analyst_fatigue_reduction_pct", "N/A"
                    ),
                    "description": "Estimated analyst fatigue reduction (%)",
                }
            )

    # H2 improvements
    if h2_results and "improvement" in h2_results:
        imp = h2_results["improvement"]
        rows.append(
            {
                "hypothesis": "H2",
                "metric": "Brier Score",
                "baseline_value": imp.get("baseline_brier", "N/A"),
                "aicra_value": h2_results.get("calibration", {}).get(
                    "brier_calibrated", "N/A"
                ),
                "improvement_pct": imp.get("brier_improvement_pct", "N/A"),
                "description": "Brier score improvement (calibrated vs uncalibrated)",
            }
        )
        rows.append(
            {
                "hypothesis": "H2",
                "metric": "ECE",
                "baseline_value": imp.get("baseline_ece", "N/A"),
                "aicra_value": h2_results.get("calibration", {}).get(
                    "ece_calibrated", "N/A"
                ),
                "improvement_pct": imp.get("ece_improvement_pct", "N/A"),
                "description": "Expected Calibration Error improvement",
            }
        )

    # H3 improvements
    if h3_results and "aggregated_metrics" in h3_results:
        agg = h3_results["aggregated_metrics"]
        if "improvements" in agg:
            imp = agg["improvements"]
            rows.append(
                {
                    "hypothesis": "H3",
                    "metric": "Coverage",
                    "baseline_value": f"{agg.get('learned', {}).get('coverage_%', {}).get('mean', 'N/A'):.2f}%",
                    "aicra_value": f"{agg.get('deterministic', {}).get('coverage_%', {}).get('mean', 'N/A'):.2f}%",
                    "improvement_pct": imp.get("coverage_improvement_pct", "N/A"),
                    "description": "Mapping coverage improvement (deterministic vs learned)",
                }
            )
            rows.append(
                {
                    "hypothesis": "H3",
                    "metric": "Variance Reduction",
                    "baseline_value": f"{agg.get('learned', {}).get('score_consistency', {}).get('variance', {}).get('mean', 'N/A'):.6f}",
                    "aicra_value": f"{agg.get('deterministic', {}).get('score_consistency', {}).get('variance', {}).get('mean', 'N/A'):.6f}",
                    "improvement_pct": imp.get("variance_reduction_pct", "N/A"),
                    "description": "Risk score variance reduction (deterministic vs learned)",
                }
            )
            rows.append(
                {
                    "hypothesis": "H3",
                    "metric": "Alert Fatigue Reduction",
                    "baseline_value": "N/A",
                    "aicra_value": "N/A",
                    "improvement_pct": imp.get(
                        "estimated_fatigue_reduction_pct", "N/A"
                    ),
                    "description": "Estimated alert fatigue reduction (%)",
                }
            )

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Save CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_improvements.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved benchmark improvements CSV to: {csv_path}")

    # Generate markdown summary
    md_path = output_dir / "benchmark_improvements.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# AICRA Benchmark Improvements Summary\n\n")
        f.write(
            "This document summarizes the % improvements achieved by AICRA over baseline methods.\n\n"
        )
        f.write("## H1: Static PE Classification Reliability\n\n")
        f.write(
            "**Hypothesis**: LightGBM on EMBER-2024 static PE features predicts ransomware susceptibility.\n\n"
        )

        h1_rows = df[df["hypothesis"] == "H1"]
        if len(h1_rows) > 0:
            f.write("| Metric | Baseline | AICRA | Improvement |\n")
            f.write("|--------|----------|-------|-------------|\n")
            for _, row in h1_rows.iterrows():
                baseline = row["baseline_value"]
                aicra = row["aicra_value"]
                improvement = row["improvement_pct"]
                if isinstance(improvement, (int, float)):
                    improvement_str = f"{improvement:.2f}%"
                else:
                    improvement_str = str(improvement)
                f.write(
                    f"| {row['metric']} | {baseline} | {aicra} | {improvement_str} |\n"
                )
            f.write("\n")

        f.write("## H2: Calibration and Transferability\n\n")
        f.write(
            "**Hypothesis**: Isotonic calibration improves susceptibility score transferability.\n\n"
        )

        h2_rows = df[df["hypothesis"] == "H2"]
        if len(h2_rows) > 0:
            f.write("| Metric | Baseline | AICRA | Improvement |\n")
            f.write("|--------|----------|-------|-------------|\n")
            for _, row in h2_rows.iterrows():
                baseline = row["baseline_value"]
                aicra = row["aicra_value"]
                improvement = row["improvement_pct"]
                if isinstance(improvement, (int, float)):
                    improvement_str = f"{improvement:.2f}%"
                else:
                    improvement_str = str(improvement)
                f.write(
                    f"| {row['metric']} | {baseline} | {aicra} | {improvement_str} |\n"
                )
            f.write("\n")

        f.write("## H3: Deterministic vs Learned Mapping\n\n")
        f.write(
            "**Hypothesis**: Deterministic ATT&CK–D3FEND lookup beats learned mapping.\n\n"
        )

        h3_rows = df[df["hypothesis"] == "H3"]
        if len(h3_rows) > 0:
            f.write(
                "| Metric | Baseline (Learned) | AICRA (Deterministic) | Improvement |\n"
            )
            f.write(
                "|--------|-------------------|----------------------|-------------|\n"
            )
            for _, row in h3_rows.iterrows():
                baseline = row["baseline_value"]
                aicra = row["aicra_value"]
                improvement = row["improvement_pct"]
                if isinstance(improvement, (int, float)):
                    improvement_str = f"{improvement:.2f}%"
                else:
                    improvement_str = str(improvement)
                f.write(
                    f"| {row['metric']} | {baseline} | {aicra} | {improvement_str} |\n"
                )
            f.write("\n")

        f.write("## Summary Statements\n\n")

        if h1_results and "improvement_statement" in h1_results:
            f.write(f"**H1**: {h1_results['improvement_statement']}\n\n")

        if h2_results and "improvement_statement" in h2_results:
            f.write(f"**H2**: {h2_results['improvement_statement']}\n\n")

        if h3_results and "aggregated_metrics" in h3_results:
            agg = h3_results["aggregated_metrics"]
            if "improvements" in agg:
                imp = agg["improvements"]
                f.write(
                    f"**H3**: Deterministic mapping increases ATT&CK–D3FEND mapping coverage by "
                    f"+{imp.get('coverage_improvement_pct', 0):.1f}% and reduces risk-score variance by "
                    f"{imp.get('variance_reduction_pct', 0):.1f}%.\n\n"
                )

    logger.info(f"Saved benchmark improvements markdown to: {md_path}")


if __name__ == "__main__":
    # Allow running as script
    import sys
    from pathlib import Path

    output_dir = Path("artifacts")
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    generate_benchmark_improvements_table(output_dir=output_dir)
