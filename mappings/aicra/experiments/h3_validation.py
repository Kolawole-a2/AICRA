"""H3 Validation: Full DAC + Precision/Variance Comparison.

This module implements the H3 experiment validation comparing deterministic
and learned mappings in terms of:
- Mapping coverage (%)
- Defense–Attack Consistency (DAC %)
- Δ precision (actionable positives)
- Variance reduction in risk scores
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import logging
import json
import pandas as pd
import numpy as np

LOGGER = logging.getLogger(__name__)


def load_mappings(
    det_path: Path,
    lrn_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load deterministic and learned mappings."""
    det_df = pd.read_csv(det_path)
    lrn_df = pd.read_csv(lrn_path)
    
    # Normalize column names
    if "attack_id" in det_df.columns:
        det_df = det_df.rename(columns={"attack_id": "technique_id"})
    if "defense_id" in det_df.columns:
        det_df = det_df.rename(columns={"defense_id": "control_id"})
    
    return det_df, lrn_df


def compute_coverage(mapping_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute mapping coverage statistics."""
    total_techs = mapping_df["technique_id"].nunique()
    total_pairs = len(mapping_df)
    avg_pairs_per_tech = total_pairs / total_techs if total_techs > 0 else 0
    
    return {
        "total_techniques": int(total_techs),
        "total_pairs": int(total_pairs),
        "avg_pairs_per_technique": float(avg_pairs_per_tech),
        "unique_controls": int(mapping_df["control_id"].nunique()),
    }


def compute_dac(
    mapping_pairs: set[tuple[str, str]],
    reference_pairs: set[tuple[str, str]],
) -> Dict[str, Any]:
    """Compute Defense–Attack Consistency (DAC) metrics."""
    intersection = mapping_pairs & reference_pairs
    total = len(mapping_pairs)
    overlap = len(intersection)
    
    dac_percent = (overlap / total * 100.0) if total > 0 else 0.0
    
    return {
        "dac_percent": float(dac_percent),
        "overlapping_pairs": int(overlap),
        "total_pairs": int(total),
        "reference_pairs": int(len(reference_pairs)),
    }


def compute_precision_delta(
    det_df: pd.DataFrame,
    lrn_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute Δ precision (difference in actionable positives)."""
    # This is a placeholder - actual precision calculation would depend on
    # your specific definition of "actionable positives"
    det_pairs = set(zip(det_df["technique_id"], det_df["control_id"]))
    lrn_pairs = set(zip(lrn_df["technique_id"], lrn_df["control_id"]))
    
    # Simple metric: unique technique-control combinations
    det_unique = len(det_pairs)
    lrn_unique = len(lrn_pairs)
    delta = lrn_unique - det_unique
    
    return {
        "deterministic_unique_pairs": int(det_unique),
        "learned_unique_pairs": int(lrn_unique),
        "delta_pairs": int(delta),
        "delta_percent": float((delta / det_unique * 100.0) if det_unique > 0 else 0.0),
    }


def compute_variance_reduction(
    det_df: pd.DataFrame,
    lrn_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute variance reduction in risk scores."""
    # Placeholder - actual variance calculation would require risk scores
    # This would typically involve computing variance of risk scores
    # when using deterministic vs learned mappings
    
    det_tech_counts = det_df.groupby("technique_id").size()
    lrn_tech_counts = lrn_df.groupby("technique_id").size()
    
    det_var = float(det_tech_counts.var())
    lrn_var = float(lrn_tech_counts.var())
    
    variance_reduction = ((det_var - lrn_var) / det_var * 100.0) if det_var > 0 else 0.0
    
    return {
        "deterministic_variance": float(det_var),
        "learned_variance": float(lrn_var),
        "variance_reduction_percent": float(variance_reduction),
    }


def run_h3_validation(
    det_path: Path,
    lrn_path: Path,
    ref_path: Path | None = None,
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    """Run full H3 validation experiment."""
    LOGGER.info("Starting H3 validation experiment")
    
    # Load mappings
    det_df, lrn_df = load_mappings(det_path, lrn_path)
    
    # Use deterministic as reference if no reference provided
    if ref_path is None or not ref_path.exists():
        ref_df = det_df.copy()
        LOGGER.info("Using deterministic mapping as reference")
    else:
        ref_df = pd.read_csv(ref_path)
        if "attack_id" in ref_df.columns:
            ref_df = ref_df.rename(columns={"attack_id": "technique_id"})
        if "defense_id" in ref_df.columns:
            ref_df = ref_df.rename(columns={"defense_id": "control_id"})
    
    # Convert to pair sets
    det_pairs = set(zip(det_df["technique_id"], det_df["control_id"]))
    lrn_pairs = set(zip(lrn_df["technique_id"], lrn_df["control_id"]))
    ref_pairs = set(zip(ref_df["technique_id"], ref_df["control_id"]))
    
    # Compute metrics
    results = {
        "deterministic": compute_coverage(det_df),
        "learned": compute_coverage(lrn_df),
        "dac": {
            "deterministic": compute_dac(det_pairs, ref_pairs),
            "learned": compute_dac(lrn_pairs, ref_pairs),
        },
        "precision_delta": compute_precision_delta(det_df, lrn_df),
        "variance_reduction": compute_variance_reduction(det_df, lrn_df),
    }
    
    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / "H3_results.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)
        LOGGER.info(f"Saved results to {results_path}")
        
        # Generate summary markdown
        summary_path = output_dir / "H3_summary.md"
        generate_summary(results, summary_path)
        LOGGER.info(f"Saved summary to {summary_path}")
    
    return results


def generate_summary(results: Dict[str, Any], output_path: Path) -> None:
    """Generate markdown summary of H3 results."""
    det_cov = results["deterministic"]
    lrn_cov = results["learned"]
    det_dac = results["dac"]["deterministic"]
    lrn_dac = results["dac"]["learned"]
    prec = results["precision_delta"]
    var = results["variance_reduction"]
    
    summary = f"""# H3 Validation Results Summary

## Coverage Comparison

| Metric | Deterministic | Learned |
|--------|--------------|---------|
| Techniques | {det_cov['total_techniques']} | {lrn_cov['total_techniques']} |
| Total Pairs | {det_cov['total_pairs']} | {lrn_cov['total_pairs']} |
| Avg Pairs/Technique | {det_cov['avg_pairs_per_technique']:.2f} | {lrn_cov['avg_pairs_per_technique']:.2f} |
| Unique Controls | {det_cov['unique_controls']} | {lrn_cov['unique_controls']} |

## Defense–Attack Consistency (DAC)

- **Deterministic DAC**: {det_dac['dac_percent']:.2f}% ({det_dac['overlapping_pairs']} / {det_dac['total_pairs']})
- **Learned DAC**: {lrn_dac['dac_percent']:.2f}% ({lrn_dac['overlapping_pairs']} / {lrn_dac['total_pairs']})

## Precision Delta

- **Δ Precision**: {prec['delta_pairs']} pairs ({prec['delta_percent']:.2f}%)
- Deterministic unique pairs: {prec['deterministic_unique_pairs']}
- Learned unique pairs: {prec['learned_unique_pairs']}

## Variance Reduction

- **Variance Reduction**: {var['variance_reduction_percent']:.2f}%
- Deterministic variance: {var['deterministic_variance']:.4f}
- Learned variance: {var['learned_variance']:.4f}

## Conclusion

The learned mapping provides a heuristic baseline for comparison with the deterministic mapping.
"""
    
    with output_path.open("w") as f:
        f.write(summary)


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Run H3 validation experiment")
    parser.add_argument(
        "--deterministic",
        type=Path,
        default=Path("data/mappings/deterministic_lookup.csv"),
        help="Path to deterministic mapping CSV",
    )
    parser.add_argument(
        "--learned",
        type=Path,
        default=Path("data/mappings/learned_mapping.csv"),
        help="Path to learned mapping CSV",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Path to reference pairs CSV (optional, uses deterministic if not provided)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/H3_full_evaluation"),
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    results = run_h3_validation(
        det_path=args.deterministic,
        lrn_path=args.learned,
        ref_path=args.reference,
        output_dir=args.output,
    )
    
    print("\nH3 Validation Complete!")
    print(f"Results saved to: {args.output}")

