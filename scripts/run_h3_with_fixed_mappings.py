#!/usr/bin/env python3
"""
Run H3 evaluation with fixed mappings and display results clearly.
"""

import sys
import tempfile
from pathlib import Path

import yaml

# Add repo to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from aicra.experiments.h3_evaluation import run_h3_evaluation  # noqa: E402

print("=" * 80)
print("RUNNING H3 EVALUATION WITH FIXED MAPPINGS")
print("=" * 80)

# Create config using risk_scores.csv
splits_config = {"splits": {"main": "risk_scores.csv"}}

# Save temp config
with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    yaml.dump(splits_config, f)
    temp_config = Path(f.name)

try:
    print("\nRunning H3 evaluation...")
    print("  Deterministic: data/mappings/deterministic_lookup.csv")
    print("  Learned: data/mappings/learned_mapping.csv")
    print("  Reference: d3fend_reference_pairs.csv")
    print("  Output: results/H3_full_evaluation")

    results = run_h3_evaluation(
        splits_config_path=temp_config,
        det_mapping_path=repo_root / "data/mappings/deterministic_lookup.csv",
        learned_mapping_path=repo_root / "data/mappings/learned_mapping.csv",
        ref_pairs_path=repo_root / "d3fend_reference_pairs.csv",
        output_dir=repo_root / "results/H3_full_evaluation",
        repo_root=repo_root,
    )

    print("\n" + "=" * 80)
    print("H3 EVALUATION COMPLETE!")
    print("=" * 80)

    # Check results
    if "aggregated_metrics" in results:
        agg = results["aggregated_metrics"]

        det_prec = (
            agg.get("deterministic", {}).get("actionable_precision", {}).get("mean", 0)
        )
        lrn_prec = agg.get("learned", {}).get("actionable_precision", {}).get("mean", 0)
        delta_prec = det_prec - lrn_prec

        det_var = (
            agg.get("deterministic", {}).get("variance_reduction", {}).get("mean", 0)
        )
        lrn_var = agg.get("learned", {}).get("variance_reduction", {}).get("mean", 0)
        delta_var = det_var - lrn_var

        det_dac = agg.get("deterministic", {}).get("dac_%", {}).get("mean", 0)
        lrn_dac = agg.get("learned", {}).get("dac_%", {}).get("mean", 0)
        delta_dac = det_dac - lrn_dac

        print("\nResults Summary:")
        print(
            f"  Precision - Det: {det_prec:.4f}, Learned: {lrn_prec:.4f}, Delta: {delta_prec:.4f}"
        )
        print(
            f"  Variance - Det: {det_var:.6f}, Learned: {lrn_var:.6f}, Delta: {delta_var:.6f}"
        )
        print(
            f"  DAC - Det: {det_dac:.2f}%, Learned: {lrn_dac:.2f}%, Delta: {delta_dac:.2f}%"
        )

        if abs(delta_prec) < 0.0001 and abs(delta_var) < 0.000001:
            print("\n⚠️  WARNING: Metrics are still identical!")
            print("   The learned mapping may need further adjustment.")
        else:
            print("\n✓ Metrics are different - plots should show distinct values!")

    # Check overlap metrics
    if "mapping_overlap" in results:
        overlap = results["mapping_overlap"]
        jaccard = overlap.get("global_jaccard", 0)
        exact_match = overlap.get("fraction_exact_match_techniques", 0)

        print("\nMapping Overlap:")
        print(f"  Jaccard similarity: {jaccard:.4f} ({jaccard * 100:.2f}%)")
        print(f"  EXACT_MATCH fraction: {exact_match:.4f} ({exact_match * 100:.2f}%)")

        if jaccard > 0.95:
            print("  ⚠️  WARNING: Very high overlap (>95%)")
        elif jaccard > 0.80:
            print("  ⚠️  WARNING: High overlap (>80%)")
        else:
            print("  ✓ Reasonable diversity")

    print("\nResults saved to:")
    print("  - results/H3_full_evaluation/H3_full_results.json")
    print("  - results/H3_full_evaluation/H3_full_summary.md")
    print("  - results/H3_full_evaluation/plots/")

    print("\n" + "=" * 80)

finally:
    # Clean up temp config
    temp_config.unlink()
