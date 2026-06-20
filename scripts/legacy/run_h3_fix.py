#!/usr/bin/env python3
"""Direct script to fix and run H3 evaluation."""

import sys
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

from aicra.experiments.h3_evaluation import run_h3_evaluation
import yaml

# Create a simple config using risk_scores.csv in root
splits_config = {
    "splits": {
        "main": "risk_scores.csv"
    }
}

# Save temp config
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
    yaml.dump(splits_config, f)
    temp_config = Path(f.name)

try:
    print("="*80)
    print("Running H3 Evaluation with Fixed Mappings")
    print("="*80)
    
    results = run_h3_evaluation(
        splits_config_path=temp_config,
        det_mapping_path=repo_root / "data/mappings/deterministic_lookup.csv",
        learned_mapping_path=repo_root / "data/mappings/learned_mapping.csv",
        ref_pairs_path=repo_root / "d3fend_reference_pairs.csv",
        output_dir=repo_root / "results/H3_full_evaluation",
        repo_root=repo_root,
    )
    
    print("\n" + "="*80)
    print("H3 Evaluation Complete!")
    print("="*80)
    print(f"\nResults saved to: results/H3_full_evaluation/")
    print(f"Plots: results/H3_full_evaluation/plots/")
    
    # Check if metrics are different
    if "aggregated_metrics" in results:
        agg = results["aggregated_metrics"]
        det_prec = agg.get("deterministic", {}).get("actionable_precision", {}).get("mean", 0)
        lrn_prec = agg.get("learned", {}).get("actionable_precision", {}).get("mean", 0)
        delta = det_prec - lrn_prec
        
        print(f"\nPrecision - Deterministic: {det_prec:.4f}, Learned: {lrn_prec:.4f}, Delta: {delta:.4f}")
        
        if abs(delta) < 0.0001:
            print("\n⚠️  WARNING: Metrics are still identical!")
            print("   The learned mapping may be identical to deterministic for techniques in risk scores.")
        else:
            print("\n✓ Metrics are different - plots should show distinct values!")
    
finally:
    # Clean up temp config
    temp_config.unlink()
