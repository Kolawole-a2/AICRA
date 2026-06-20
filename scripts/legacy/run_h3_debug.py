#!/usr/bin/env python3
"""Run H3 evaluation with explicit error handling and output."""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from aicra.experiments.h3_evaluation import run_h3_evaluation
    import yaml
    import tempfile
    
    repo_root = Path('.')
    
    # Create config
    splits_config = {
        "splits": {
            "main": "risk_scores.csv"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(splits_config, f)
        temp_config = Path(f.name)
    
    print("="*80)
    print("Running H3 Evaluation")
    print("="*80)
    print(f"Config: {temp_config}")
    print(f"Deterministic: data/mappings/deterministic_lookup.csv")
    print(f"Learned: data/mappings/learned_mapping.csv")
    print(f"Reference: d3fend_reference_pairs.csv")
    print(f"Output: results/H3_full_evaluation")
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
    
    if "aggregated_metrics" in results:
        agg = results["aggregated_metrics"]
        det_prec = agg.get("deterministic", {}).get("actionable_precision", {}).get("mean", 0)
        lrn_prec = agg.get("learned", {}).get("actionable_precision", {}).get("mean", 0)
        delta_prec = det_prec - lrn_prec
        
        det_var = agg.get("deterministic", {}).get("variance_reduction", {}).get("mean", 0)
        lrn_var = agg.get("learned", {}).get("variance_reduction", {}).get("mean", 0)
        delta_var = det_var - lrn_var
        
        print(f"\nResults:")
        print(f"  Precision - Det: {det_prec:.4f}, Learned: {lrn_prec:.4f}, Delta: {delta_prec:.4f}")
        print(f"  Variance - Det: {det_var:.6f}, Learned: {lrn_var:.6f}, Delta: {delta_var:.6f}")
        
        if abs(delta_prec) < 0.0001 and abs(delta_var) < 0.000001:
            print("\n⚠️  WARNING: Metrics are still identical!")
        else:
            print("\n✓ Metrics are different - plots should show distinct values!")
    
    import os
    os.unlink(temp_config)
    
    print(f"\nResults saved to: results/H3_full_evaluation/")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
