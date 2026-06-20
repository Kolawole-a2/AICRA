#!/usr/bin/env python3
"""Simple script to run H3 evaluation."""

import sys
from pathlib import Path

print("Starting H3 evaluation...", flush=True)

try:
    repo_root = Path(__file__).parent.resolve()
    print(f"Repo root: {repo_root}", flush=True)
    
    from aicra.experiments.h3_evaluation import run_h3_evaluation
    import yaml
    import tempfile
    
    # Create temp config
    splits_config = {"splits": {"main": "risk_scores.csv"}}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(splits_config, f)
        temp_config = Path(f.name)
    
    print("Running evaluation...", flush=True)
    
    results = run_h3_evaluation(
        splits_config_path=temp_config,
        det_mapping_path=repo_root / "data/mappings/deterministic_lookup.csv",
        learned_mapping_path=repo_root / "data/mappings/learned_mapping.csv",
        ref_pairs_path=repo_root / "d3fend_reference_pairs.csv",
        output_dir=repo_root / "results/H3_full_evaluation",
        repo_root=repo_root,
    )
    
    print("Evaluation completed!", flush=True)
    
    # Check outputs
    out_dir = repo_root / "results/H3_full_evaluation"
    json_file = out_dir / "H3_full_results.json"
    md_file = out_dir / "H3_full_summary.md"
    plots_dir = out_dir / "plots"
    
    print(f"\nOutputs:", flush=True)
    print(f"  JSON: {json_file.exists()}", flush=True)
    print(f"  MD: {md_file.exists()}", flush=True)
    print(f"  Plots: {plots_dir.exists()}", flush=True)
    
    if plots_dir.exists():
        plots = list(plots_dir.glob("*.png"))
        print(f"  Plot count: {len(plots)}", flush=True)
    
    temp_config.unlink()
    
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
