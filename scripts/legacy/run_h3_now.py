#!/usr/bin/env python3
"""Run H3 evaluation and create H3_full_evaluation folder."""

import sys
import traceback
from pathlib import Path

# Write status to file
status_file = Path("h3_status.txt")

def write_status(msg):
    with open(status_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg)

try:
    write_status("=" * 80)
    write_status("Starting H3 evaluation...")
    write_status("=" * 80)
    
    repo_root = Path(__file__).parent.resolve()
    write_status(f"Repo root: {repo_root}")
    
    # Import
    write_status("Importing modules...")
    from aicra.experiments.h3_evaluation import run_h3_evaluation
    import yaml
    write_status("✓ Imports OK")
    
    # Paths
    config_path = repo_root / "config" / "h3_splits.yaml"
    det_path = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    lrn_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_path = repo_root / "d3fend_reference_pairs.csv"
    out_dir = repo_root / "results" / "H3_full_evaluation"
    
    write_status(f"Config: {config_path.exists()}")
    write_status(f"Det: {det_path.exists()}")
    write_status(f"Learned: {lrn_path.exists()}")
    write_status(f"Ref: {ref_path.exists()}")
    write_status(f"Output: {out_dir}")
    
    if not all([config_path.exists(), det_path.exists(), lrn_path.exists(), ref_path.exists()]):
        write_status("✗ Missing files!")
        sys.exit(1)
    
    write_status("\nRunning evaluation...")
    results = run_h3_evaluation(
        splits_config_path=config_path,
        det_mapping_path=det_path,
        learned_mapping_path=lrn_path,
        ref_pairs_path=ref_path,
        output_dir=out_dir,
        repo_root=repo_root,
    )
    
    write_status("\n✓ Evaluation complete!")
    
    # Check outputs
    json_file = out_dir / "H3_full_results.json"
    md_file = out_dir / "H3_full_summary.md"
    plots_dir = out_dir / "plots"
    
    write_status(f"\nOutputs:")
    write_status(f"  JSON: {json_file.exists()} ({json_file})")
    write_status(f"  MD: {md_file.exists()} ({md_file})")
    write_status(f"  Plots: {plots_dir.exists()} ({plots_dir})")
    
    if plots_dir.exists():
        plots = list(plots_dir.glob("*.png"))
        write_status(f"  Plot files: {len(plots)}")
        for p in plots:
            write_status(f"    - {p.name}")
    
    write_status("\n" + "=" * 80)
    write_status("SUCCESS!")
    write_status("=" * 80)
    
except Exception as e:
    write_status(f"\n✗ ERROR: {e}")
    write_status(traceback.format_exc())
    sys.exit(1)
