#!/usr/bin/env python3
"""
Run H3 evaluation and create H3_full_evaluation folder with verification.
"""

import json
import sys
import traceback
from pathlib import Path

# Create verification log
verification_log = []

def log(msg):
    """Log message to both console and verification log."""
    print(msg)
    verification_log.append(msg)

try:
    log("=" * 80)
    log("H3 Full Evaluation - Creation Script")
    log("=" * 80)
    
    repo_root = Path(__file__).parent
    log(f"Repository root: {repo_root}")
    
    # Import required modules
    log("\nImporting modules...")
    from aicra.experiments.h3_evaluation import run_h3_evaluation
    import yaml
    log("✓ Imports successful")
    
    # Set up paths
    splits_config_path = repo_root / "config" / "h3_splits.yaml"
    det_mapping_path = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    learned_mapping_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_pairs_path = repo_root / "d3fend_reference_pairs.csv"
    output_dir = repo_root / "results" / "H3_full_evaluation"
    
    log("\nChecking input files:")
    log(f"  Splits config: {splits_config_path} - {'✓' if splits_config_path.exists() else '✗'}")
    log(f"  Deterministic: {det_mapping_path} - {'✓' if det_mapping_path.exists() else '✗'}")
    log(f"  Learned: {learned_mapping_path} - {'✓' if learned_mapping_path.exists() else '✗'}")
    log(f"  Reference: {ref_pairs_path} - {'✓' if ref_pairs_path.exists() else '✗'}")
    
    if not all([splits_config_path.exists(), det_mapping_path.exists(), 
                learned_mapping_path.exists(), ref_pairs_path.exists()]):
        log("\n✗ ERROR: Missing required input files!")
        sys.exit(1)
    
    log(f"\nOutput directory: {output_dir}")
    
    # Run evaluation
    log("\n" + "=" * 80)
    log("Running H3 evaluation...")
    log("=" * 80)
    
    results = run_h3_evaluation(
        splits_config_path=splits_config_path,
        det_mapping_path=det_mapping_path,
        learned_mapping_path=learned_mapping_path,
        ref_pairs_path=ref_pairs_path,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    
    log("\n" + "=" * 80)
    log("Evaluation completed!")
    log("=" * 80)
    
    # Verify outputs
    log("\nVerifying outputs:")
    json_path = output_dir / "H3_full_results.json"
    md_path = output_dir / "H3_full_summary.md"
    plots_dir = output_dir / "plots"
    
    log(f"  H3_full_results.json: {'✓' if json_path.exists() else '✗'} ({json_path})")
    log(f"  H3_full_summary.md: {'✓' if md_path.exists() else '✗'} ({md_path})")
    log(f"  plots/ directory: {'✓' if plots_dir.exists() else '✗'} ({plots_dir})")
    
    if plots_dir.exists():
        plot_files = sorted(plots_dir.glob("*.png"))
        log(f"  Plot files found: {len(plot_files)}")
        expected_plots = [
            "dac_per_split.png",
            "precision_per_split.png", 
            "variance_reduction_per_split.png",
            "summary_metrics.png"
        ]
        for expected in expected_plots:
            plot_path = plots_dir / expected
            log(f"    {expected}: {'✓' if plot_path.exists() else '✗'}")
        for plot_file in plot_files:
            if plot_file.name not in expected_plots:
                log(f"    {plot_file.name}: ✓ (additional)")
    
    # Check JSON structure
    if json_path.exists():
        with open(json_path) as f:
            json_data = json.load(f)
        log(f"\nJSON structure check:")
        log(f"  per_split_results: {'✓' if 'per_split_results' in json_data else '✗'}")
        log(f"  aggregated_metrics: {'✓' if 'aggregated_metrics' in json_data else '✗'}")
        log(f"  file_hashes: {'✓' if 'file_hashes' in json_data else '✗'}")
        log(f"  splits_evaluated: {'✓' if 'splits_evaluated' in json_data else '✗'}")
    
    log("\n" + "=" * 80)
    log("SUCCESS! H3_full_evaluation folder created.")
    log("=" * 80)
    log(f"\nAll outputs saved to: {output_dir}")
    log(f"  - H3_full_results.json")
    log(f"  - H3_full_summary.md")
    log(f"  - plots/")
    
except Exception as e:
    log(f"\n✗ ERROR: {e}")
    log(traceback.format_exc())
    sys.exit(1)
finally:
    # Write verification log
    log_path = repo_root / "h3_verification_log.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(verification_log))
    log(f"\nVerification log saved to: {log_path}")
