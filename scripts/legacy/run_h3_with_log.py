#!/usr/bin/env python3
"""Run H3 evaluation with full logging to file."""

import sys
import traceback
from pathlib import Path
from datetime import datetime

# Log file
log_file = Path("h3_execution_log.txt")
log_file.write_text(f"H3 Evaluation Execution Log - {datetime.now()}\n{'='*80}\n\n")

def log(msg):
    """Write to both console and log file."""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)

try:
    log("Starting H3 evaluation execution...")
    log("")
    
    repo_root = Path(__file__).parent.resolve()
    log(f"Repository root: {repo_root}")
    log("")
    
    # Check Python
    log(f"Python version: {sys.version}")
    log(f"Python executable: {sys.executable}")
    log("")
    
    # Import
    log("Step 1: Importing modules...")
    try:
        from aicra.experiments.h3_evaluation import run_h3_evaluation
        import yaml
        log("✓ Imports successful")
    except Exception as e:
        log(f"✗ Import failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)
    
    log("")
    log("Step 2: Checking input files...")
    
    # Paths
    config_path = repo_root / "config" / "h3_splits.yaml"
    det_path = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    lrn_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_path = repo_root / "d3fend_reference_pairs.csv"
    output_dir = repo_root / "results" / "H3_full_evaluation"
    
    files = {
        "Config": config_path,
        "Deterministic": det_path,
        "Learned": lrn_path,
        "Reference": ref_path,
    }
    
    all_exist = True
    for name, path in files.items():
        exists = path.exists()
        log(f"  {name}: {path} - {'✓' if exists else '✗'}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        log("")
        log("✗ ERROR: Missing required files!")
        sys.exit(1)
    
    log("")
    log(f"Step 3: Setting output directory: {output_dir}")
    log("")
    
    log("Step 4: Running evaluation...")
    log("=" * 80)
    
    try:
        results = run_h3_evaluation(
            splits_config_path=config_path,
            det_mapping_path=det_path,
            learned_mapping_path=lrn_path,
            ref_pairs_path=ref_path,
            output_dir=output_dir,
            repo_root=repo_root,
        )
        log("")
        log("✓ Evaluation function completed")
    except Exception as e:
        log("")
        log(f"✗ Evaluation failed: {e}")
        log(traceback.format_exc())
        sys.exit(1)
    
    log("")
    log("Step 5: Verifying outputs...")
    
    json_file = output_dir / "H3_full_results.json"
    md_file = output_dir / "H3_full_summary.md"
    plots_dir = output_dir / "plots"
    
    log(f"  H3_full_results.json: {'✓' if json_file.exists() else '✗'} - {json_file}")
    log(f"  H3_full_summary.md: {'✓' if md_file.exists() else '✗'} - {md_file}")
    log(f"  plots/ directory: {'✓' if plots_dir.exists() else '✗'} - {plots_dir}")
    
    if plots_dir.exists():
        plots = sorted(plots_dir.glob("*.png"))
        log(f"    Plot files found: {len(plots)}")
        for p in plots:
            log(f"      - {p.name}")
    
    log("")
    log("=" * 80)
    log("Execution completed!")
    log("=" * 80)
    log(f"\nLog file: {log_file}")
    
except Exception as e:
    log("")
    log("=" * 80)
    log("FATAL ERROR")
    log("=" * 80)
    log(f"Error: {e}")
    log("")
    log(traceback.format_exc())
    sys.exit(1)
