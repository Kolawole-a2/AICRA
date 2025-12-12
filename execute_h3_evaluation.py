#!/usr/bin/env python3
"""
Execute H3 evaluation and save all output to a log file.
"""

import sys
import traceback
from pathlib import Path
from datetime import datetime

# Redirect all output to a log file
log_path = Path("h3_evaluation_output.log")
log_file = open(log_path, "w", encoding="utf-8")

def log(msg):
    """Write to log file and also try to print."""
    log_file.write(msg + "\n")
    log_file.flush()
    try:
        print(msg, flush=True)
    except:
        pass

class TeeOutput:
    """Tee output to both file and stdout."""
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout
    
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        try:
            self.stdout.write(data)
            self.stdout.flush()
        except:
            pass
    
    def flush(self):
        self.file.flush()
        try:
            self.stdout.flush()
        except:
            pass

# Redirect stdout and stderr
sys.stdout = TeeOutput(log_file)
sys.stderr = TeeOutput(log_file)

try:
    log("=" * 80)
    log(f"H3 Evaluation Execution - {datetime.now()}")
    log("=" * 80)
    log("")
    
    repo_root = Path(__file__).parent.resolve()
    log(f"Repository root: {repo_root}")
    log("")
    
    # Import
    log("Importing modules...")
    from aicra.experiments.h3_evaluation import run_h3_evaluation
    import yaml
    log("✓ Imports successful")
    log("")
    
    # Paths
    config_path = repo_root / "config" / "h3_splits.yaml"
    det_path = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    lrn_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_path = repo_root / "d3fend_reference_pairs.csv"
    output_dir = repo_root / "results" / "H3_full_evaluation"
    
    log("Input files:")
    log(f"  Config: {config_path} ({'✓' if config_path.exists() else '✗'})")
    log(f"  Deterministic: {det_path} ({'✓' if det_path.exists() else '✗'})")
    log(f"  Learned: {lrn_path} ({'✓' if lrn_path.exists() else '✗'})")
    log(f"  Reference: {ref_path} ({'✓' if ref_path.exists() else '✗'})")
    log(f"  Output: {output_dir}")
    log("")
    
    if not all([config_path.exists(), det_path.exists(), lrn_path.exists(), ref_path.exists()]):
        log("✗ ERROR: Missing required files!")
        sys.exit(1)
    
    log("=" * 80)
    log("Running H3 evaluation...")
    log("=" * 80)
    log("")
    
    # Run evaluation
    results = run_h3_evaluation(
        splits_config_path=config_path,
        det_mapping_path=det_path,
        learned_mapping_path=lrn_path,
        ref_pairs_path=ref_path,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    
    log("")
    log("=" * 80)
    log("Evaluation function returned successfully!")
    log("=" * 80)
    log("")
    
    # Verify outputs
    log("Verifying outputs:")
    json_file = output_dir / "H3_full_results.json"
    md_file = output_dir / "H3_full_summary.md"
    plots_dir = output_dir / "plots"
    
    json_exists = json_file.exists()
    md_exists = md_file.exists()
    plots_exists = plots_dir.exists()
    
    log(f"  H3_full_results.json: {'✓' if json_exists else '✗'} - {json_file}")
    log(f"  H3_full_summary.md: {'✓' if md_exists else '✗'} - {md_file}")
    log(f"  plots/ directory: {'✓' if plots_exists else '✗'} - {plots_dir}")
    
    if plots_exists:
        plots = sorted(plots_dir.glob("*.png"))
        log(f"    Plot files: {len(plots)}")
        for p in plots:
            log(f"      - {p.name}")
    
    log("")
    log("=" * 80)
    if json_exists and md_exists and plots_exists:
        log("SUCCESS! All outputs created.")
    else:
        log("WARNING: Some outputs may be missing.")
    log("=" * 80)
    log(f"\nLog file: {log_path}")
    
except Exception as e:
    log("")
    log("=" * 80)
    log("ERROR OCCURRED")
    log("=" * 80)
    log(f"Error: {e}")
    log("")
    log(traceback.format_exc())
finally:
    log_file.close()
