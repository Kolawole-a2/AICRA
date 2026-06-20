#!/usr/bin/env python3
"""Test script to run H3 evaluation with explicit output."""

import sys
import traceback
from pathlib import Path

# Force output
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

print("=" * 80, flush=True)
print("H3 Evaluation Test Run", flush=True)
print("=" * 80, flush=True)
print()

try:
    repo_root = Path(__file__).parent.resolve()
    print(f"Repository root: {repo_root}", flush=True)
    
    # Import
    print("Importing...", flush=True)
    from aicra.experiments.h3_evaluation import run_h3_evaluation
    import yaml
    print("✓ Imported successfully", flush=True)
    
    # Paths
    config = repo_root / "config" / "h3_splits.yaml"
    det = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    lrn = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref = repo_root / "d3fend_reference_pairs.csv"
    out = repo_root / "results" / "H3_full_evaluation"
    
    print(f"\nPaths:", flush=True)
    print(f"  Config: {config} ({'exists' if config.exists() else 'MISSING'})", flush=True)
    print(f"  Det: {det} ({'exists' if det.exists() else 'MISSING'})", flush=True)
    print(f"  Learned: {lrn} ({'exists' if lrn.exists() else 'MISSING'})", flush=True)
    print(f"  Ref: {ref} ({'exists' if ref.exists() else 'MISSING'})", flush=True)
    print(f"  Output: {out}", flush=True)
    
    if not all([config.exists(), det.exists(), lrn.exists(), ref.exists()]):
        print("\n✗ Missing required files!", flush=True)
        sys.exit(1)
    
    print("\n" + "=" * 80, flush=True)
    print("Running evaluation...", flush=True)
    print("=" * 80, flush=True)
    print()
    
    # Run
    results = run_h3_evaluation(
        splits_config_path=config,
        det_mapping_path=det,
        learned_mapping_path=lrn,
        ref_pairs_path=ref,
        output_dir=out,
        repo_root=repo_root,
    )
    
    print("\n" + "=" * 80, flush=True)
    print("Evaluation completed!", flush=True)
    print("=" * 80, flush=True)
    
    # Verify
    json_file = out / "H3_full_results.json"
    md_file = out / "H3_full_summary.md"
    plots_dir = out / "plots"
    
    print(f"\nVerifying outputs:", flush=True)
    print(f"  H3_full_results.json: {'✓' if json_file.exists() else '✗'} ({json_file})", flush=True)
    print(f"  H3_full_summary.md: {'✓' if md_file.exists() else '✗'} ({md_file})", flush=True)
    print(f"  plots/: {'✓' if plots_dir.exists() else '✗'} ({plots_dir})", flush=True)
    
    if plots_dir.exists():
        plots = sorted(plots_dir.glob("*.png"))
        print(f"\n  Plot files ({len(plots)}):", flush=True)
        for p in plots:
            print(f"    - {p.name}", flush=True)
    
    print("\n" + "=" * 80, flush=True)
    print("SUCCESS!", flush=True)
    print("=" * 80, flush=True)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
