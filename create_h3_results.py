#!/usr/bin/env python3
"""
Standalone script to create H3_full_evaluation folder with all outputs.
Run this script directly: python create_h3_results.py
"""

import json
import sys
import traceback
from pathlib import Path

def main():
    print("=" * 80)
    print("H3 Full Evaluation - Creating results folder")
    print("=" * 80)
    print()
    
    try:
        # Get repository root
        repo_root = Path(__file__).parent.resolve()
        print(f"Repository root: {repo_root}")
        print()
        
        # Check and import
        print("Importing modules...")
        try:
            from aicra.experiments.h3_evaluation import run_h3_evaluation
            import yaml
            print("✓ Imports successful")
        except ImportError as e:
            print(f"✗ Import error: {e}")
            traceback.print_exc()
            return 1
        
        # Set up paths
        splits_config_path = repo_root / "config" / "h3_splits.yaml"
        det_mapping_path = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
        learned_mapping_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
        ref_pairs_path = repo_root / "d3fend_reference_pairs.csv"
        output_dir = repo_root / "results" / "H3_full_evaluation"
        
        print("\nChecking input files:")
        files_ok = True
        for name, path in [
            ("Splits config", splits_config_path),
            ("Deterministic mapping", det_mapping_path),
            ("Learned mapping", learned_mapping_path),
            ("Reference pairs", ref_pairs_path),
        ]:
            exists = path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {name}: {path}")
            if not exists:
                files_ok = False
        
        if not files_ok:
            print("\n✗ ERROR: Missing required input files!")
            return 1
        
        print(f"\nOutput directory: {output_dir}")
        print()
        
        # Run evaluation
        print("=" * 80)
        print("Running H3 evaluation...")
        print("=" * 80)
        print()
        
        results = run_h3_evaluation(
            splits_config_path=splits_config_path,
            det_mapping_path=det_mapping_path,
            learned_mapping_path=learned_mapping_path,
            ref_pairs_path=ref_pairs_path,
            output_dir=output_dir,
            repo_root=repo_root,
        )
        
        print()
        print("=" * 80)
        print("Evaluation completed!")
        print("=" * 80)
        print()
        
        # Verify outputs
        print("Verifying outputs:")
        json_path = output_dir / "H3_full_results.json"
        md_path = output_dir / "H3_full_summary.md"
        plots_dir = output_dir / "plots"
        
        json_exists = json_path.exists()
        md_exists = md_path.exists()
        plots_exists = plots_dir.exists()
        
        print(f"  {'✓' if json_exists else '✗'} H3_full_results.json")
        print(f"  {'✓' if md_exists else '✗'} H3_full_summary.md")
        print(f"  {'✓' if plots_exists else '✗'} plots/ directory")
        
        if plots_exists:
            plot_files = sorted(plots_dir.glob("*.png"))
            print(f"\n  Plot files ({len(plot_files)}):")
            expected_plots = [
                "dac_per_split.png",
                "precision_per_split.png", 
                "variance_reduction_per_split.png",
                "summary_metrics.png"
            ]
            for expected in expected_plots:
                exists = (plots_dir / expected).exists()
                print(f"    {'✓' if exists else '✗'} {expected}")
        
        # Check JSON content
        if json_exists:
            try:
                with open(json_path) as f:
                    json_data = json.load(f)
                print(f"\n  JSON structure:")
                print(f"    ✓ per_split_results: {len(json_data.get('per_split_results', []))} splits")
                print(f"    ✓ aggregated_metrics: {'present' if 'aggregated_metrics' in json_data else 'missing'}")
                print(f"    ✓ file_hashes: {'present' if 'file_hashes' in json_data else 'missing'}")
                print(f"    ✓ splits_evaluated: {json_data.get('splits_evaluated', [])}")
            except Exception as e:
                print(f"    ✗ Error reading JSON: {e}")
        
        print()
        print("=" * 80)
        if json_exists and md_exists and plots_exists:
            print("SUCCESS! H3_full_evaluation folder created with all outputs.")
        else:
            print("WARNING: Some outputs may be missing.")
        print("=" * 80)
        print()
        print(f"Results location: {output_dir}")
        print(f"  - H3_full_results.json")
        print(f"  - H3_full_summary.md")
        print(f"  - plots/")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR occurred!")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
