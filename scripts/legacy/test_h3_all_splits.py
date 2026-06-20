#!/usr/bin/env python3
"""Test H3 evaluation with all 4 splits."""

import sys
from pathlib import Path
import yaml

print("=" * 80)
print("Testing H3 Evaluation with All 4 Splits")
print("=" * 80)

# Check config
config_path = Path("config/h3_splits.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

splits = config["splits"]
print(f"\nSplits in config: {len(splits)}")
for name, path in splits.items():
    full_path = Path(path)
    exists = full_path.exists()
    print(f"  {name}: {path} -> {'✓' if exists else '✗'}")
    if not exists:
        print(f"    ERROR: File not found!")

# Run evaluation
print("\n" + "=" * 80)
print("Running H3 Evaluation...")
print("=" * 80)

from aicra.experiments.h3_evaluation import run_h3_evaluation

try:
    result = run_h3_evaluation(
        splits_config_path=config_path,
        det_mapping_path=Path("data/mappings/deterministic_lookup.csv"),
        learned_mapping_path=Path("data/mappings/learned_mapping.csv"),
        ref_pairs_path=Path("d3fend_reference_pairs.csv"),
        output_dir=Path("results/H3_full_evaluation"),
        repo_root=Path("."),
    )
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"Evaluated splits: {result['splits_evaluated']}")
    print(f"Number of splits: {len(result['splits_evaluated'])}")
    print("\nPer-split results:")
    for split_result in result["per_split_results"]:
        print(f"  {split_result['split']}: {split_result['n_samples']} samples, {split_result['n_techniques']} techniques")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
