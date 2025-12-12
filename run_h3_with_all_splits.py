#!/usr/bin/env python3
"""Run H3 evaluation ensuring all 4 splits are included."""

import sys
import logging
from pathlib import Path
import yaml

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Load config
config_path = Path("config/h3_splits.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

splits = config["splits"]
repo_root = Path(".")

print("=" * 80)
print("H3 Evaluation - All Splits")
print("=" * 80)
print(f"\nSplits in config: {len(splits)}")
for name, rel_path in splits.items():
    full_path = repo_root / rel_path
    exists = full_path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {name}: {rel_path}")
    if not exists:
        print(f"    ERROR: File not found at {full_path.absolute()}")

# Verify all files exist
missing = [name for name, rel_path in splits.items() if not (repo_root / rel_path).exists()]
if missing:
    print(f"\nERROR: Missing splits: {missing}")
    sys.exit(1)

# Run evaluation
print("\n" + "=" * 80)
print("Running H3 Evaluation...")
print("=" * 80)

from aicra.experiments.h3_evaluation import run_h3_evaluation

result = run_h3_evaluation(
    splits_config_path=config_path,
    det_mapping_path=repo_root / "data/mappings/deterministic_lookup.csv",
    learned_mapping_path=repo_root / "data/mappings/learned_mapping.csv",
    ref_pairs_path=repo_root / "d3fend_reference_pairs.csv",
    output_dir=repo_root / "results/H3_full_evaluation",
    repo_root=repo_root,
)

print("\n" + "=" * 80)
print("Evaluation Complete!")
print("=" * 80)
print(f"Evaluated splits: {result['splits_evaluated']}")
print(f"Number of splits: {len(result['splits_evaluated'])}")
print(f"Expected: {len(splits)}")

if len(result['splits_evaluated']) != len(splits):
    print(f"\nWARNING: Expected {len(splits)} splits but only {len(result['splits_evaluated'])} were evaluated!")
    print(f"Missing: {set(splits.keys()) - set(result['splits_evaluated'])}")
else:
    print("\n✓ All splits evaluated successfully!")

print("\nPer-split summary:")
for split_result in result["per_split_results"]:
    print(f"  {split_result['split']}: {split_result['n_samples']} samples, {split_result['n_techniques']} techniques")
