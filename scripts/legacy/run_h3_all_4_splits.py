#!/usr/bin/env python3
"""Run H3 evaluation with all 4 splits including smoke_test."""

import sys
from pathlib import Path
import json
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from aicra.experiments.h3_evaluation import (
    evaluate_split,
    load_mapping_csv,
    aggregate_metrics,
    generate_markdown_report,
    create_plots,
    compute_file_hash,
)

print("=" * 80)
print("H3 Evaluation - All 4 Splits (main, small_ember, full_ember, smoke_test)")
print("=" * 80)

# Load config
config_path = Path("config/h3_splits.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

splits = config["splits"]
repo_root = Path(".")

print(f"\nConfig has {len(splits)} splits: {list(splits.keys())}")

# Load mappings
print("\nLoading mappings...")
det_mapping = load_mapping_csv(Path("data/mappings/deterministic_lookup.csv"))
learned_mapping = load_mapping_csv(Path("data/mappings/learned_mapping.csv"))
ref_pairs = load_mapping_csv(Path("d3fend_reference_pairs.csv"))

# Compute file hashes
file_hashes = {
    "deterministic": compute_file_hash(Path("data/mappings/deterministic_lookup.csv")),
    "learned": compute_file_hash(Path("data/mappings/learned_mapping.csv")),
    "reference": compute_file_hash(Path("d3fend_reference_pairs.csv")),
}

# Evaluate all splits
print("\nEvaluating splits...")
all_results = []
for name, rel_path in splits.items():
    full_path = repo_root / rel_path
    if not full_path.exists():
        print(f"  ⚠ Skipping {name}: file not found at {full_path}")
        continue
    
    try:
        print(f"  Evaluating {name}...")
        result = evaluate_split(name, full_path, det_mapping, learned_mapping, ref_pairs)
        all_results.append(result)
        print(f"    ✓ {name}: {result['n_samples']} samples, {result['n_techniques']} techniques")
    except Exception as e:
        print(f"    ✗ Error evaluating {name}: {e}")
        import traceback
        traceback.print_exc()

print(f"\nSuccessfully evaluated {len(all_results)} splits")

# Aggregate
print("\nAggregating metrics...")
aggregated = aggregate_metrics(all_results)

# Save results
output_dir = Path("results/H3_full_evaluation")
output_dir.mkdir(parents=True, exist_ok=True)

output = {
    "per_split_results": all_results,
    "aggregated_metrics": aggregated,
    "file_hashes": file_hashes,
    "splits_evaluated": [r["split"] for r in all_results],
    "splits_config": splits,
    "splits_skipped": [name for name in splits.keys() if name not in [r["split"] for r in all_results]],
    "mapping_overlap": {},  # Will be computed in full run_h3_evaluation
}

json_path = output_dir / "H3_full_results.json"
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"  ✓ Saved JSON to {json_path}")

# Generate markdown
markdown_path = output_dir / "H3_full_summary.md"
generate_markdown_report(all_results, aggregated, config, file_hashes, markdown_path, {})
print(f"  ✓ Saved markdown to {markdown_path}")

# Create plots
create_plots(all_results, aggregated, output_dir)
print(f"  ✓ Created plots in {output_dir / 'plots'}")

print("\n" + "=" * 80)
print("H3 Evaluation Complete!")
print("=" * 80)
print(f"Evaluated splits: {output['splits_evaluated']}")
print(f"Total splits: {len(output['splits_evaluated'])}")
if output['splits_skipped']:
    print(f"Skipped splits: {output['splits_skipped']}")
