#!/usr/bin/env python3
"""Regenerate H3 plots from existing JSON results."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aicra.experiments.h3_evaluation import aggregate_metrics, create_plots

repo_root = Path(__file__).parent.parent
json_path = repo_root / "results" / "H3_full_evaluation" / "H3_full_results.json"
output_dir = repo_root / "results" / "H3_full_evaluation"

# Load JSON
print(f"Loading JSON from: {json_path}")
with open(json_path, encoding="utf-8") as f:
    output = json.load(f)

# Extract components
all_results = output.get("per_split_results", [])
aggregated = output.get("aggregated_metrics", {})

print(f"Found {len(all_results)} split results")

# Regenerate aggregated metrics if needed (to ensure consistency)
if not aggregated or "dac_%" not in aggregated.get("deterministic", {}):
    print("Regenerating aggregated metrics...")
    aggregated = aggregate_metrics(all_results)

# Regenerate plots
print(f"Regenerating plots to: {output_dir / 'plots'}")
create_plots(all_results, aggregated, output_dir)

print("✓ Plots regenerated successfully!")














