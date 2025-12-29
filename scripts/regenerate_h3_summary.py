#!/usr/bin/env python3
"""Regenerate H3 markdown summary with mapping_behavior section."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aicra.experiments.h3_evaluation import generate_markdown_report

repo_root = Path(__file__).parent.parent
json_path = repo_root / "results" / "H3_full_evaluation" / "H3_full_results.json"
output_path = repo_root / "results" / "H3_full_evaluation" / "H3_full_summary.md"

# Load JSON
print(f"Loading JSON from: {json_path}")
with open(json_path, encoding="utf-8") as f:
    output = json.load(f)

# Extract components
all_results = output.get("per_split_results", [])
aggregated = output.get("aggregated_metrics", {})
splits_config = output.get("splits_config", {})
file_hashes = output.get("file_hashes", {})

print(f"Found {len(all_results)} split results")
print(f"Has mapping_behavior: {'mapping_behavior' in output}")

# Regenerate markdown
print(f"Regenerating markdown summary to: {output_path}")
generate_markdown_report(
    all_results=all_results,
    aggregated=aggregated,
    splits_config=splits_config,
    file_hashes=file_hashes,
    output_path=output_path,
    output=output,  # Pass full output to include mapping_behavior
)

print("✓ Markdown summary regenerated successfully!")














