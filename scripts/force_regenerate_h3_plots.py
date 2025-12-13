#!/usr/bin/env python3
"""Force regenerate H3 plots with verification."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aicra.experiments.h3_evaluation import create_plots

repo_root = Path(__file__).parent.parent
json_path = repo_root / "results" / "H3_full_evaluation" / "H3_full_results.json"
output_dir = repo_root / "results" / "H3_full_evaluation"
plots_dir = output_dir / "plots"

# Load JSON
print(f"Loading JSON from: {json_path}")
with open(json_path, encoding="utf-8") as f:
    output = json.load(f)

# Extract components
all_results = output.get("per_split_results", [])
aggregated = output.get("aggregated_metrics", {})

print(f"\nFound {len(all_results)} split results")

# Verify data
print("\n=== Data Verification ===")
splits = [r["split"] for r in all_results]
det_dac = [
    r["deterministic"]["mapping_metrics"].get("dac_%", 100.0) for r in all_results
]
learned_dac = [r["learned"]["mapping_metrics"].get("dac_%", 0.0) for r in all_results]
det_prec = [
    r["deterministic"]["actionable_metrics"]["actionable_precision"]
    for r in all_results
]
learned_prec = [
    r["learned"]["actionable_metrics"]["actionable_precision"] for r in all_results
]

print("\nDAC values per split:")
for s, d, learned in zip(splits, det_dac, learned_dac, strict=False):
    print(f"  {s}: det={d}%, learned={learned}%")

print("\nPrecision values per split:")
for s, d, learned in zip(splits, det_prec, learned_prec, strict=False):
    print(f"  {s}: det={d:.4f}, learned={learned:.4f}")

print("\nAggregated metrics:")
print(
    f"  DAC (det): mean={aggregated['deterministic']['dac_%']['mean']:.2f}%, std={aggregated['deterministic']['dac_%']['std']:.2f}%"
)
print(
    f"  DAC (learned): mean={aggregated['learned']['dac_%']['mean']:.2f}%, std={aggregated['learned']['dac_%']['std']:.2f}%"
)
print(
    f"  Precision (det): mean={aggregated['deterministic']['actionable_precision']['mean']:.4f}, std={aggregated['deterministic']['actionable_precision']['std']:.4f}"
)
print(
    f"  Precision (learned): mean={aggregated['learned']['actionable_precision']['mean']:.4f}, std={aggregated['learned']['actionable_precision']['std']:.4f}"
)

# Delete old plots
print(f"\nDeleting old plots in {plots_dir}...")
if plots_dir.exists():
    for f in plots_dir.glob("*.png"):
        print(f"  Deleting: {f.name}")
        f.unlink()

# Regenerate plots
print(f"\nRegenerating plots to: {plots_dir}")
create_plots(all_results, aggregated, output_dir)

# Verify plots were created
print("\n✓ Plots regenerated!")
print("\nPlot files created:")
for f in sorted(plots_dir.glob("*.png")):
    print(f"  - {f.name} ({f.stat().st_size} bytes)")
