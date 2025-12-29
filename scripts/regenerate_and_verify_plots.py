#!/usr/bin/env python3
"""Regenerate H3 plots and print exact values for verification."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aicra.experiments.h3_evaluation import create_plots

json_path = Path("results/H3_full_evaluation/H3_full_results.json")
output_dir = Path("results/H3_full_evaluation")
plots_dir = output_dir / "plots"

# Load data
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

all_results = data["per_split_results"]
aggregated = data["aggregated_metrics"]

# Extract values
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

print("=" * 80)
print("PLOT VALUES VERIFICATION")
print("=" * 80)
print(f"\nSplits: {splits}")
print("\nDAC values (dac_per_split.png):")
print(f"  Deterministic: {det_dac}")
print(f"  Learned: {learned_dac}")
print("\nPrecision values (precision_per_split.png):")
print(f"  Deterministic: {[f'{p:.4f}' for p in det_prec]}")
print(f"  Learned: {[f'{p:.4f}' for p in learned_prec]}")
print("\nSummary plot (summary_metrics.png):")
print(
    f"  DAC det: {aggregated['deterministic']['dac_%']['mean']:.2f}% ± {aggregated['deterministic']['dac_%']['std']:.2f}%"
)
print(
    f"  DAC learned: {aggregated['learned']['dac_%']['mean']:.2f}% ± {aggregated['learned']['dac_%']['std']:.2f}%"
)
print(
    f"  Precision det: {aggregated['deterministic']['actionable_precision']['mean']:.4f} ± {aggregated['deterministic']['actionable_precision']['std']:.4f}"
)
print(
    f"  Precision learned: {aggregated['learned']['actionable_precision']['mean']:.4f} ± {aggregated['learned']['actionable_precision']['std']:.4f}"
)

# Delete old plots
print("\nDeleting old plots...")
for f in plots_dir.glob("*.png"):
    f.unlink()

# Regenerate
print("Regenerating plots...")
create_plots(all_results, aggregated, output_dir)

print(f"\n✓ Plots regenerated in: {plots_dir}")
print("\nGenerated files:")
for f in sorted(plots_dir.glob("*.png")):
    print(f"  - {f.name}")

print("\n" + "=" * 80)
print("If plots still show wrong values:")
print("1. Close the plot files in your viewer/IDE (they may be cached)")
print("2. Reopen the plot files")
print("3. Check the file timestamps to confirm they were just regenerated")
print("=" * 80)














