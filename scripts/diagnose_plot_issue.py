#!/usr/bin/env python3
"""Diagnose plot issue by showing exact values at each step."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aicra.experiments.h3_evaluation import create_plots

json_path = Path("results/H3_full_evaluation/H3_full_results.json")
output_dir = Path("results/H3_full_evaluation")

print("=" * 80)
print("PLOT DIAGNOSTIC - Step by Step Verification")
print("=" * 80)

# Step 1: Load JSON
print("\n[STEP 1] Loading JSON...")
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
print(f"✓ Loaded JSON from: {json_path}")

# Step 2: Extract data
print("\n[STEP 2] Extracting data from JSON...")
all_results = data["per_split_results"]
aggregated = data["aggregated_metrics"]

print(f"✓ Found {len(all_results)} split results")

# Step 3: Extract values exactly as plotting code does
print("\n[STEP 3] Extracting values (same as plotting code)...")
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

print("\nValues extracted for plotting:")
print(f"  Splits: {splits}")
print(f"  Det DAC: {det_dac}")
print(f"  Learned DAC: {learned_dac}")
print(f"  Det Precision: {[f'{p:.4f}' for p in det_prec]}")
print(f"  Learned Precision: {[f'{p:.4f}' for p in learned_prec]}")

# Step 4: Check aggregated metrics
print("\n[STEP 4] Aggregated metrics (for summary plot):")
print(
    f"  DAC det: mean={aggregated['deterministic']['dac_%']['mean']:.2f}%, std={aggregated['deterministic']['dac_%']['std']:.2f}%"
)
print(
    f"  DAC learned: mean={aggregated['learned']['dac_%']['mean']:.2f}%, std={aggregated['learned']['dac_%']['std']:.2f}%"
)
print(
    f"  Precision det: mean={aggregated['deterministic']['actionable_precision']['mean']:.4f}, std={aggregated['deterministic']['actionable_precision']['std']:.4f}"
)
print(
    f"  Precision learned: mean={aggregated['learned']['actionable_precision']['mean']:.4f}, std={aggregated['learned']['actionable_precision']['std']:.4f}"
)

# Step 5: Verify JSON structure
print("\n[STEP 5] Verifying JSON structure for first split...")
first_result = all_results[0]
print(f"  Split: {first_result['split']}")
print(f"  Has 'deterministic'? {'deterministic' in first_result}")
if "deterministic" in first_result:
    print(
        f"  Has 'mapping_metrics'? {'mapping_metrics' in first_result['deterministic']}"
    )
    if "mapping_metrics" in first_result["deterministic"]:
        mm = first_result["deterministic"]["mapping_metrics"]
        print(f"  Keys in mapping_metrics: {list(mm.keys())}")
        print(f"  dac_% value: {mm.get('dac_%', 'NOT FOUND')}")
        print(f"  Has dac_internal_%? {'dac_internal_%' in mm}")
        print(f"  Has dac_external_%? {'dac_external_%' in mm}")

# Step 6: Generate plots
print("\n[STEP 6] Generating plots...")
plots_dir = output_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Delete old plots
for f in plots_dir.glob("*.png"):
    f.unlink()
    print(f"  Deleted: {f.name}")

# Generate
create_plots(all_results, aggregated, output_dir)

print(f"\n✓ Plots generated in: {plots_dir}")
print("\nGenerated files:")
for f in sorted(plots_dir.glob("*.png")):
    size = f.stat().st_size
    print(f"  - {f.name} ({size:,} bytes)")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print("\nThe plots now have VALUE LABELS showing the exact values.")
print("Please check the plots and tell me:")
print("1. What values are shown in the plot (from the labels on the bars)?")
print("2. What values do you EXPECT to see?")
print("3. Which specific plot file is showing wrong values?")
print("=" * 80)






