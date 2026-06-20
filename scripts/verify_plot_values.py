#!/usr/bin/env python3
"""Verify and print exact values being plotted."""

import json
from pathlib import Path

json_path = Path("results/H3_full_evaluation/H3_full_results.json")
data = json.load(open(json_path, encoding="utf-8"))

all_results = data["per_split_results"]
aggregated = data["aggregated_metrics"]

print("=" * 80)
print("EXACT VALUES BEING PLOTTED")
print("=" * 80)

# Per-split plots
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
det_var = [
    r["deterministic"]["consistency_metrics"]["variance_reduction"] for r in all_results
]
learned_var = [
    r["learned"]["consistency_metrics"]["variance_reduction"] for r in all_results
]

print("\n1. DAC PER SPLIT PLOT (dac_per_split.png):")
print("   Splits:", splits)
print("   Deterministic DAC:", det_dac)
print("   Learned DAC:", learned_dac)

print("\n2. PRECISION PER SPLIT PLOT (precision_per_split.png):")
print("   Splits:", splits)
print("   Deterministic Precision:", [f"{p:.4f}" for p in det_prec])
print("   Learned Precision:", [f"{p:.4f}" for p in learned_prec])

print("\n3. VARIANCE REDUCTION PER SPLIT PLOT (variance_reduction_per_split.png):")
print("   Splits:", splits)
print("   Deterministic Var Red:", [f"{v:.6f}" for v in det_var])
print("   Learned Var Red:", [f"{v:.6f}" for v in learned_var])

print("\n4. SUMMARY METRICS PLOT (summary_metrics.png):")
print("   DAC:")
print(
    f"     Deterministic: mean={aggregated['deterministic']['dac_%']['mean']:.2f}%, std={aggregated['deterministic']['dac_%']['std']:.2f}%"
)
print(
    f"     Learned: mean={aggregated['learned']['dac_%']['mean']:.2f}%, std={aggregated['learned']['dac_%']['std']:.2f}%"
)
print("   Precision:")
print(
    f"     Deterministic: mean={aggregated['deterministic']['actionable_precision']['mean']:.4f}, std={aggregated['deterministic']['actionable_precision']['std']:.4f}"
)
print(
    f"     Learned: mean={aggregated['learned']['actionable_precision']['mean']:.4f}, std={aggregated['learned']['actionable_precision']['std']:.4f}"
)
print("   Variance Reduction:")
print(
    f"     Deterministic: mean={aggregated['deterministic']['variance_reduction']['mean']:.6f}, std={aggregated['deterministic']['variance_reduction']['std']:.6f}"
)
print(
    f"     Learned: mean={aggregated['learned']['variance_reduction']['mean']:.6f}, std={aggregated['learned']['variance_reduction']['std']:.6f}"
)

print("\n" + "=" * 80)
print("If your plots show different values, please:")
print("1. Close and reopen the plot files (they may be cached)")
print("2. Check which specific plot file you're viewing")
print("3. Tell me which plot and what values you're seeing")
print("=" * 80)
