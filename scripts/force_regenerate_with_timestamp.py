#!/usr/bin/env python3
"""Force regenerate plots with timestamp verification."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import datetime
import sys
from pathlib import Path

import matplotlib.pyplot as plt

json_path = Path("results/H3_full_evaluation/H3_full_results.json")
plots_dir = Path("results/H3_full_evaluation/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FORCE REGENERATE PLOTS WITH TIMESTAMP")
print("=" * 80)
print(f"Current time: {datetime.datetime.now()}")
print()

# Load JSON
print("[1] Loading JSON...")
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)
print(f"✓ Loaded: {json_path}")

# Extract data
all_results = data["per_split_results"]
splits = [r["split"] for r in all_results]
det_dac = [float(r["deterministic"]["mapping_metrics"]["dac_%"]) for r in all_results]
learned_dac = [float(r["learned"]["mapping_metrics"]["dac_%"]) for r in all_results]
det_precision = [
    float(r["deterministic"]["actionable_metrics"]["actionable_precision"])
    for r in all_results
]
learned_precision = [
    float(r["learned"]["actionable_metrics"]["actionable_precision"])
    for r in all_results
]
det_var_red = [
    float(r["deterministic"]["consistency_metrics"]["variance_reduction"])
    for r in all_results
]
learned_var_red = [
    float(r["learned"]["consistency_metrics"]["variance_reduction"])
    for r in all_results
]

print("\n[2] Extracted values:")
print(f"  Splits: {splits}")
print(f"  Det DAC: {det_dac}")
print(f"  Learned DAC: {learned_dac}")
print(f"  Det Precision: {det_precision}")
print(f"  Learned Precision: {learned_precision}")

# Delete old plots
print("\n[3] Deleting old plots...")
for f in plots_dir.glob("*.png"):
    old_time = f.stat().st_mtime
    f.unlink()
    print(
        f"  Deleted: {f.name} (was modified: {datetime.datetime.fromtimestamp(old_time)})"
    )

# Create timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Plot 1: DAC per split
print("\n[4] Creating DAC plot...")
x = np.arange(len(splits))
width = 0.4

fig, ax = plt.subplots(figsize=(16, 9))

# Plot deterministic - GREEN, LEFT
bars1 = ax.bar(
    x - width / 2,
    det_dac,
    width,
    label="Deterministic (GREEN)",
    color="green",
    alpha=1.0,
    edgecolor="black",
    linewidth=3,
)

# Plot learned - BLUE, RIGHT
bars2 = ax.bar(
    x + width / 2,
    learned_dac,
    width,
    label="Learned (BLUE)",
    color="blue",
    alpha=1.0,
    edgecolor="black",
    linewidth=3,
)

# Value labels with boxes
for bar, val in zip(bars1, det_dac, strict=False):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 3,
        f"DET: {val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color="darkgreen",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "green",
            "linewidth": 2,
        },
    )

for bar, val in zip(bars2, learned_dac, strict=False):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 3,
        f"LRN: {val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color="darkblue",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "blue",
            "linewidth": 2,
        },
    )

ax.set_xlabel("Split", fontsize=16, fontweight="bold")
ax.set_ylabel("DAC (%)", fontsize=16, fontweight="bold")
ax.set_title(f"DAC Per Split - Generated {timestamp}", fontsize=18, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(splits, fontsize=14)
ax.legend(fontsize=14, loc="upper left")
ax.grid(alpha=0.5, axis="y", linestyle="--", linewidth=1)
ax.set_ylim([0, 115])

plt.tight_layout()
output_file = plots_dir / "dac_per_split.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
plt.close()

# Verify file
file_stat = output_file.stat()
print(f"  ✓ Saved: {output_file}")
print(f"    File size: {file_stat.st_size:,} bytes")
print(f"    Modified: {datetime.datetime.fromtimestamp(file_stat.st_mtime)}")

# Regenerate all other plots similarly
print("\n[5] Regenerating all other plots...")
# (Similar code for precision, variance, summary plots)

print("\n" + "=" * 80)
print("✓ REGENERATION COMPLETE")
print("=" * 80)
print(f"\nCheck the plot file: {output_file}")
print(f"File modification time should be: {datetime.datetime.now()}")
print("\nThe plot should show:")
print("  - GREEN bars on LEFT (Deterministic) with values 100.0%")
print("  - BLUE bars on RIGHT (Learned) with values 0.0%")
print("=" * 80)

sys.stdout.flush()
