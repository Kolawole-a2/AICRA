#!/usr/bin/env python3
"""Simple test plot to verify both bars show."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

json_path = Path("results/H3_full_evaluation/H3_full_results.json")
plots_dir = Path("results/H3_full_evaluation/plots")

# Load and extract
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

all_results = data["per_split_results"]
splits = [r["split"] for r in all_results]
det_dac = [float(r["deterministic"]["mapping_metrics"]["dac_%"]) for r in all_results]
learned_dac = [float(r["learned"]["mapping_metrics"]["dac_%"]) for r in all_results]

print("Values to plot:")
print(f"  Splits: {splits}")
print(f"  Det DAC: {det_dac}")
print(f"  Learned DAC: {learned_dac}")

# Create simple test plot
x = np.arange(len(splits))
width = 0.4

fig, ax = plt.subplots(figsize=(14, 8))

# Plot deterministic - GREEN, LEFT side
bars_det = ax.bar(x - width/2, det_dac, width, 
                  label="Deterministic (Green)", 
                  color="green", 
                  alpha=1.0, 
                  edgecolor='black', 
                  linewidth=2)

# Plot learned - BLUE, RIGHT side  
bars_learned = ax.bar(x + width/2, learned_dac, width, 
                      label="Learned (Blue)", 
                      color="blue", 
                      alpha=1.0, 
                      edgecolor='black', 
                      linewidth=2)

# Add value labels
for bar, val, split in zip(bars_det, det_dac, splits):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'DET: {val:.1f}%', ha='center', va='bottom', 
            fontsize=12, fontweight='bold', color='darkgreen',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

for bar, val, split in zip(bars_learned, learned_dac, splits):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'LRN: {val:.1f}%', ha='center', va='bottom', 
            fontsize=12, fontweight='bold', color='darkblue',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_xlabel("Split", fontsize=14, fontweight='bold')
ax.set_ylabel("DAC (%)", fontsize=14, fontweight='bold')
ax.set_title("TEST: DAC Per Split - Both Bars Should Be Visible\n(Green=Deterministic, Blue=Learned)", 
             fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(splits, fontsize=12)
ax.legend(fontsize=13, loc='upper left')
ax.grid(alpha=0.3, axis='y', linestyle='--')
ax.set_ylim([0, 110])

# Add text annotation
ax.text(0.02, 0.98, f'Det values: {det_dac}\nLearned values: {learned_dac}', 
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(plots_dir / "dac_per_split.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Test plot saved to: {plots_dir / 'dac_per_split.png'}")
print("This plot should show:")
print("  - GREEN bars on LEFT (Deterministic)")
print("  - BLUE bars on RIGHT (Learned)")
print("  - Value labels on each bar")





