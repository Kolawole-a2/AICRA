#!/usr/bin/env python3
"""Regenerate all H3 plots from JSON - completely fresh, no caching."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
json_path = Path("results/H3_full_evaluation/H3_full_results.json")
plots_dir = Path("results/H3_full_evaluation/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FRESH PLOT REGENERATION - Direct from JSON")
print("=" * 80)

# Load JSON
print("\n[1] Loading JSON...")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"✓ Loaded: {json_path}")

# Extract data
all_results = data["per_split_results"]
aggregated = data["aggregated_metrics"]

# Extract values EXACTLY as they appear in JSON
splits = [r["split"] for r in all_results]
det_dac = [r["deterministic"]["mapping_metrics"]["dac_%"] for r in all_results]
learned_dac = [r["learned"]["mapping_metrics"]["dac_%"] for r in all_results]
det_precision = [r["deterministic"]["actionable_metrics"]["actionable_precision"] for r in all_results]
learned_precision = [r["learned"]["actionable_metrics"]["actionable_precision"] for r in all_results]
det_var_red = [r["deterministic"]["consistency_metrics"]["variance_reduction"] for r in all_results]
learned_var_red = [r["learned"]["consistency_metrics"]["variance_reduction"] for r in all_results]

print("\n[2] Extracted values:")
print(f"  Splits: {splits}")
print(f"  Det DAC: {det_dac}")
print(f"  Learned DAC: {learned_dac}")
print(f"  Det Precision: {[f'{p:.4f}' for p in det_precision]}")
print(f"  Learned Precision: {[f'{p:.4f}' for p in learned_precision]}")

# Aggregated values
det_dac_mean = aggregated["deterministic"]["dac_%"]["mean"]
det_dac_std = aggregated["deterministic"]["dac_%"]["std"]
learned_dac_mean = aggregated["learned"]["dac_%"]["mean"]
learned_dac_std = aggregated["learned"]["dac_%"]["std"]
det_prec_mean = aggregated["deterministic"]["actionable_precision"]["mean"]
det_prec_std = aggregated["deterministic"]["actionable_precision"]["std"]
learned_prec_mean = aggregated["learned"]["actionable_precision"]["mean"]
learned_prec_std = aggregated["learned"]["actionable_precision"]["std"]
det_var_mean = aggregated["deterministic"]["variance_reduction"]["mean"]
det_var_std = aggregated["deterministic"]["variance_reduction"]["std"]
learned_var_mean = aggregated["learned"]["variance_reduction"]["mean"]
learned_var_std = aggregated["learned"]["variance_reduction"]["std"]

print("\n[3] Aggregated values:")
print(f"  DAC det: {det_dac_mean:.2f}% ± {det_dac_std:.2f}%")
print(f"  DAC learned: {learned_dac_mean:.2f}% ± {learned_dac_std:.2f}%")
print(f"  Precision det: {det_prec_mean:.4f} ± {det_prec_std:.4f}")
print(f"  Precision learned: {learned_prec_mean:.4f} ± {learned_prec_std:.4f}")

# Delete old plots
print("\n[4] Deleting old plots...")
for f in plots_dir.glob("*.png"):
    f.unlink()
    print(f"  Deleted: {f.name}")

# Plot 1: DAC per split
print("\n[5] Creating DAC per split plot...")
print(f"  DEBUG: det_dac = {det_dac}")
print(f"  DEBUG: learned_dac = {learned_dac}")
x = np.arange(len(splits))
width = 0.35
fig, ax = plt.subplots(figsize=(12, 6))
# Plot deterministic FIRST (green bars on left)
bars1 = ax.bar(x - width/2, det_dac, width, label="Deterministic", color="#2e7d32", alpha=0.8, edgecolor='black', linewidth=1)
# Plot learned SECOND (blue bars on right)
bars2 = ax.bar(x + width/2, learned_dac, width, label="Learned", color="#1976d2", alpha=0.8, edgecolor='black', linewidth=1)
# Value labels - deterministic (green)
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen')
# Value labels - learned (blue)
for i, bar in enumerate(bars2):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkblue')
ax.set_xlabel("Split", fontsize=12)
ax.set_ylabel("DAC (%)", fontsize=12)
ax.set_title("Defense-Attack Consistency (DAC) per Split\n(H3: Agreement with Deterministic Mapping)", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(splits, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(plots_dir / "dac_per_split.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: dac_per_split.png")

# Plot 2: Precision per split
print("\n[6] Creating precision per split plot...")
print(f"  DEBUG: det_precision = {det_precision}")
print(f"  DEBUG: learned_precision = {learned_precision}")
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, det_precision, width, label="Deterministic", color="#2e7d32", alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, learned_precision, width, label="Learned", color="#1976d2", alpha=0.8, edgecolor='black', linewidth=1)
# Value labels - deterministic
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen')
# Value labels - learned
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkblue')
ax.set_xlabel("Split", fontsize=12)
ax.set_ylabel("Actionable Precision", fontsize=12)
ax.set_title("Actionable Precision per Split", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(splits, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(plots_dir / "precision_per_split.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: precision_per_split.png")

# Plot 3: Variance reduction per split
print("\n[7] Creating variance reduction per split plot...")
print(f"  DEBUG: det_var_red = {det_var_red}")
print(f"  DEBUG: learned_var_red = {learned_var_red}")
fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, det_var_red, width, label="Deterministic", color="#2e7d32", alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, learned_var_red, width, label="Learned", color="#1976d2", alpha=0.8, edgecolor='black', linewidth=1)
# Value labels - deterministic
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.6f}', ha='center', va='bottom', fontsize=9, color='darkgreen')
# Value labels - learned
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.6f}', ha='center', va='bottom', fontsize=9, color='darkblue')
ax.set_xlabel("Split", fontsize=12)
ax.set_ylabel("Variance Reduction", fontsize=12)
ax.set_title("Variance Reduction per Split", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(splits, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(plots_dir / "variance_reduction_per_split.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: variance_reduction_per_split.png")

# Plot 4: Summary metrics
print("\n[8] Creating summary metrics plot...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metrics = ["DAC (%)", "Actionable Precision", "Variance Reduction"]
det_means = [det_dac_mean, det_prec_mean, det_var_mean]
det_stds = [det_dac_std, det_prec_std, det_var_std]
learned_means = [learned_dac_mean, learned_prec_mean, learned_var_mean]
learned_stds = [learned_dac_std, learned_prec_std, learned_var_std]

for ax, metric, det_mean, det_std, learned_mean, learned_std in zip(
    axes, metrics, det_means, det_stds, learned_means, learned_stds
):
    print(f"  DEBUG {metric}: det={det_mean:.4f}, learned={learned_mean:.4f}")
    bars1 = ax.bar(0 - width/2, det_mean, width, yerr=det_std, label="Deterministic", 
           color="#2e7d32", alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
    bars2 = ax.bar(0 + width/2, learned_mean, width, yerr=learned_std, label="Learned", 
           color="#1976d2", alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
    # Value labels
    for bar in bars1:
        height = bar.get_height()
        label_y = height + det_std + (0.05 * height if height > 0 else 0.05)
        if metric == "DAC (%)":
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        elif metric == "Actionable Precision":
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:.6f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        label_y = height + learned_std + (0.05 * height if height > 0 else 0.05)
        if metric == "DAC (%)":
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        elif metric == "Actionable Precision":
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                    f'{height:.6f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xticks([0])
    ax.set_xticklabels([""])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(plots_dir / "summary_metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: summary_metrics.png")

print("\n" + "=" * 80)
print("✓ ALL PLOTS REGENERATED SUCCESSFULLY")
print("=" * 80)
print(f"\nPlots saved to: {plots_dir}")
print("\nGenerated files:")
for f in sorted(plots_dir.glob("*.png")):
    size = f.stat().st_size
    print(f"  - {f.name} ({size:,} bytes)")
print("\nAll plots have VALUE LABELS showing the exact values from JSON.")
print("=" * 80)





