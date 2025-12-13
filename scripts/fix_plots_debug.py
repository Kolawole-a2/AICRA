#!/usr/bin/env python3
"""Fix plots with extensive debugging."""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

json_path = Path("results/H3_full_evaluation/H3_full_results.json")
plots_dir = Path("results/H3_full_evaluation/plots")
plots_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FIXING PLOTS WITH DEBUG")
print("=" * 80)

# Load JSON
with open(json_path, encoding="utf-8") as f:
    data = json.load(f)

all_results = data["per_split_results"]
aggregated = data["aggregated_metrics"]

# Extract values
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

print("\nExtracted values:")
print(f"  Splits: {splits}")
print(f"  Det DAC: {det_dac} (type: {[type(x) for x in det_dac]})")
print(f"  Learned DAC: {learned_dac} (type: {[type(x) for x in learned_dac]})")
print(f"  Det Precision: {det_precision}")
print(f"  Learned Precision: {learned_precision}")

# Delete old plots
for f in plots_dir.glob("*.png"):
    f.unlink()

# Plot 1: DAC per split - FIXED
print("\nCreating DAC plot...")
x = np.arange(len(splits))
width = 0.35
fig, ax = plt.subplots(figsize=(14, 7))

# Convert to numpy arrays to ensure proper types
det_dac_arr = np.array(det_dac)
learned_dac_arr = np.array(learned_dac)

print(f"  Plotting det_dac: {det_dac_arr}")
print(f"  Plotting learned_dac: {learned_dac_arr}")

# Plot deterministic (GREEN, LEFT)
bars1 = ax.bar(
    x - width / 2,
    det_dac_arr,
    width,
    label="Deterministic",
    color="#2e7d32",
    alpha=0.9,
    edgecolor="black",
    linewidth=1.5,
)

# Plot learned (BLUE, RIGHT)
bars2 = ax.bar(
    x + width / 2,
    learned_dac_arr,
    width,
    label="Learned",
    color="#1976d2",
    alpha=0.9,
    edgecolor="black",
    linewidth=1.5,
)

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, det_dac_arr, strict=False)):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="darkgreen",
    )

for i, (bar, val) in enumerate(zip(bars2, learned_dac_arr, strict=False)):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="darkblue",
    )

ax.set_xlabel("Split", fontsize=13, fontweight="bold")
ax.set_ylabel("DAC (%)", fontsize=13, fontweight="bold")
ax.set_title(
    "Defense-Attack Consistency (DAC) per Split\n(H3: Agreement with Deterministic Mapping)",
    fontsize=15,
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(splits, rotation=45, ha="right", fontsize=11)
ax.legend(fontsize=12, loc="upper right")
ax.grid(alpha=0.3, axis="y")
ax.set_ylim([0, max(max(det_dac_arr), max(learned_dac_arr)) * 1.2])

plt.tight_layout()
plt.savefig(plots_dir / "dac_per_split.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved dac_per_split.png")

# Plot 2: Precision per split
print("\nCreating Precision plot...")
fig, ax = plt.subplots(figsize=(14, 7))
det_prec_arr = np.array(det_precision)
learned_prec_arr = np.array(learned_precision)

bars1 = ax.bar(
    x - width / 2,
    det_prec_arr,
    width,
    label="Deterministic",
    color="#2e7d32",
    alpha=0.9,
    edgecolor="black",
    linewidth=1.5,
)
bars2 = ax.bar(
    x + width / 2,
    learned_prec_arr,
    width,
    label="Learned",
    color="#1976d2",
    alpha=0.9,
    edgecolor="black",
    linewidth=1.5,
)

for bar, val in zip(bars1, det_prec_arr, strict=False):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="darkgreen",
    )

for bar, val in zip(bars2, learned_prec_arr, strict=False):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.02,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="darkblue",
    )

ax.set_xlabel("Split", fontsize=13, fontweight="bold")
ax.set_ylabel("Actionable Precision", fontsize=13, fontweight="bold")
ax.set_title("Actionable Precision per Split", fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(splits, rotation=45, ha="right", fontsize=11)
ax.legend(fontsize=12)
ax.grid(alpha=0.3, axis="y")
ax.set_ylim(
    [
        0,
        max(max(det_prec_arr), max(learned_prec_arr)) * 1.2
        if max(max(det_prec_arr), max(learned_prec_arr)) > 0
        else 1.2,
    ]
)

plt.tight_layout()
plt.savefig(plots_dir / "precision_per_split.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved precision_per_split.png")

# Plot 3: Variance reduction
print("\nCreating Variance Reduction plot...")
fig, ax = plt.subplots(figsize=(14, 7))
det_var_arr = np.array(det_var_red)
learned_var_arr = np.array(learned_var_red)

bars1 = ax.bar(
    x - width / 2,
    det_var_arr,
    width,
    label="Deterministic",
    color="#2e7d32",
    alpha=0.9,
    edgecolor="black",
    linewidth=1.5,
)
bars2 = ax.bar(
    x + width / 2,
    learned_var_arr,
    width,
    label="Learned",
    color="#1976d2",
    alpha=0.9,
    edgecolor="black",
    linewidth=1.5,
)

for bar, val in zip(bars1, det_var_arr, strict=False):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{val:.6f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="darkgreen",
    )

for bar, val in zip(bars2, learned_var_arr, strict=False):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{val:.6f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="darkblue",
    )

ax.set_xlabel("Split", fontsize=13, fontweight="bold")
ax.set_ylabel("Variance Reduction", fontsize=13, fontweight="bold")
ax.set_title("Variance Reduction per Split", fontsize=15, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(splits, rotation=45, ha="right", fontsize=11)
ax.legend(fontsize=12)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(
    plots_dir / "variance_reduction_per_split.png", dpi=150, bbox_inches="tight"
)
plt.close()
print("  ✓ Saved variance_reduction_per_split.png")

# Plot 4: Summary
print("\nCreating Summary plot...")
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
metrics = ["DAC (%)", "Actionable Precision", "Variance Reduction"]
det_means = [
    float(aggregated["deterministic"]["dac_%"]["mean"]),
    float(aggregated["deterministic"]["actionable_precision"]["mean"]),
    float(aggregated["deterministic"]["variance_reduction"]["mean"]),
]
det_stds = [
    float(aggregated["deterministic"]["dac_%"]["std"]),
    float(aggregated["deterministic"]["actionable_precision"]["std"]),
    float(aggregated["deterministic"]["variance_reduction"]["std"]),
]
learned_means = [
    float(aggregated["learned"]["dac_%"]["mean"]),
    float(aggregated["learned"]["actionable_precision"]["mean"]),
    float(aggregated["learned"]["variance_reduction"]["mean"]),
]
learned_stds = [
    float(aggregated["learned"]["dac_%"]["std"]),
    float(aggregated["learned"]["actionable_precision"]["std"]),
    float(aggregated["learned"]["variance_reduction"]["std"]),
]

for ax, metric, det_mean, det_std, learned_mean, learned_std in zip(
    axes, metrics, det_means, det_stds, learned_means, learned_stds, strict=False
):
    bars1 = ax.bar(
        0 - width / 2,
        det_mean,
        width,
        yerr=det_std,
        label="Deterministic",
        color="#2e7d32",
        alpha=0.9,
        capsize=5,
        edgecolor="black",
        linewidth=1.5,
    )
    bars2 = ax.bar(
        0 + width / 2,
        learned_mean,
        width,
        yerr=learned_std,
        label="Learned",
        color="#1976d2",
        alpha=0.9,
        capsize=5,
        edgecolor="black",
        linewidth=1.5,
    )

    # Value labels
    label_y1 = det_mean + det_std + (0.05 * det_mean if det_mean > 0 else 0.05)
    label_y2 = (
        learned_mean + learned_std + (0.05 * learned_mean if learned_mean > 0 else 0.05)
    )

    if metric == "DAC (%)":
        ax.text(
            0 - width / 2,
            label_y1,
            f"{det_mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="darkgreen",
        )
        ax.text(
            0 + width / 2,
            label_y2,
            f"{learned_mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="darkblue",
        )
    elif metric == "Actionable Precision":
        ax.text(
            0 - width / 2,
            label_y1,
            f"{det_mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="darkgreen",
        )
        ax.text(
            0 + width / 2,
            label_y2,
            f"{learned_mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="darkblue",
        )
    else:
        ax.text(
            0 - width / 2,
            label_y1,
            f"{det_mean:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="darkgreen",
        )
        ax.text(
            0 + width / 2,
            label_y2,
            f"{learned_mean:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="darkblue",
        )

    ax.set_ylabel(metric, fontsize=12, fontweight="bold")
    ax.set_title(metric, fontsize=13, fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels([""])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(plots_dir / "summary_metrics.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved summary_metrics.png")

print("\n" + "=" * 80)
print("✓ ALL PLOTS REGENERATED")
print("=" * 80)
print(f"\nPlots in: {plots_dir}")
for f in sorted(plots_dir.glob("*.png")):
    print(f"  - {f.name}")
