#!/usr/bin/env python3
"""Final regeneration of all H3 plots with full verification."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import os

json_path = Path("results/H3_full_evaluation/H3_full_results.json")
plots_dir = Path("results/H3_full_evaluation/plots")
plots_dir.mkdir(parents=True, exist_ok=True)
log_file = Path("plot_regeneration_log.txt")

with open(log_file, "w", encoding="utf-8") as log:
    log.write("=" * 80 + "\n")
    log.write("PLOT REGENERATION LOG\n")
    log.write("=" * 80 + "\n")
    log.write(f"Time: {datetime.datetime.now()}\n\n")
    
    # Load JSON
    log.write("[1] Loading JSON...\n")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    log.write(f"✓ Loaded: {json_path}\n\n")
    
    # Extract data
    all_results = data["per_split_results"]
    aggregated = data["aggregated_metrics"]
    splits = [r["split"] for r in all_results]
    det_dac = [float(r["deterministic"]["mapping_metrics"]["dac_%"]) for r in all_results]
    learned_dac = [float(r["learned"]["mapping_metrics"]["dac_%"]) for r in all_results]
    det_precision = [float(r["deterministic"]["actionable_metrics"]["actionable_precision"]) for r in all_results]
    learned_precision = [float(r["learned"]["actionable_metrics"]["actionable_precision"]) for r in all_results]
    det_var_red = [float(r["deterministic"]["consistency_metrics"]["variance_reduction"]) for r in all_results]
    learned_var_red = [float(r["learned"]["consistency_metrics"]["variance_reduction"]) for r in all_results]
    
    log.write("[2] Extracted values:\n")
    log.write(f"  Splits: {splits}\n")
    log.write(f"  Det DAC: {det_dac}\n")
    log.write(f"  Learned DAC: {learned_dac}\n")
    log.write(f"  Det Precision: {det_precision}\n")
    log.write(f"  Learned Precision: {learned_precision}\n\n")
    
    # Delete old plots
    log.write("[3] Deleting old plots...\n")
    for f in plots_dir.glob("*.png"):
        if f.exists():
            old_stat = f.stat()
            f.unlink()
            log.write(f"  Deleted: {f.name} (was {old_stat.st_size} bytes, modified {datetime.datetime.fromtimestamp(old_stat.st_mtime)})\n")
    log.write("\n")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    x = np.arange(len(splits))
    width = 0.4
    
    # Plot 1: DAC
    log.write("[4] Creating DAC plot...\n")
    fig, ax = plt.subplots(figsize=(16, 9))
    bars1 = ax.bar(x - width/2, det_dac, width, label="Deterministic (GREEN)", color="green", alpha=1.0, edgecolor='black', linewidth=3)
    bars2 = ax.bar(x + width/2, learned_dac, width, label="Learned (BLUE)", color="blue", alpha=1.0, edgecolor='black', linewidth=3)
    for bar, val in zip(bars1, det_dac):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3, f'DET:{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold', color='darkgreen', bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2))
    for bar, val in zip(bars2, learned_dac):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3, f'LRN:{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold', color='darkblue', bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', linewidth=2))
    ax.set_xlabel("Split", fontsize=16, fontweight='bold')
    ax.set_ylabel("DAC (%)", fontsize=16, fontweight='bold')
    ax.set_title(f"DAC Per Split - {timestamp}", fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.5, axis='y')
    ax.set_ylim([0, 115])
    plt.tight_layout()
    f1 = plots_dir / "dac_per_split.png"
    plt.savefig(f1, dpi=150, bbox_inches='tight')
    plt.close()
    stat1 = f1.stat()
    log.write(f"  ✓ Saved: {f1.name} ({stat1.st_size:,} bytes, modified {datetime.datetime.fromtimestamp(stat1.st_mtime)})\n")
    
    # Plot 2: Precision
    log.write("\n[5] Creating Precision plot...\n")
    fig, ax = plt.subplots(figsize=(16, 9))
    bars1 = ax.bar(x - width/2, det_precision, width, label="Deterministic (GREEN)", color="green", alpha=1.0, edgecolor='black', linewidth=3)
    bars2 = ax.bar(x + width/2, learned_precision, width, label="Learned (BLUE)", color="blue", alpha=1.0, edgecolor='black', linewidth=3)
    for bar, val in zip(bars1, det_precision):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, f'DET:{val:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold', color='darkgreen', bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', linewidth=2))
    for bar, val in zip(bars2, learned_precision):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, f'LRN:{val:.3f}', ha='center', va='bottom', fontsize=14, fontweight='bold', color='darkblue', bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', linewidth=2))
    ax.set_xlabel("Split", fontsize=16, fontweight='bold')
    ax.set_ylabel("Actionable Precision", fontsize=16, fontweight='bold')
    ax.set_title(f"Actionable Precision Per Split - {timestamp}", fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.5, axis='y')
    y_max = max(max(det_precision), max(learned_precision)) * 1.2 if max(max(det_precision), max(learned_precision)) > 0 else 1.2
    ax.set_ylim([0, y_max])
    plt.tight_layout()
    f2 = plots_dir / "precision_per_split.png"
    plt.savefig(f2, dpi=150, bbox_inches='tight')
    plt.close()
    stat2 = f2.stat()
    log.write(f"  ✓ Saved: {f2.name} ({stat2.st_size:,} bytes, modified {datetime.datetime.fromtimestamp(stat2.st_mtime)})\n")
    
    # Plot 3: Variance
    log.write("\n[6] Creating Variance Reduction plot...\n")
    fig, ax = plt.subplots(figsize=(16, 9))
    bars1 = ax.bar(x - width/2, det_var_red, width, label="Deterministic (GREEN)", color="green", alpha=1.0, edgecolor='black', linewidth=3)
    bars2 = ax.bar(x + width/2, learned_var_red, width, label="Learned (BLUE)", color="blue", alpha=1.0, edgecolor='black', linewidth=3)
    for bar, val in zip(bars1, det_var_red):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'DET:{val:.6f}', ha='center', va='bottom', fontsize=12, color='darkgreen')
    for bar, val in zip(bars2, learned_var_red):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'LRN:{val:.6f}', ha='center', va='bottom', fontsize=12, color='darkblue')
    ax.set_xlabel("Split", fontsize=16, fontweight='bold')
    ax.set_ylabel("Variance Reduction", fontsize=16, fontweight='bold')
    ax.set_title(f"Variance Reduction Per Split - {timestamp}", fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.5, axis='y')
    plt.tight_layout()
    f3 = plots_dir / "variance_reduction_per_split.png"
    plt.savefig(f3, dpi=150, bbox_inches='tight')
    plt.close()
    stat3 = f3.stat()
    log.write(f"  ✓ Saved: {f3.name} ({stat3.st_size:,} bytes, modified {datetime.datetime.fromtimestamp(stat3.st_mtime)})\n")
    
    # Plot 4: Summary
    log.write("\n[7] Creating Summary plot...\n")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    metrics = ["DAC (%)", "Actionable Precision", "Variance Reduction"]
    det_means = [aggregated["deterministic"]["dac_%"]["mean"], aggregated["deterministic"]["actionable_precision"]["mean"], aggregated["deterministic"]["variance_reduction"]["mean"]]
    det_stds = [aggregated["deterministic"]["dac_%"]["std"], aggregated["deterministic"]["actionable_precision"]["std"], aggregated["deterministic"]["variance_reduction"]["std"]]
    learned_means = [aggregated["learned"]["dac_%"]["mean"], aggregated["learned"]["actionable_precision"]["mean"], aggregated["learned"]["variance_reduction"]["mean"]]
    learned_stds = [aggregated["learned"]["dac_%"]["std"], aggregated["learned"]["actionable_precision"]["std"], aggregated["learned"]["variance_reduction"]["std"]]
    
    for ax, metric, det_mean, det_std, learned_mean, learned_std in zip(axes, metrics, det_means, det_stds, learned_means, learned_stds):
        bars1 = ax.bar(0 - width/2, det_mean, width, yerr=det_std, label="Deterministic", color="green", alpha=1.0, capsize=5, edgecolor='black', linewidth=2)
        bars2 = ax.bar(0 + width/2, learned_mean, width, yerr=learned_std, label="Learned", color="blue", alpha=1.0, capsize=5, edgecolor='black', linewidth=2)
        label_y1 = det_mean + det_std + 2
        label_y2 = learned_mean + learned_std + 2
        if metric == "DAC (%)":
            ax.text(0 - width/2, label_y1, f'{det_mean:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkgreen')
            ax.text(0 + width/2, label_y2, f'{learned_mean:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkblue')
        elif metric == "Actionable Precision":
            ax.text(0 - width/2, label_y1, f'{det_mean:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkgreen')
            ax.text(0 + width/2, label_y2, f'{learned_mean:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold', color='darkblue')
        else:
            ax.text(0 - width/2, label_y1, f'{det_mean:.6f}', ha='center', va='bottom', fontsize=10, color='darkgreen')
            ax.text(0 + width/2, label_y2, f'{learned_mean:.6f}', ha='center', va='bottom', fontsize=10, color='darkblue')
        ax.set_ylabel(metric, fontsize=13, fontweight='bold')
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_xticks([0])
        ax.set_xticklabels([""])
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    f4 = plots_dir / "summary_metrics.png"
    plt.savefig(f4, dpi=150, bbox_inches='tight')
    plt.close()
    stat4 = f4.stat()
    log.write(f"  ✓ Saved: {f4.name} ({stat4.st_size:,} bytes, modified {datetime.datetime.fromtimestamp(stat4.st_mtime)})\n")
    
    log.write("\n" + "=" * 80 + "\n")
    log.write("✓ ALL PLOTS REGENERATED\n")
    log.write("=" * 80 + "\n")
    log.write(f"\nAll plots should show:\n")
    log.write("  - GREEN bars (Deterministic) on LEFT\n")
    log.write("  - BLUE bars (Learned) on RIGHT\n")
    log.write("  - Value labels on each bar\n")
    log.write(f"\nCheck log file: {log_file}\n")

print(f"✓ Regeneration complete. Check log: {log_file}")




