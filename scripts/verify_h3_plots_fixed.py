#!/usr/bin/env python3
"""Verify that H3 plots have been fixed and show different values."""

import json
from pathlib import Path

repo_root = Path(__file__).parent.parent
results_path = repo_root / "results/H3_full_evaluation/H3_full_results.json"

if not results_path.exists():
    print(f"❌ Results file not found: {results_path}")
    print("   Run H3 evaluation first:")
    print("   python -m aicra.experiments.h3_evaluation --output results/H3_full_evaluation")
    exit(1)

with open(results_path) as f:
    results = json.load(f)

print("="*80)
print("H3 PLOTS FIX VERIFICATION")
print("="*80)

# Check actionable precision
ap = results.get("actionable_precision", {})
det_prec = ap.get("deterministic", {}).get("precision", 0)
lrn_prec = ap.get("learned", {}).get("precision", 0)
delta_prec = ap.get("delta_precision", 0)

print(f"\nActionable Precision:")
print(f"  Deterministic: {det_prec:.4f}")
print(f"  Learned: {lrn_prec:.4f}")
print(f"  Delta: {delta_prec:.4f}")

if abs(delta_prec) < 0.0001:
    print("  ❌ IDENTICAL - Plots will show same values")
else:
    print("  ✓ DIFFERENT - Plots will show different values")

# Check variance reduction
vc = results.get("variance_consistency", {})
det_var = vc.get("deterministic", {}).get("variance_reduction", 0)
lrn_var = vc.get("learned", {}).get("variance_reduction", 0)
delta_var = vc.get("delta_variance_reduction", 0)

print(f"\nVariance Reduction:")
print(f"  Deterministic: {det_var:.6f}")
print(f"  Learned: {lrn_var:.6f}")
print(f"  Delta: {delta_var:.6f}")

if abs(delta_var) < 0.000001:
    print("  ❌ IDENTICAL - Plots will show same values")
else:
    print("  ✓ DIFFERENT - Plots will show different values")

# Check mapping metrics
det_mapping = results.get("deterministic_mapping", {})
lrn_mapping = results.get("learned_mapping", {})

det_dac = det_mapping.get("consistency_%", 0)
lrn_dac = lrn_mapping.get("consistency_%", 0)
delta_dac = det_dac - lrn_dac

print(f"\nDAC (Defense-Attack Consistency):")
print(f"  Deterministic: {det_dac:.2f}%")
print(f"  Learned: {lrn_dac:.2f}%")
print(f"  Delta: {delta_dac:.2f}%")

if abs(delta_dac) < 0.01:
    print("  ❌ IDENTICAL - Plots will show same values")
else:
    print("  ✓ DIFFERENT - Plots will show different values")

# Overall status
print("\n" + "="*80)
if abs(delta_prec) < 0.0001 and abs(delta_var) < 0.000001 and abs(delta_dac) < 0.01:
    print("❌ ALL METRICS ARE IDENTICAL")
    print("   Plots will show overlapping/identical bars")
    print("\n   SOLUTION:")
    print("   1. Run: python scripts/final_fix_h3_mappings.py")
    print("   2. Then: python -m aicra.experiments.h3_evaluation --output results/H3_full_evaluation")
    exit(1)
else:
    print("✓ METRICS ARE DIFFERENT")
    print("   Plots should show distinct values for deterministic vs learned")
    print("\n   Check plots in: results/H3_full_evaluation/plots/")
    exit(0)
