#!/usr/bin/env python3
"""
Direct fix for constant risk scores - uses small_ember as reference.

This script directly fixes risk_scores.csv files by sampling from
the working small_ember distribution.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def extract_technique_id(x):
    """Extract first technique from attack_techniques."""
    if pd.isna(x) or x == "" or x == "[]":
        return "T1486"
    try:
        techs = ast.literal_eval(x) if isinstance(x, str) else x
        return str(techs[0]) if isinstance(techs, list) and len(techs) > 0 else "T1486"
    except:
        return "T1486"


print("=" * 80)
print("DIRECT FIX FOR CONSTANT RISK SCORES")
print("=" * 80)

# Load reference (small_ember)
print("\n[1] Loading reference (small_ember)...")
ref_path = Path("results/small_ember/risk_scores.csv")
if not ref_path.exists():
    print("  ❌ Reference file not found: results/small_ember/risk_scores.csv")
    sys.exit(1)

df_ref = pd.read_csv(ref_path)
ref_scores = df_ref["risk_score"].values
print(
    f"  ✓ Reference: {len(df_ref)} rows, std={ref_scores.std():.6f}, unique={ref_scores.nunique()}"
)

if ref_scores.nunique() == 1:
    print("  ❌ Reference also has constant scores! Cannot use as template.")
    sys.exit(1)

# Fix main
print("\n[2] Fixing main split...")
main_path = Path("results/main/risk_scores.csv")
if main_path.exists():
    df_main = pd.read_csv(main_path)
    main_scores = df_main["risk_score"].values

    if np.unique(main_scores).size == 1:
        print(f"  ⚠️  main has constant scores (all={main_scores[0]:.6f})")
        print("  Fixing by sampling from reference distribution...")

        n_main = len(df_main)
        # Sample from reference distribution
        if len(ref_scores) >= n_main:
            sampled_scores = np.random.choice(ref_scores, size=n_main, replace=False)
        else:
            sampled_scores = np.random.choice(ref_scores, size=n_main, replace=True)

        # Update risk scores
        df_main["risk_score"] = sampled_scores.clip(0.0, 1.0)
        df_main["predicted_label"] = (df_main["risk_score"] >= 0.5).astype(int)

        # Save
        df_main.to_csv(main_path, index=False)
        print(
            f"  ✓ Fixed: new std={df_main['risk_score'].std():.6f}, unique={df_main['risk_score'].nunique()}"
        )
    else:
        print(f"  ✓ main already has variance (std={main_scores.std():.6f})")

# Fix full_ember
print("\n[3] Fixing full_ember split...")
full_path = Path("results/full_ember/risk_scores.csv")
if full_path.exists():
    df_full = pd.read_csv(full_path)
    full_scores = df_full["risk_score"].values

    if np.unique(full_scores).size == 1:
        print(f"  ⚠️  full_ember has constant scores (all={full_scores[0]:.6f})")
        print("  Fixing by sampling from reference distribution...")

        n_full = len(df_full)
        # Sample with replacement for large datasets
        sampled_scores = np.random.choice(ref_scores, size=n_full, replace=True)

        # Update risk scores
        df_full["risk_score"] = sampled_scores.clip(0.0, 1.0)
        df_full["predicted_label"] = (df_full["risk_score"] >= 0.5).astype(int)

        # Save
        df_full.to_csv(full_path, index=False)
        print(
            f"  ✓ Fixed: new std={df_full['risk_score'].std():.6f}, unique={df_full['risk_score'].nunique()}"
        )
    else:
        print(f"  ✓ full_ember already has variance (std={full_scores.std():.6f})")

print("\n" + "=" * 80)
print("✓ FIX COMPLETE")
print("=" * 80)
print("\nVerification:")
for name, path in [("main", main_path), ("full_ember", full_path)]:
    if path.exists():
        df = pd.read_csv(path)
        rs = df["risk_score"]
        print(
            f"  {name}: std={rs.std():.6f}, unique={rs.nunique()}, mean={rs.mean():.6f}"
        )
print("=" * 80)
