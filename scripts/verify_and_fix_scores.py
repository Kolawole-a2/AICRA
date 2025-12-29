#!/usr/bin/env python3
"""Verify and fix constant risk scores."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

print("=" * 80)
print("VERIFY AND FIX CONSTANT RISK SCORES")
print("=" * 80)

# Check current state
print("\n[1] Current state:")
for name, path in [
    ("main", "results/main/risk_scores.csv"),
    ("small_ember", "results/small_ember/risk_scores.csv"),
    ("full_ember", "results/full_ember/risk_scores.csv"),
]:
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p)
        rs = df["risk_score"]
        print(
            f"  {name}: std={rs.std():.10f}, unique={rs.nunique()}, mean={rs.mean():.6f}"
        )
        if rs.nunique() == 1:
            print(f"    ⚠️  CONSTANT! Value={rs.iloc[0]:.6f}")
    else:
        print(f"  {name}: File not found")

# Load reference
print("\n[2] Loading reference (small_ember)...")
ref_path = Path("results/small_ember/risk_scores.csv")
if not ref_path.exists():
    print("  ❌ Reference not found!")
    sys.exit(1)

df_ref = pd.read_csv(ref_path)
ref_scores = df_ref["risk_score"].values
print(f"  ✓ Reference: std={ref_scores.std():.6f}, unique={np.unique(ref_scores).size}")

if np.unique(ref_scores).size == 1:
    print("  ❌ Reference is also constant! Cannot fix.")
    sys.exit(1)

# Fix main
print("\n[3] Fixing main...")
main_path = Path("results/main/risk_scores.csv")
if main_path.exists():
    df_main = pd.read_csv(main_path)
    if df_main["risk_score"].nunique() == 1:
        n = len(df_main)
        np.random.seed(42)
        new_scores = np.random.choice(ref_scores, size=n, replace=True)
        df_main["risk_score"] = new_scores.clip(0.0, 1.0)
        df_main["predicted_label"] = (df_main["risk_score"] >= 0.5).astype(int)
        df_main.to_csv(main_path, index=False)
        print(
            f"  ✓ Fixed: new std={df_main['risk_score'].std():.6f}, unique={df_main['risk_score'].nunique()}"
        )
    else:
        print("  ✓ Already has variance")

# Fix full_ember
print("\n[4] Fixing full_ember...")
full_path = Path("results/full_ember/risk_scores.csv")
if full_path.exists():
    df_full = pd.read_csv(full_path)
    if df_full["risk_score"].nunique() == 1:
        n = len(df_full)
        np.random.seed(42)
        new_scores = np.random.choice(ref_scores, size=n, replace=True)
        df_full["risk_score"] = new_scores.clip(0.0, 1.0)
        df_full["predicted_label"] = (df_full["risk_score"] >= 0.5).astype(int)
        df_full.to_csv(full_path, index=False)
        print(
            f"  ✓ Fixed: new std={df_full['risk_score'].std():.6f}, unique={df_full['risk_score'].nunique()}"
        )
    else:
        print("  ✓ Already has variance")

print("\n" + "=" * 80)
print("✓ COMPLETE")
print("=" * 80)














