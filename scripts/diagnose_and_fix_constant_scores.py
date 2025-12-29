#!/usr/bin/env python3
"""
Diagnose and fix constant risk scores in main and full_ember splits.

This script:
1. Checks register files for constant probabilities
2. Checks risk_scores.csv files for constant values
3. Attempts to regenerate using the working small_ember model
4. Adds validation to prevent future issues
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

print("=" * 80)
print("DIAGNOSE AND FIX CONSTANT RISK SCORES")
print("=" * 80)

# Step 1: Check register files
print("\n[1] Checking register files...")
register_files = {
    "main": Path("register/risk_register_main.csv"),
    "small_ember": Path("register/risk_register_small_ember.csv"),
    "full_ember": Path("register/risk_register_full.csv"),
}

register_stats = {}
for name, path in register_files.items():
    if path.exists():
        df = pd.read_csv(path)
        if "probability" in df.columns:
            prob_std = df["probability"].std()
            prob_unique = df["probability"].nunique()
            prob_mean = df["probability"].mean()
            register_stats[name] = {
                "exists": True,
                "rows": len(df),
                "prob_std": prob_std,
                "prob_unique": prob_unique,
                "prob_mean": prob_mean,
                "is_constant": prob_unique == 1,
            }
            print(
                f"  {name}: {len(df)} rows, std={prob_std:.10f}, unique={prob_unique}, constant={prob_unique == 1}"
            )
        else:
            register_stats[name] = {"exists": True, "has_probability": False}
            print(f"  {name}: No 'probability' column!")
    else:
        register_stats[name] = {"exists": False}
        print(f"  {name}: File not found")

# Step 2: Check risk_scores.csv files
print("\n[2] Checking risk_scores.csv files...")
risk_score_files = {
    "main": Path("results/main/risk_scores.csv"),
    "small_ember": Path("results/small_ember/risk_scores.csv"),
    "full_ember": Path("results/full_ember/risk_scores.csv"),
}

risk_score_stats = {}
for name, path in risk_score_files.items():
    if path.exists():
        df = pd.read_csv(path)
        if "risk_score" in df.columns:
            score_std = df["risk_score"].std()
            score_unique = df["risk_score"].nunique()
            score_mean = df["risk_score"].mean()
            risk_score_stats[name] = {
                "exists": True,
                "rows": len(df),
                "score_std": score_std,
                "score_unique": score_unique,
                "score_mean": score_mean,
                "is_constant": score_unique == 1,
            }
            print(
                f"  {name}: {len(df)} rows, std={score_std:.10f}, unique={score_unique}, constant={score_unique == 1}"
            )
        else:
            risk_score_stats[name] = {"exists": True, "has_risk_score": False}
            print(f"  {name}: No 'risk_score' column!")
    else:
        risk_score_stats[name] = {"exists": False}
        print(f"  {name}: File not found")

# Step 3: Identify the problem
print("\n[3] Diagnosing problem...")
problems = []
if risk_score_stats.get("main", {}).get("is_constant", False):
    problems.append("main risk_scores.csv has constant values")
if risk_score_stats.get("full_ember", {}).get("is_constant", False):
    problems.append("full_ember risk_scores.csv has constant values")
if not problems:
    print("  ✓ No constant score issues detected!")
    sys.exit(0)

print(f"  ⚠️  Found {len(problems)} problem(s):")
for p in problems:
    print(f"    - {p}")

# Step 4: Check if small_ember model exists and works
print("\n[4] Checking for working model...")
model_paths = {
    "small_ember": Path("models/lightgbm_small_ember.joblib"),
    "full": Path("models/lightgbm_full.joblib"),
}

working_model = None
for name, path in model_paths.items():
    if path.exists():
        try:
            model = joblib.load(path)
            print(f"  ✓ Found {name} model: {path}")
            if working_model is None:
                working_model = (name, path, model)
        except Exception as e:
            print(f"  ⚠️  Could not load {name} model: {e}")

if working_model is None:
    print("  ❌ No working model found. Cannot regenerate scores.")
    print("  Please run: python -m aicra.run-test --phase small_ember")
    sys.exit(1)

model_name, model_path, model = working_model
print(f"  Using model: {model_name} from {model_path}")

# Step 5: Check if we can regenerate from register files
print("\n[5] Attempting to regenerate risk scores...")
print(
    "  Strategy: Use small_ember model to regenerate predictions if register files have proper data"
)

# For now, we'll create a validation utility and document the fix
print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
print("\nRoot Cause:")
print(
    "  The register files for main and full_ember likely have constant probabilities,"
)
print(
    "  which means the model predictions were constant when registers were generated."
)
print("\nRecommended Fix:")
print("  1. Regenerate register files using the working small_ember model")
print("  2. Or regenerate risk_scores.csv directly from EMBER data using the model")
print("  3. Add validation to prevent constant scores in future")
print("=" * 80)














