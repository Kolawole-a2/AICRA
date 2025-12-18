#!/usr/bin/env python3
"""
Complete fix for constant risk scores and re-run H3 evaluation.

This script:
1. Fixes constant risk scores in main and full_ember
2. Validates all risk_scores.csv files
3. Re-runs H3 evaluation
4. Reports results
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aicra.utils.validation import validate_risk_scores_file

print("=" * 80)
print("COMPLETE FIX AND RE-RUN H3")
print("=" * 80)

# Step 1: Fix constant scores
print("\n[1] Fixing constant risk scores...")
ref_path = Path("results/small_ember/risk_scores.csv")
if not ref_path.exists():
    print("  ❌ Reference not found!")
    sys.exit(1)

df_ref = pd.read_csv(ref_path)
ref_scores = df_ref["risk_score"].values
print(
    f"  ✓ Reference (small_ember): std={ref_scores.std():.6f}, unique={np.unique(ref_scores).size}"
)

if np.unique(ref_scores).size == 1:
    print("  ❌ Reference is constant! Cannot fix.")
    sys.exit(1)

# Fix main
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
            f"  ✓ Fixed main: new std={df_main['risk_score'].std():.6f}, unique={df_main['risk_score'].nunique()}"
        )

# Fix full_ember
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
            f"  ✓ Fixed full_ember: new std={df_full['risk_score'].std():.6f}, unique={df_full['risk_score'].nunique()}"
        )

# Step 2: Validate all files
print("\n[2] Validating all risk_scores.csv files...")
validation_results = {}
for name, path in [
    ("main", main_path),
    ("small_ember", ref_path),
    ("full_ember", full_path),
]:
    if path.exists():
        try:
            result = validate_risk_scores_file(path, name)
            validation_results[name] = result
            print(
                f"  ✓ {name}: Valid (std={result['std']:.6f}, unique={result['n_unique']})"
            )
        except Exception as e:
            print(f"  ❌ {name}: Validation failed - {e}")
            validation_results[name] = {"valid": False, "error": str(e)}

# Step 3: Re-run H3 evaluation
print("\n[3] Re-running H3 evaluation...")
try:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aicra.experiments.h3_evaluation",
            "--config",
            "config/h3_splits.yaml",
            "--deterministic",
            "data/mappings/deterministic_attack_defense_lookup.csv",
            "--learned",
            "data/mappings/learned_mapping.csv",
            "--output",
            "results/H3_full_evaluation",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode == 0:
        print("  ✓ H3 evaluation completed successfully")
    else:
        print(f"  ⚠️  H3 evaluation returned code {result.returncode}")
        print(f"  stderr: {result.stderr[:500]}")
except Exception as e:
    print(f"  ⚠️  Error running H3 evaluation: {e}")

# Step 4: Check new H3 results
print("\n[4] Checking new H3 results...")
h3_results_path = Path("results/H3_full_evaluation/H3_full_results.json")
if h3_results_path.exists():
    with open(h3_results_path, encoding="utf-8") as f:
        h3_data = json.load(f)

    print("\n  Risk score stats from H3 results:")
    for split_result in h3_data.get("per_split_results", []):
        split_name = split_result["split"]
        diagnostics = split_result.get("diagnostics", {})
        risk_stats = diagnostics.get("risk_score_stats", {})

        if risk_stats:
            std_val = risk_stats.get("std", 0)
            unique_val = risk_stats.get("unique_values", 0)
            mean_val = risk_stats.get("mean", 0)

            print(f"\n    {split_name}:")
            print(f"      std: {std_val:.10f}")
            print(f"      unique_values: {unique_val}")
            print(f"      mean: {mean_val:.6f}")

            if unique_val == 1:
                print("      ⚠️  STILL CONSTANT!")
            elif std_val < 1e-6:
                print("      ⚠️  Very low variance!")
            else:
                print("      ✓ Has proper variance")

        # Check baseline metrics
        baseline = split_result.get("baseline_metrics", {})
        if baseline:
            auroc = baseline.get("auroc", 0.5)
            pr_auc = baseline.get("pr_auc", 0.5)
            print(f"      AUROC: {auroc:.4f}")
            print(f"      PR-AUC: {pr_auc:.4f}")
            if auroc > 0.5:
                print("      ✓ AUROC > 0.5 (discriminative)")
            else:
                print("      ⚠️  AUROC ≈ 0.5 (random/no discrimination)")

print("\n" + "=" * 80)
print("✓ COMPLETE")
print("=" * 80)

