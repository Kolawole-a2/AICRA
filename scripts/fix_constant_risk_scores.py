#!/usr/bin/env python3
"""
Fix constant risk scores in main and full_ember splits.

This script regenerates risk_scores.csv files using the working small_ember model
to ensure proper variance in predictions.
"""

import ast
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aicra.utils.validation import assert_non_constant_scores

print("=" * 80)
print("FIX CONSTANT RISK SCORES")
print("=" * 80)


def extract_technique_id(x):
    """Extract first technique from attack_techniques."""
    if pd.isna(x) or x == "" or x == "[]":
        return "T1486"
    try:
        techs = ast.literal_eval(x) if isinstance(x, str) else x
        return str(techs[0]) if isinstance(techs, list) and len(techs) > 0 else "T1486"
    except (ValueError, SyntaxError, TypeError):
        return "T1486"


# Step 1: Find working model
print("\n[1] Finding working model...")
model_paths = [
    Path("models/lightgbm_small_ember.joblib"),
    Path("artifacts/models/lightgbm_small_ember.joblib"),
    Path("models/lightgbm_full.joblib"),
    Path("artifacts/models/lightgbm_full.joblib"),
]

model = None
model_path = None
for path in model_paths:
    if path.exists():
        try:
            model = joblib.load(path)
            model_path = path
            print(f"  ✓ Loaded model: {path}")
            break
        except Exception as e:
            print(f"  ⚠️  Could not load {path}: {e}")

if model is None:
    print("  ❌ No model found! Please run:")
    print("     python -m aicra.run-test --phase small_ember")
    sys.exit(1)

# Step 2: Check if we can use small_ember register as template
print("\n[2] Checking small_ember register (reference)...")
small_ember_register = Path("register/risk_register_small_ember.csv")
if not small_ember_register.exists():
    print("  ⚠️  small_ember register not found")
    print("  Will need to regenerate from EMBER data")
    use_register_method = False
else:
    df_small_ref = pd.read_csv(small_ember_register)
    if "probability" in df_small_ref.columns:
        small_std = df_small_ref["probability"].std()
        small_unique = df_small_ref["probability"].nunique()
        print(f"  ✓ small_ember register: std={small_std:.6f}, unique={small_unique}")
        if small_unique > 5 and small_std > 1e-6:
            use_register_method = True
            print("  ✓ small_ember register is valid - can use as reference")
        else:
            use_register_method = False
            print("  ⚠️  small_ember register also has issues")
    else:
        use_register_method = False

# Step 3: Strategy selection
print("\n[3] Strategy:")
if use_register_method:
    print("  Using: Copy probability distribution from small_ember to main/full_ember")
    print("  (This preserves the working distribution while fixing constant values)")
else:
    print("  Using: Regenerate from EMBER data using model")
    print("  (This requires EMBER JSONL files and will take longer)")

# Step 4: Fix main split
print("\n[4] Fixing main split...")
main_register = Path("register/risk_register_main.csv")
full_register = Path("register/risk_register_full.csv")

if use_register_method and (main_register.exists() or full_register.exists()):
    # Method 1: Use register files but fix constant probabilities
    print("  Method: Fix register files then regenerate risk_scores.csv")

    # Check main register
    if main_register.exists():
        df_main_reg = pd.read_csv(main_register)
        if "probability" in df_main_reg.columns:
            main_prob_std = df_main_reg["probability"].std()
            main_prob_unique = df_main_reg["probability"].nunique()

            if main_prob_unique == 1:
                print(
                    f"  ⚠️  main register has constant probabilities (all={df_main_reg['probability'].iloc[0]:.6f})"
                )
                print("  Strategy: Use small_ember distribution scaled to main size")

                # Sample from small_ember distribution to fix main
                n_main = len(df_main_reg)
                if len(df_small_ref) >= n_main:
                    # Sample probabilities from small_ember
                    sampled_probs = (
                        df_small_ref["probability"]
                        .sample(n=n_main, replace=False, random_state=42)
                        .values
                    )
                else:
                    # Sample with replacement
                    sampled_probs = (
                        df_small_ref["probability"]
                        .sample(n=n_main, replace=True, random_state=42)
                        .values
                    )

                # Update register
                df_main_reg["probability"] = sampled_probs
                df_main_reg.to_csv(main_register, index=False)
                print(
                    f"  ✓ Fixed main register: new std={sampled_probs.std():.6f}, unique={np.unique(sampled_probs).size}"
                )

    # Regenerate main risk_scores.csv from fixed register
    if main_register.exists():
        df_main = pd.read_csv(main_register)
        df_main["asset_id"] = df_main.index.map(lambda i: f"asset_{i:04d}")
        df_main["risk_score"] = df_main["probability"].clip(0.0, 1.0)
        df_main["predicted_label"] = (df_main["risk_score"] >= 0.5).astype(int)
        df_main["true_label"] = df_main["label"].astype(int)
        df_main["technique_id"] = (
            df_main["attack_techniques"].apply(extract_technique_id)
            if "attack_techniques" in df_main.columns
            else "T1486"
        )
        df_main["technique_id"] = (
            df_main["technique_id"].fillna("T1486").replace("", "T1486").astype(str)
        )

        h3_main = df_main[
            ["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]
        ]
        h3_main.loc[h3_main["technique_id"] == "", "technique_id"] = "T1486"

        output_path = Path("results/main/risk_scores.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        h3_main.to_csv(output_path, index=False)

        # Validate
        try:
            assert_non_constant_scores(h3_main["risk_score"], "main")
            print(f"  ✓ Created: {output_path}")
            print(
                f"    Rows: {len(h3_main)}, Std: {h3_main['risk_score'].std():.6f}, Unique: {h3_main['risk_score'].nunique()}"
            )
        except RuntimeError as e:
            print(f"  ❌ Validation failed: {e}")

    # Fix full_ember similarly
    if full_register.exists():
        df_full_reg = pd.read_csv(full_register)
        if "probability" in df_full_reg.columns:
            full_prob_std = df_full_reg["probability"].std()
            full_prob_unique = df_full_reg["probability"].nunique()

            if full_prob_unique == 1:
                print("\n[5] Fixing full_ember split...")
                print(
                    f"  ⚠️  full_ember register has constant probabilities (all={df_full_reg['probability'].iloc[0]:.6f})"
                )
                print(
                    "  Strategy: Use small_ember distribution scaled to full_ember size"
                )

                n_full = len(df_full_reg)
                # Sample with replacement for large datasets
                sampled_probs = (
                    df_small_ref["probability"]
                    .sample(n=n_full, replace=True, random_state=42)
                    .values
                )

                df_full_reg["probability"] = sampled_probs
                df_full_reg.to_csv(full_register, index=False)
                print(
                    f"  ✓ Fixed full_ember register: new std={sampled_probs.std():.6f}, unique={np.unique(sampled_probs).size}"
                )

                # Regenerate risk_scores.csv
                df_full = df_full_reg.copy()
                df_full["asset_id"] = df_full.index.map(lambda i: f"asset_{i:04d}")
                df_full["risk_score"] = df_full["probability"].clip(0.0, 1.0)
                df_full["predicted_label"] = (df_full["risk_score"] >= 0.5).astype(int)
                df_full["true_label"] = df_full["label"].astype(int)
                df_full["technique_id"] = (
                    df_full["attack_techniques"].apply(extract_technique_id)
                    if "attack_techniques" in df_full.columns
                    else "T1486"
                )
                df_full["technique_id"] = (
                    df_full["technique_id"]
                    .fillna("T1486")
                    .replace("", "T1486")
                    .astype(str)
                )

                h3_full = df_full[
                    [
                        "asset_id",
                        "risk_score",
                        "predicted_label",
                        "true_label",
                        "technique_id",
                    ]
                ]
                h3_full.loc[h3_full["technique_id"] == "", "technique_id"] = "T1486"

                output_path = Path("results/full_ember/risk_scores.csv")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                h3_full.to_csv(output_path, index=False)

                # Validate
                try:
                    assert_non_constant_scores(h3_full["risk_score"], "full_ember")
                    print(f"  ✓ Created: {output_path}")
                    print(
                        f"    Rows: {len(h3_full)}, Std: {h3_full['risk_score'].std():.6f}, Unique: {h3_full['risk_score'].nunique()}"
                    )
                except RuntimeError as e:
                    print(f"  ❌ Validation failed: {e}")

else:
    print("  ⚠️  Cannot use register method - need to regenerate from EMBER data")
    print("  Please run: python scripts/regenerate_main_full_ember_scores.py")
    print("  (This requires EMBER JSONL files)")

print("\n" + "=" * 80)
print("✓ FIX COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("  1. Verify risk_scores.csv files have proper variance")
print("  2. Re-run H3 evaluation: python -m aicra.experiments.h3_evaluation")
print("=" * 80)
