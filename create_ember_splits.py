#!/usr/bin/env python3
"""Create H3 splits from EMBER register files."""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from aicra.utils.validation import assert_non_constant_scores, validate_risk_scores_file

def extract_technique_id(x):
    """Extract first technique from attack_techniques."""
    if pd.isna(x) or x == '' or x == '[]':
        return 'T1486'  # Default technique for empty/missing
    try:
        techs = ast.literal_eval(x) if isinstance(x, str) else x
        return str(techs[0]) if isinstance(techs, list) and len(techs) > 0 else 'T1486'
    except:
        return 'T1486'  # Default technique on error

def validate_probabilities(df: pd.DataFrame, split_name: str, min_unique: int = 5, min_std: float = 1e-6) -> None:
    """
    Validate that probabilities are not constant.
    
    Raises RuntimeError if probabilities appear constant.
    """
    if "probability" not in df.columns:
        raise ValueError(f"[{split_name}] Missing 'probability' column in register file")
    
    probs = df["probability"].values
    unique_vals = np.unique(probs)
    std_val = np.std(probs)
    
    if len(unique_vals) < min_unique:
        raise RuntimeError(
            f"[{split_name}] Register probabilities appear constant: "
            f"{len(unique_vals)} unique values (minimum: {min_unique}). "
            f"Mean={np.mean(probs):.6f}, std={std_val:.10f}. "
            f"This indicates the model predictions were constant. "
            f"Check model training, feature processing, or calibration pipeline."
        )
    
    if std_val < min_std:
        raise RuntimeError(
            f"[{split_name}] Register probabilities have very low variance: "
            f"std={std_val:.10f} (minimum: {min_std}). "
            f"Mean={np.mean(probs):.6f}, unique={len(unique_vals)}. "
            f"This indicates the model predictions were nearly constant. "
            f"Check model training, feature processing, or calibration pipeline."
        )

# Main split (10,000 samples)
print("Processing main...")
if Path("register/risk_register_main.csv").exists():
    df_main = pd.read_csv("register/risk_register_main.csv")
elif Path("register/risk_register_full.csv").exists():
    # Create main split by sampling from full register
    print("  Creating main split by sampling 10,000 rows from risk_register_full.csv...")
    df_full = pd.read_csv("register/risk_register_full.csv")
    df_main = df_full.sample(n=10000, random_state=42).reset_index(drop=True)
    print(f"  Sampled {len(df_main)} rows")
else:
    df_main = None

if df_main is not None:
    # Validate probabilities before processing
    try:
        validate_probabilities(df_main, "main")
    except RuntimeError as e:
        print(f"  ERROR: VALIDATION FAILED: {e}")
        print(f"  WARNING: Cannot create main split with constant probabilities.")
        print(f"  Please regenerate register file with proper model predictions.")
        sys.exit(1)
    
    df_main["asset_id"] = df_main.index.map(lambda i: f"asset_{i:04d}")
    df_main["risk_score"] = df_main["probability"].clip(0.0, 1.0)
    df_main["predicted_label"] = (df_main["risk_score"] >= 0.5).astype(int)
    df_main["true_label"] = df_main["label"].astype(int)
    df_main["technique_id"] = df_main["attack_techniques"].apply(extract_technique_id)
    df_main["technique_id"] = df_main["technique_id"].fillna('T1486').replace('', 'T1486').astype(str)
    h3_main = df_main[["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]]
    # Final safety check - ensure no empty values
    h3_main.loc[h3_main['technique_id'] == '', 'technique_id'] = 'T1486'
    h3_main['technique_id'] = h3_main['technique_id'].fillna('T1486')
    
    # VALIDATION: Ensure risk scores are not constant before writing
    assert_non_constant_scores(h3_main["risk_score"], split_name="main", min_unique=5, min_std=1e-6)
    
    Path("results/main").mkdir(parents=True, exist_ok=True)
    h3_main.to_csv("results/main/risk_scores.csv", index=False)
    # Post-process to ensure no empty technique_ids
    h3_main_fixed = pd.read_csv("results/main/risk_scores.csv", keep_default_na=False)
    h3_main_fixed.loc[h3_main_fixed['technique_id'] == '', 'technique_id'] = 'T1486'
    h3_main_fixed['technique_id'] = h3_main_fixed['technique_id'].fillna('T1486')
    h3_main_fixed.to_csv("results/main/risk_scores.csv", index=False)
    
    # FINAL VALIDATION: Ensure risk scores are not constant after final write
    validate_risk_scores_file("results/main/risk_scores.csv", "main")
    
    print(f"  OK: Created results/main/risk_scores.csv")
    print(f"    Records: {len(h3_main_fixed)}, With technique_id: {(h3_main_fixed['technique_id'] != '').sum()}")
    print(f"    Unique techniques: {h3_main_fixed['technique_id'].nunique()}")
else:
    print("  WARNING: Register file not found: register/risk_register_main.csv or register/risk_register_full.csv")

# Small EMBER
print("Processing small_ember...")
df_small = pd.read_csv("register/risk_register_small_ember.csv")
# Validate probabilities
try:
    validate_probabilities(df_small, "small_ember")
except RuntimeError as e:
    print(f"  ERROR: VALIDATION FAILED: {e}")
    print(f"  WARNING: Cannot create small_ember split with constant probabilities.")
    sys.exit(1)

df_small["asset_id"] = df_small.index.map(lambda i: f"asset_{i:04d}")
df_small["risk_score"] = df_small["probability"].clip(0.0, 1.0)
df_small["predicted_label"] = (df_small["risk_score"] >= 0.5).astype(int)
df_small["true_label"] = df_small["label"].astype(int)
df_small["technique_id"] = df_small["attack_techniques"].apply(extract_technique_id)
df_small["technique_id"] = df_small["technique_id"].fillna('T1486').replace('', 'T1486').astype(str)
h3_small = df_small[["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]]
# Final safety check - ensure no empty values
h3_small.loc[h3_small['technique_id'] == '', 'technique_id'] = 'T1486'
h3_small['technique_id'] = h3_small['technique_id'].fillna('T1486')

# VALIDATION: Ensure risk scores are not constant before writing
assert_non_constant_scores(h3_small["risk_score"], split_name="small_ember", min_unique=5, min_std=1e-6)

Path("results/small_ember").mkdir(parents=True, exist_ok=True)
h3_small.to_csv("results/small_ember/risk_scores.csv", index=False)
# Post-process to ensure no empty technique_ids
h3_small_fixed = pd.read_csv("results/small_ember/risk_scores.csv", keep_default_na=False)
h3_small_fixed.loc[h3_small_fixed['technique_id'] == '', 'technique_id'] = 'T1486'
h3_small_fixed['technique_id'] = h3_small_fixed['technique_id'].fillna('T1486')
h3_small_fixed.to_csv("results/small_ember/risk_scores.csv", index=False)

# FINAL VALIDATION: Ensure risk scores are not constant after final write
validate_risk_scores_file("results/small_ember/risk_scores.csv", "small_ember")

print(f"  OK: Created results/small_ember/risk_scores.csv")
print(f"    Records: {len(h3_small_fixed)}, With technique_id: {(h3_small_fixed['technique_id'] != '').sum()}")

# Full EMBER
print("\nProcessing full_ember...")
df_full = pd.read_csv("register/risk_register_full.csv")
# Validate probabilities
try:
    validate_probabilities(df_full, "full_ember")
except RuntimeError as e:
    print(f"  ERROR: VALIDATION FAILED: {e}")
    print(f"  WARNING: Cannot create full_ember split with constant probabilities.")
    print(f"  Please regenerate register file with proper model predictions.")
    sys.exit(1)

df_full["asset_id"] = df_full.index.map(lambda i: f"asset_{i:04d}")
df_full["risk_score"] = df_full["probability"].clip(0.0, 1.0)
df_full["predicted_label"] = (df_full["risk_score"] >= 0.5).astype(int)
df_full["true_label"] = df_full["label"].astype(int)
df_full["technique_id"] = df_full["attack_techniques"].apply(extract_technique_id)
df_full["technique_id"] = df_full["technique_id"].fillna('T1486').replace('', 'T1486').astype(str)
h3_full = df_full[["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]]
# Final safety check - ensure no empty values
h3_full.loc[h3_full['technique_id'] == '', 'technique_id'] = 'T1486'
h3_full['technique_id'] = h3_full['technique_id'].fillna('T1486')

# VALIDATION: Ensure risk scores are not constant before writing
assert_non_constant_scores(h3_full["risk_score"], split_name="full_ember", min_unique=5, min_std=1e-6)

Path("results/full_ember").mkdir(parents=True, exist_ok=True)
h3_full.to_csv("results/full_ember/risk_scores.csv", index=False)
# Post-process to ensure no empty technique_ids
h3_full_fixed = pd.read_csv("results/full_ember/risk_scores.csv", keep_default_na=False)
h3_full_fixed.loc[h3_full_fixed['technique_id'] == '', 'technique_id'] = 'T1486'
h3_full_fixed['technique_id'] = h3_full_fixed['technique_id'].fillna('T1486')
h3_full_fixed.to_csv("results/full_ember/risk_scores.csv", index=False)

# FINAL VALIDATION: Ensure risk scores are not constant after final write
validate_risk_scores_file("results/full_ember/risk_scores.csv", "full_ember")

print(f"  OK: Created results/full_ember/risk_scores.csv")
print(f"    Records: {len(h3_full_fixed)}, With technique_id: {(h3_full_fixed['technique_id'] != '').sum()}")

# Smoke Test
print("\nProcessing smoke_test...")
df_smoke = pd.read_csv("register/smoke_test_register.csv")
df_smoke["asset_id"] = df_smoke.index.map(lambda i: f"asset_{i:04d}")
df_smoke["risk_score"] = df_smoke["probability"].clip(0.0, 1.0)
df_smoke["predicted_label"] = (df_smoke["risk_score"] >= 0.5).astype(int)
df_smoke["true_label"] = df_smoke["label"].astype(int)
df_smoke["technique_id"] = df_smoke["attack_techniques"].apply(extract_technique_id)
df_smoke["technique_id"] = df_smoke["technique_id"].fillna('T1486').replace('', 'T1486').astype(str)
h3_smoke = df_smoke[["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]]
# Final safety check - ensure no empty values
h3_smoke.loc[h3_smoke['technique_id'] == '', 'technique_id'] = 'T1486'
h3_smoke['technique_id'] = h3_smoke['technique_id'].fillna('T1486')
Path("results/smoke_test").mkdir(parents=True, exist_ok=True)
h3_smoke.to_csv("results/smoke_test/risk_scores.csv", index=False)
# Post-process to ensure no empty technique_ids
h3_smoke_fixed = pd.read_csv("results/smoke_test/risk_scores.csv", keep_default_na=False)
h3_smoke_fixed.loc[h3_smoke_fixed['technique_id'] == '', 'technique_id'] = 'T1486'
h3_smoke_fixed['technique_id'] = h3_smoke_fixed['technique_id'].fillna('T1486')
h3_smoke_fixed.to_csv("results/smoke_test/risk_scores.csv", index=False)
print(f"  OK: Created results/smoke_test/risk_scores.csv")
print(f"    Records: {len(h3_smoke_fixed)}, With technique_id: {(h3_smoke_fixed['technique_id'] != '').sum()}")

print("\n" + "="*80)
print("All splits created! Config updated in config/h3_splits.yaml")
print("You now have 4 splits: main, small_ember, full_ember, smoke_test")
print("Run: python run_h3_praxis.py")
print("="*80)
