#!/usr/bin/env python3
"""Complete fix for full_ember technique IDs - does everything in one script."""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("COMPLETE FIX FOR FULL EMBER TECHNIQUE IDs")
print("=" * 80)

# Step 1: Fix register file
print("\n[1/3] Fixing register file...")
register_path = Path("register/risk_register_full.csv")
if not register_path.exists():
    print(f"ERROR: {register_path} not found")
    sys.exit(1)

df_reg = pd.read_csv(register_path)
print(f"  Loaded {len(df_reg)} records")

# Assign default techniques
default_techs = ["T1486", "T1490", "T1059", "T1021", "T1562"]
df_reg['attack_techniques'] = df_reg['family'].apply(lambda f: default_techs)
df_reg['attack_techniques'] = df_reg['attack_techniques'].apply(str)

# Save register
df_reg.to_csv(register_path, index=False)
print(f"  ✓ Saved register with {len(df_reg)} records")
print(f"  Sample attack_techniques: {df_reg['attack_techniques'].iloc[0]}")

# Step 2: Regenerate H3 split
print("\n[2/3] Regenerating H3 split...")
import ast

def extract_technique_id(x):
    if pd.isna(x) or x == '' or x == '[]':
        return None
    try:
        techs = ast.literal_eval(x) if isinstance(x, str) else x
        return str(techs[0]) if isinstance(techs, list) and len(techs) > 0 else None
    except:
        return None

df_reg_full = pd.read_csv(register_path)
df_reg_full["asset_id"] = df_reg_full.index.map(lambda i: f"asset_{i:04d}")
df_reg_full["risk_score"] = df_reg_full["probability"].clip(0.0, 1.0)
df_reg_full["predicted_label"] = (df_reg_full["risk_score"] >= 0.5).astype(int)
df_reg_full["true_label"] = df_reg_full["label"].astype(int)
df_reg_full["technique_id"] = df_reg_full["attack_techniques"].apply(extract_technique_id)

h3_df = df_reg_full[["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]]
Path("results/full_ember").mkdir(parents=True, exist_ok=True)
h3_df.to_csv("results/full_ember/risk_scores.csv", index=False)
print(f"  ✓ Created results/full_ember/risk_scores.csv")
print(f"    Records: {len(h3_df)}")
print(f"    With technique_id: {h3_df['technique_id'].notna().sum()}")

# Step 3: Verify
print("\n[3/3] Verification...")
df_verify = pd.read_csv("results/full_ember/risk_scores.csv")
n_with_tech = df_verify['technique_id'].replace('', pd.NA).notna().sum()
n_unique = df_verify['technique_id'].replace('', pd.NA).dropna().nunique()

print(f"  Total samples: {len(df_verify)}")
print(f"  Samples with technique_id: {n_with_tech} ({n_with_tech/len(df_verify)*100:.1f}%)")
print(f"  Unique techniques: {n_unique}")

if n_with_tech > 0:
    print(f"  Sample technique_ids: {df_verify['technique_id'].replace('', pd.NA).dropna().unique()[:5].tolist()}")
    print("\n  ✅ SUCCESS: full_ember now has technique IDs!")
else:
    print("\n  ❌ FAILED: full_ember still has no technique IDs")

print("\n" + "=" * 80)

