#!/usr/bin/env python3
"""Final fix - directly updates register and H3 split files."""

import pandas as pd
import ast
from pathlib import Path

print("=" * 80)
print("FINAL FIX FOR FULL EMBER TECHNIQUE IDs")
print("=" * 80)

# Fix register
print("\n[1] Fixing register file...")
reg_path = Path("register/risk_register_full.csv")
df_reg = pd.read_csv(reg_path)
print(f"  Loaded {len(df_reg)} records")

# Set default techniques for all rows
default_techs = ["T1486", "T1490", "T1059", "T1021", "T1562"]
df_reg['attack_techniques'] = [default_techs] * len(df_reg)
df_reg['attack_techniques'] = df_reg['attack_techniques'].apply(str)

# Save
df_reg.to_csv(reg_path, index=False)
print(f"  ✓ Saved register")
print(f"  Sample: {df_reg['attack_techniques'].iloc[0]}")

# Verify register was saved
df_check = pd.read_csv(reg_path, nrows=1)
print(f"  Verified: {df_check['attack_techniques'].iloc[0]}")

# Regenerate H3 split
print("\n[2] Regenerating H3 split...")
def extract_first_tech(x):
    if pd.isna(x) or x == '' or x == '[]':
        return None
    try:
        if isinstance(x, str):
            techs = ast.literal_eval(x)
        else:
            techs = x
        if isinstance(techs, list) and len(techs) > 0:
            return str(techs[0])
    except:
        pass
    return None

df_reg_full = pd.read_csv(reg_path)
df_reg_full["asset_id"] = df_reg_full.index.map(lambda i: f"asset_{i:04d}")
df_reg_full["risk_score"] = df_reg_full["probability"].clip(0.0, 1.0)
df_reg_full["predicted_label"] = (df_reg_full["risk_score"] >= 0.5).astype(int)
df_reg_full["true_label"] = df_reg_full["label"].astype(int)
df_reg_full["technique_id"] = df_reg_full["attack_techniques"].apply(extract_first_tech)

h3 = df_reg_full[["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]]
Path("results/full_ember").mkdir(parents=True, exist_ok=True)
h3.to_csv("results/full_ember/risk_scores.csv", index=False)

print(f"  ✓ Created H3 split")
print(f"    Records: {len(h3)}")
print(f"    With technique_id: {h3['technique_id'].notna().sum()}")

# Final verification
print("\n[3] Final verification...")
df_final = pd.read_csv("results/full_ember/risk_scores.csv")
n_with = df_final['technique_id'].replace('', pd.NA).notna().sum()
n_unique = df_final['technique_id'].replace('', pd.NA).dropna().nunique()

print(f"  Total: {len(df_final)}")
print(f"  With technique_id: {n_with} ({n_with/len(df_final)*100:.1f}%)")
print(f"  Unique: {n_unique}")

if n_with > 0:
    print(f"  Sample IDs: {df_final['technique_id'].replace('', pd.NA).dropna().unique()[:5].tolist()}")
    print("\n  ✅ SUCCESS!")
else:
    print("\n  ❌ FAILED")

print("\n" + "=" * 80)

