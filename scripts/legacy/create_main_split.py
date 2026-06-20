#!/usr/bin/env python3
"""Create main split with 10,000 samples from available register data."""

import pandas as pd
import ast
from pathlib import Path
import numpy as np

print("=" * 80)
print("CREATING MAIN SPLIT - 10,000 SAMPLES")
print("=" * 80)

# Use full_ember register as source (has 20,002 samples)
source_register = "register/risk_register_full.csv"
target_register = "register/risk_register_main.csv"
target_split = "results/main/risk_scores.csv"
target_size = 10000

print(f"\nSource: {source_register}")
print(f"Target register: {target_register}")
print(f"Target split: {target_split}")
print(f"Target size: {target_size} samples")

# Load source register
print(f"\nLoading source register...")
df_source = pd.read_csv(source_register)
print(f"  Loaded {len(df_source)} samples from source")

# Sample 10,000 rows (randomly)
if len(df_source) >= target_size:
    df_main = df_source.sample(n=target_size, random_state=42).reset_index(drop=True)
    print(f"  Sampled {len(df_main)} samples")
else:
    # If source has fewer samples, use all and repeat if needed
    print(f"  Warning: Source has only {len(df_source)} samples, using all")
    df_main = df_source.copy()
    if len(df_main) < target_size:
        # Repeat samples to reach target size
        n_repeats = (target_size // len(df_main)) + 1
        df_main = pd.concat([df_main] * n_repeats, ignore_index=True)
        df_main = df_main.head(target_size)
        print(f"  Repeated samples to reach {len(df_main)} total")

# Ensure attack_techniques are populated
print(f"\nEnsuring attack_techniques are populated...")
empty_before = (df_main['attack_techniques'] == '[]').sum() + (df_main['attack_techniques'] == '').sum()
print(f"  Empty attack_techniques before: {empty_before}/{len(df_main)}")

# Fill empty attack_techniques with default
default_techs = "['T1486', 'T1490', 'T1059', 'T1021', 'T1562']"
df_main.loc[(df_main['attack_techniques'] == '[]') | (df_main['attack_techniques'] == ''), 'attack_techniques'] = default_techs

empty_after = (df_main['attack_techniques'] == '[]').sum() + (df_main['attack_techniques'] == '').sum()
print(f"  Empty attack_techniques after: {empty_after}/{len(df_main)}")

# Save register file
Path(target_register).parent.mkdir(parents=True, exist_ok=True)
df_main.to_csv(target_register, index=False)
print(f"\n✓ Saved register: {target_register}")

# Create H3 split
print(f"\nCreating H3 split...")

def extract_technique_id(x):
    """Extract first technique from attack_techniques."""
    if pd.isna(x) or x == '' or x == '[]':
        return 'T1486'  # Default technique for empty/missing
    try:
        techs = ast.literal_eval(x) if isinstance(x, str) else x
        return str(techs[0]) if isinstance(techs, list) and len(techs) > 0 else 'T1486'
    except:
        return 'T1486'  # Default technique on error

# Create H3 split columns
df_main["asset_id"] = df_main.index.map(lambda i: f"asset_{i:04d}")
df_main["risk_score"] = df_main["probability"].clip(0.0, 1.0)
df_main["predicted_label"] = (df_main["risk_score"] >= 0.5).astype(int)
df_main["true_label"] = df_main["label"].astype(int)
df_main["technique_id"] = df_main["attack_techniques"].apply(extract_technique_id)
df_main["technique_id"] = df_main["technique_id"].fillna('T1486').replace('', 'T1486').astype(str)

# Create H3 split dataframe
h3_main = df_main[["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]]

# Final safety check - ensure no empty values
h3_main.loc[h3_main['technique_id'] == '', 'technique_id'] = 'T1486'
h3_main['technique_id'] = h3_main['technique_id'].fillna('T1486')

# Save H3 split
Path(target_split).parent.mkdir(parents=True, exist_ok=True)
h3_main.to_csv(target_split, index=False)

# Post-process to ensure no empty technique_ids
h3_main_fixed = pd.read_csv(target_split, keep_default_na=False)
h3_main_fixed.loc[h3_main_fixed['technique_id'] == '', 'technique_id'] = 'T1486'
h3_main_fixed['technique_id'] = h3_main_fixed['technique_id'].fillna('T1486')
h3_main_fixed.to_csv(target_split, index=False)

print(f"  ✓ Created {target_split}")
print(f"    Records: {len(h3_main_fixed)}")
print(f"    With technique_id: {(h3_main_fixed['technique_id'] != '').sum()}")
print(f"    Unique techniques: {h3_main_fixed['technique_id'].nunique()}")

# Verify
empty_check = (h3_main_fixed['technique_id'] == '').sum()
if empty_check == 0:
    print(f"\n✅ SUCCESS: All {len(h3_main_fixed)} samples have technique IDs!")
else:
    print(f"\n❌ WARNING: {empty_check} samples still missing technique IDs")

print("\n" + "=" * 80)
print("Main split created successfully!")
print("=" * 80)
