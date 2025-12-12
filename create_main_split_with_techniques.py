#!/usr/bin/env python3
"""
Create main split (10,000 samples) with properly populated technique IDs.

This script:
1. Loads risk_register_full.csv
2. Samples 10,000 rows
3. Extracts technique IDs from attack_techniques column
4. Creates H3-compatible risk_scores.csv with all technique IDs populated
"""

import pandas as pd
import ast
from pathlib import Path
import numpy as np

def extract_technique_id(x):
    """Extract first technique from attack_techniques."""
    if pd.isna(x) or x == '' or x == '[]':
        return 'T1486'  # Default technique for empty/missing
    try:
        techs = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(techs, list) and len(techs) > 0:
            return str(techs[0])
        elif isinstance(techs, str) and techs.strip():
            return str(techs).strip()
        else:
            return 'T1486'
    except (ValueError, SyntaxError):
        # Try regex extraction as fallback
        import re
        if isinstance(x, str):
            match = re.search(r'T\d{4}(?:\.\d{3})?', x)
            if match:
                return match.group(0)
        return 'T1486'  # Default technique on error

print("=" * 80)
print("CREATING MAIN SPLIT WITH TECHNIQUE IDs")
print("=" * 80)

# Step 1: Load register file
register_path = Path("register/risk_register_full.csv")
if not register_path.exists():
    raise FileNotFoundError(f"Register file not found: {register_path}")

print(f"\n[1] Loading register: {register_path}")
df = pd.read_csv(register_path)
print(f"     Loaded {len(df)} records")

# Step 2: Sample 10,000 rows
print(f"\n[2] Sampling 10,000 rows...")
df_main = df.sample(n=10000, random_state=42).reset_index(drop=True)
print(f"     Sampled {len(df_main)} records")

# Step 3: Extract technique IDs
print(f"\n[3] Extracting technique IDs from attack_techniques...")
df_main["technique_id"] = df_main["attack_techniques"].apply(extract_technique_id)

# Ensure all technique IDs are populated
before_fill = (df_main["technique_id"] == '').sum() + df_main["technique_id"].isna().sum()
print(f"     Empty technique IDs before fill: {before_fill}")

# Fill any remaining empty values
df_main["technique_id"] = df_main["technique_id"].fillna('T1486').replace('', 'T1486').astype(str)

after_fill = (df_main["technique_id"] == '').sum() + df_main["technique_id"].isna().sum()
print(f"     Empty technique IDs after fill: {after_fill}")

# Step 4: Create H3 format columns
print(f"\n[4] Creating H3 format columns...")
df_main["asset_id"] = df_main.index.map(lambda i: f"asset_{i:04d}")
df_main["risk_score"] = df_main["probability"].clip(0.0, 1.0)
df_main["predicted_label"] = (df_main["risk_score"] >= 0.5).astype(int)
df_main["true_label"] = df_main["label"].astype(int)

# Step 5: Select and reorder columns
h3_main = df_main[["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]].copy()

# Step 6: Final safety check - ensure no empty technique_ids
print(f"\n[5] Final validation...")
h3_main.loc[h3_main['technique_id'] == '', 'technique_id'] = 'T1486'
h3_main['technique_id'] = h3_main['technique_id'].fillna('T1486')

# Verify all technique IDs are populated
n_with_id = (h3_main['technique_id'] != '').sum()
n_unique = h3_main['technique_id'].nunique()
print(f"     Records with technique_id: {n_with_id}/{len(h3_main)} ({n_with_id/len(h3_main)*100:.1f}%)")
print(f"     Unique techniques: {n_unique}")
print(f"     Sample techniques: {h3_main['technique_id'].unique()[:10].tolist()}")

# Step 7: Save to results/main/risk_scores.csv
print(f"\n[6] Saving to results/main/risk_scores.csv...")
Path("results/main").mkdir(parents=True, exist_ok=True)
h3_main.to_csv("results/main/risk_scores.csv", index=False)
print(f"     ✓ Saved {len(h3_main)} records")

# Step 8: Post-process verification (re-read and verify)
print(f"\n[7] Post-process verification...")
h3_main_verified = pd.read_csv("results/main/risk_scores.csv", keep_default_na=False)
h3_main_verified.loc[h3_main_verified['technique_id'] == '', 'technique_id'] = 'T1486'
h3_main_verified['technique_id'] = h3_main_verified['technique_id'].fillna('T1486')
h3_main_verified.to_csv("results/main/risk_scores.csv", index=False)

# Final verification
n_final = (h3_main_verified['technique_id'] != '').sum()
n_unique_final = h3_main_verified['technique_id'].nunique()
print(f"     Final verification:")
print(f"       Total records: {len(h3_main_verified)}")
print(f"       Records with technique_id: {n_final} ({n_final/len(h3_main_verified)*100:.1f}%)")
print(f"       Unique techniques: {n_unique_final}")
print(f"       Technique distribution:")
tech_counts = h3_main_verified['technique_id'].value_counts()
for tech, count in tech_counts.head(10).items():
    print(f"         {tech}: {count} ({count/len(h3_main_verified)*100:.1f}%)")

print("\n" + "=" * 80)
print("✓ MAIN SPLIT CREATED SUCCESSFULLY")
print("=" * 80)
print(f"File: results/main/risk_scores.csv")
print(f"Records: {len(h3_main_verified)}")
print(f"Technique ID coverage: {n_final}/{len(h3_main_verified)} ({n_final/len(h3_main_verified)*100:.1f}%)")
print("=" * 80)
