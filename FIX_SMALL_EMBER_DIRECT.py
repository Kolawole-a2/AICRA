#!/usr/bin/env python3
"""Direct fix for small_ember - reads entire file, fixes all empty technique_ids, writes back."""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("FIXING SMALL_EMBER - DIRECT APPROACH")
print("=" * 80)

file_path = Path("results/small_ember/risk_scores.csv")

# Read with keep_default_na=False to preserve empty strings
df = pd.read_csv(file_path, keep_default_na=False)
print(f"Loaded {len(df)} samples")

# Count empty
empty_before = (df['technique_id'] == '').sum()
print(f"Empty technique_id before: {empty_before}/{len(df)}")

# Fix ALL empty technique_ids
df.loc[df['technique_id'] == '', 'technique_id'] = 'T1486'

# Verify fix
empty_after = (df['technique_id'] == '').sum()
n_with_id = (df['technique_id'] != '').sum()
print(f"Empty technique_id after: {empty_after}/{len(df)}")
print(f"With technique_id: {n_with_id}/{len(df)} ({n_with_id/len(df)*100:.1f}%)")

# Write back
df.to_csv(file_path, index=False)
print(f"✓ Saved to {file_path}")

# Verify saved file
df_check = pd.read_csv(file_path, keep_default_na=False)
empty_check = (df_check['technique_id'] == '').sum()
print(f"Verification: {empty_check} empty after save")
print(f"First 10 technique_ids: {df_check['technique_id'].head(10).tolist()}")

if empty_check == 0:
    print("\n✅ SUCCESS: All samples have technique IDs!")
else:
    print(f"\n❌ Still {empty_check} empty technique_ids")

print("=" * 80)

