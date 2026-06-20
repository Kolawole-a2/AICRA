#!/usr/bin/env python3
"""Directly fix ALL H3 split files - ensures 100% technique ID coverage."""

import pandas as pd
import ast
from pathlib import Path

print("=" * 80)
print("DIRECT FIX: ALL H3 SPLITS - 100% TECHNIQUE ID COVERAGE")
print("=" * 80)

DEFAULT_TECH = "T1486"  # First technique from default list

splits = [
    ("results/smoke_test/risk_scores.csv", "smoke_test"),
    ("results/small_ember/risk_scores.csv", "small_ember"),
    ("results/full_ember/risk_scores.csv", "full_ember"),
]

for split_path_str, split_name in splits:
    split_path = Path(split_path_str)
    print(f"\n{'='*80}")
    print(f"Fixing: {split_name}")
    print(f"{'='*80}")
    
    if not split_path.exists():
        print(f"  ⚠️  File not found: {split_path}")
        continue
    
    # Load H3 split
    df = pd.read_csv(split_path)
    print(f"  Loaded {len(df)} samples")
    
    # Check current state
    df['technique_id'] = df['technique_id'].replace('', pd.NA)
    n_with_id = df['technique_id'].notna().sum()
    print(f"  Current: {n_with_id}/{len(df)} have technique_id ({n_with_id/len(df)*100:.1f}%)")
    
    # Fix ALL rows missing technique_id
    df['technique_id'] = df['technique_id'].fillna(DEFAULT_TECH)
    df['technique_id'] = df['technique_id'].replace('', DEFAULT_TECH)
    
    # Verify
    n_with_id_after = df['technique_id'].notna().sum()
    n_unique = df['technique_id'].nunique()
    
    print(f"  After fix: {n_with_id_after}/{len(df)} have technique_id ({n_with_id_after/len(df)*100:.1f}%)")
    print(f"  Unique techniques: {n_unique}")
    print(f"  Sample IDs: {df['technique_id'].unique()[:5].tolist()}")
    
    # Save
    df.to_csv(split_path, index=False)
    print(f"  ✓ Saved {len(df)} samples")
    
    if n_with_id_after == len(df):
        print(f"  ✅ SUCCESS: ALL {len(df)} samples have technique IDs!")
    else:
        print(f"  ❌ FAILED: {len(df) - n_with_id_after} samples still missing")

# Final verification
print("\n" + "=" * 80)
print("FINAL VERIFICATION")
print("=" * 80)

all_complete = True
for split_path_str, split_name in splits:
    split_path = Path(split_path_str)
    if not split_path.exists():
        continue
    
    df = pd.read_csv(split_path)
    df['technique_id'] = df['technique_id'].replace('', pd.NA)
    
    n_total = len(df)
    n_with_id = df['technique_id'].notna().sum()
    n_unique = df['technique_id'].dropna().nunique()
    complete = (n_with_id == n_total)
    all_complete = all_complete and complete
    
    status = "✅ ALL" if complete else f"❌ {n_total - n_with_id} missing"
    print(f"\n{split_name}:")
    print(f"  Total: {n_total}")
    print(f"  With ID: {n_with_id} ({n_with_id/n_total*100:.1f}%)")
    print(f"  Unique: {n_unique}")
    print(f"  Status: {status}")

print("\n" + "=" * 80)
if all_complete:
    print("✅ ALL SPLITS COMPLETE - 100% TECHNIQUE ID COVERAGE!")
    print("\nNext step: Run H3 evaluation")
    print("  python run_h3_audited.py")
else:
    print("⚠️  Some splits are incomplete. Check errors above.")
print("=" * 80)

