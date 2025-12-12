#!/usr/bin/env python3
"""Verify all splits have technique IDs."""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("VERIFICATION: ALL SPLITS TECHNIQUE IDs")
print("=" * 80)

splits = {
    'smoke_test': Path('results/smoke_test/risk_scores.csv'),
    'small_ember': Path('results/small_ember/risk_scores.csv'),
    'full_ember': Path('results/full_ember/risk_scores.csv'),
}

all_good = True
for name, path in splits.items():
    print(f"\n{name}:")
    if not path.exists():
        print(f"  ❌ File not found: {path}")
        all_good = False
        continue
    
    df = pd.read_csv(path)
    df['technique_id'] = df['technique_id'].replace('', pd.NA)
    
    n_total = len(df)
    n_with = df['technique_id'].notna().sum()
    n_unique = df['technique_id'].dropna().nunique()
    pct = (n_with / n_total * 100) if n_total > 0 else 0
    
    print(f"  Total samples: {n_total}")
    print(f"  With technique_id: {n_with} ({pct:.1f}%)")
    print(f"  Unique techniques: {n_unique}")
    
    if n_with == n_total:
        print(f"  ✅ ALL samples have technique IDs")
        if n_unique > 0:
            sample_ids = df['technique_id'].dropna().unique()[:5].tolist()
            print(f"  Sample IDs: {sample_ids}")
    else:
        print(f"  ❌ {n_total - n_with} samples missing technique IDs")
        all_good = False

print("\n" + "=" * 80)
if all_good:
    print("✅ ALL SPLITS HAVE TECHNIQUE IDs!")
else:
    print("❌ SOME SPLITS ARE MISSING TECHNIQUE IDs")
print("=" * 80)

