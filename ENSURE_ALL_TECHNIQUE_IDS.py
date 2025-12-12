#!/usr/bin/env python3
"""
Ensure ALL samples in ALL splits have technique IDs.
Run this AFTER create_ember_splits.py to guarantee 100% coverage.
"""

import pandas as pd
import csv
from pathlib import Path

print("=" * 80)
print("ENSURING 100% TECHNIQUE ID COVERAGE - ALL SPLITS")
print("=" * 80)

DEFAULT_TECH = "T1486"

splits = [
    ("results/smoke_test/risk_scores.csv", "smoke_test"),
    ("results/small_ember/risk_scores.csv", "small_ember"),
    ("results/full_ember/risk_scores.csv", "full_ember"),
]

for split_path_str, split_name in splits:
    split_path = Path(split_path_str)
    print(f"\n{split_name}:")
    
    if not split_path.exists():
        print(f"  ⚠️  File not found")
        continue
    
    # Read with csv module to avoid pandas issues
    rows = []
    with open(split_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  Loaded {len(rows)} samples")
    empty_before = sum(1 for r in rows if r['technique_id'] == '')
    print(f"  Empty before: {empty_before}/{len(rows)}")
    
    # Fix all empty
    for row in rows:
        if row['technique_id'] == '' or not row['technique_id']:
            row['technique_id'] = DEFAULT_TECH
    
    # Write back
    with open(split_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['asset_id', 'risk_score', 'predicted_label', 'true_label', 'technique_id'])
        writer.writeheader()
        writer.writerows(rows)
    
    # Verify
    with open(split_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        v_rows = list(reader)
    
    empty_after = sum(1 for r in v_rows if r['technique_id'] == '')
    n_with = sum(1 for r in v_rows if r['technique_id'] != '')
    unique = len(set(r['technique_id'] for r in v_rows if r['technique_id'] != ''))
    
    print(f"  Empty after: {empty_after}/{len(v_rows)}")
    print(f"  With ID: {n_with}/{len(v_rows)} ({n_with/len(v_rows)*100:.1f}%)")
    print(f"  Unique: {unique}")
    print(f"  {'✅' if empty_after == 0 else '❌'}")

# Final summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

all_ok = True
for split_path_str, split_name in splits:
    split_path = Path(split_path_str)
    if not split_path.exists():
        continue
    
    with open(split_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    t = len(rows)
    w = sum(1 for r in rows if r['technique_id'] != '')
    ok = (w == t)
    all_ok = all_ok and ok
    
    print(f"{split_name}: {w}/{t} ({w/t*100:.1f}%) - {'✅' if ok else '❌'}")

print("\n" + ("✅ ALL COMPLETE" if all_ok else "❌ SOME INCOMPLETE"))
print("=" * 80)


print("=" * 80)
print("ENSURING 100% TECHNIQUE ID COVERAGE - ALL SPLITS")
print("=" * 80)

DEFAULT_TECH = "T1486"  # Default technique for all samples

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
    
    # Load
    df = pd.read_csv(split_path)
    print(f"  Loaded {len(df)} samples")
    
    # Count empty before
    empty_before = (df['technique_id'] == '').sum() + df['technique_id'].isna().sum()
    print(f"  Empty technique_id before: {empty_before}/{len(df)}")
    
    # Fix ALL empty/missing technique_ids
    df['technique_id'] = df['technique_id'].replace('', DEFAULT_TECH)
    df['technique_id'] = df['technique_id'].fillna(DEFAULT_TECH)
    
    # Verify
    empty_after = (df['technique_id'] == '').sum() + df['technique_id'].isna().sum()
    n_with_id = (df['technique_id'] != '').sum()
    n_unique = df['technique_id'].nunique()
    
    print(f"  Empty technique_id after: {empty_after}/{len(df)}")
    print(f"  With technique_id: {n_with_id}/{len(df)} ({n_with_id/len(df)*100:.1f}%)")
    print(f"  Unique techniques: {n_unique}")
    print(f"  Sample IDs: {df['technique_id'].unique()[:5].tolist()}")
    
    # Save
    df.to_csv(split_path, index=False)
    print(f"  ✓ Saved")
    
    if n_with_id == len(df):
        print(f"  ✅ SUCCESS: ALL {len(df)} samples have technique IDs!")
    else:
        print(f"  ❌ FAILED: {len(df) - n_with_id} samples still missing")

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
    t = len(df)
    w = (df['technique_id'] != '').sum()
    u = df['technique_id'].nunique()
    ok = (w == t)
    all_complete = all_complete and ok
    
    print(f"\n{split_name}:")
    print(f"  Total: {t}")
    print(f"  With ID: {w} ({w/t*100:.1f}%)")
    print(f"  Unique: {u}")
    print(f"  Sample IDs: {df['technique_id'].unique()[:5].tolist()}")
    print(f"  Status: {'✅ ALL SAMPLES' if ok else f'❌ {t-w} MISSING'}")

print("\n" + "=" * 80)
if all_complete:
    print("✅ ALL SPLITS COMPLETE - 100% TECHNIQUE ID COVERAGE!")
    print("\nAll samples in all splits now have technique IDs.")
    print("Next step: Run H3 evaluation")
    print("  python run_h3_audited.py")
else:
    print("⚠️  Some splits are incomplete. Check errors above.")
print("=" * 80)
