#!/usr/bin/env python3
"""Fix small_ember CSV directly using csv module - bypasses pandas issues."""

import csv
from pathlib import Path

print("=" * 80)
print("FIXING SMALL_EMBER CSV - DIRECT CSV MODULE APPROACH")
print("=" * 80)

file_path = Path("results/small_ember/risk_scores.csv")

# Read CSV
rows = []
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Loaded {len(rows)} rows")
empty_before = sum(1 for r in rows if r['technique_id'] == '')
print(f"Empty technique_id before: {empty_before}/{len(rows)}")

# Fix all empty technique_ids
for row in rows:
    if row['technique_id'] == '':
        row['technique_id'] = 'T1486'

# Write back
with open(file_path, 'w', encoding='utf-8', newline='') as f:
    if rows:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

print(f"✓ Saved to {file_path}")

# Verify
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    v_rows = list(reader)

empty_after = sum(1 for r in v_rows if r['technique_id'] == '')
n_with_id = sum(1 for r in v_rows if r['technique_id'] != '')
print(f"Verification: {empty_after} empty after save")
print(f"With technique_id: {n_with_id}/{len(v_rows)} ({n_with_id/len(v_rows)*100:.1f}%)")
print(f"First 10 technique_ids: {[r['technique_id'] for r in v_rows[:10]]}")

if empty_after == 0:
    print("\n✅ SUCCESS: All samples have technique IDs!")
else:
    print(f"\n❌ Still {empty_after} empty technique_ids")

print("=" * 80)

