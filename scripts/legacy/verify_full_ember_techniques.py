#!/usr/bin/env python3
"""Verify that full_ember has technique IDs after regeneration."""

import pandas as pd
from pathlib import Path
import sys

print("=" * 80)
print("VERIFYING FULL EMBER TECHNIQUE IDS")
print("=" * 80)

# Step 1: Check register file
print("\n1. Checking register file...")
register_path = Path("register/risk_register_full.csv")
if register_path.exists():
    df_reg = pd.read_csv(register_path, nrows=10)
    print(f"   Register file exists: ✓")
    print(f"   Columns: {list(df_reg.columns)}")
    print(f"   Sample attack_techniques:")
    for idx, row in df_reg.head(5).iterrows():
        techs = row.get('attack_techniques', 'N/A')
        print(f"     Row {idx}: family='{row.get('family', 'N/A')}', attack_techniques={techs}")
    
    # Check if techniques are populated
    if 'attack_techniques' in df_reg.columns:
        full_df = pd.read_csv(register_path)
        non_empty = full_df['attack_techniques'].apply(
            lambda x: str(x) != '[]' and str(x) != '' and pd.notna(x)
        ).sum()
        print(f"   Records with non-empty attack_techniques: {non_empty}/{len(full_df)}")
else:
    print(f"   Register file NOT FOUND: {register_path}")
    sys.exit(1)

# Step 2: Check H3 split file
print("\n2. Checking H3 split file...")
h3_path = Path("results/full_ember/risk_scores.csv")
if h3_path.exists():
    df_h3 = pd.read_csv(h3_path)
    print(f"   H3 split file exists: ✓")
    print(f"   Total samples: {len(df_h3)}")
    
    # Count technique IDs
    df_h3['technique_id'] = df_h3['technique_id'].replace('', pd.NA)
    n_with_tech = df_h3['technique_id'].notna().sum()
    n_unique = df_h3['technique_id'].dropna().nunique()
    
    print(f"   Samples with technique_id: {n_with_tech}/{len(df_h3)} ({n_with_tech/len(df_h3)*100:.1f}%)")
    print(f"   Unique techniques: {n_unique}")
    
    if n_with_tech > 0:
        print(f"   Sample technique_ids: {df_h3['technique_id'].dropna().unique()[:10].tolist()}")
        print("\n   ✅ SUCCESS: full_ember has technique IDs!")
    else:
        print("\n   ❌ PROBLEM: full_ember has NO technique IDs")
        print("   Run: python regenerate_register_with_techniques.py")
        print("   Then: python create_ember_splits.py")
else:
    print(f"   H3 split file NOT FOUND: {h3_path}")
    print("   Run: python create_ember_splits.py")

print("\n" + "=" * 80)

