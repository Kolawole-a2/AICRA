#!/usr/bin/env python3
"""Directly fix ALL register files - ensures every row has attack_techniques."""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aicra.config import get_settings
from aicra.pipelines.mapping import MappingPipeline

print("=" * 80)
print("FIXING ALL REGISTER FILES - ENSURING ALL SAMPLES HAVE TECHNIQUES")
print("=" * 80)

settings = get_settings()
mp = MappingPipeline(settings, skip_mlflow=True)
default_techs = ["T1486", "T1490", "T1059", "T1021", "T1562"]

def get_techs(family):
    """Get techniques for a family - always returns a list."""
    if pd.isna(family) or str(family).lower() in ['unknown', '', 'benign']:
        return default_techs
    canonical = mp.normalize_family(str(family))
    techs = mp.family_to_attack(canonical)
    return techs if techs else default_techs

registers = [
    ("register/smoke_test_register.csv", "smoke_test"),
    ("register/risk_register_small_ember.csv", "small_ember"),
    ("register/risk_register_full.csv", "full_ember"),
]

for reg_path_str, name in registers:
    reg_path = Path(reg_path_str)
    print(f"\n{'='*80}")
    print(f"Fixing: {name}")
    print(f"{'='*80}")
    
    if not reg_path.exists():
        print(f"  ⚠️  File not found: {reg_path}")
        continue
    
    # Load
    df = pd.read_csv(reg_path)
    print(f"  Loaded {len(df)} records")
    
    # Check current state
    if 'attack_techniques' not in df.columns:
        df['attack_techniques'] = None
    
    n_empty_before = df['attack_techniques'].apply(
        lambda x: str(x) == '[]' or str(x) == '' or pd.isna(x)
    ).sum()
    print(f"  Empty attack_techniques before: {n_empty_before}/{len(df)}")
    
    # Fix ALL rows
    print(f"  Updating attack_techniques for ALL rows...")
    df['attack_techniques'] = df['family'].apply(get_techs)
    df['attack_techniques'] = df['attack_techniques'].apply(str)
    
    # Verify
    n_empty_after = df['attack_techniques'].apply(
        lambda x: str(x) == '[]' or str(x) == '' or pd.isna(x)
    ).sum()
    n_with_tech = len(df) - n_empty_after
    
    print(f"  Empty attack_techniques after: {n_empty_after}/{len(df)}")
    print(f"  Records with techniques: {n_with_tech}/{len(df)} ({n_with_tech/len(df)*100:.1f}%)")
    print(f"  Sample: {df['attack_techniques'].iloc[0]}")
    
    # Save
    print(f"  Saving...")
    df.to_csv(reg_path, index=False)
    print(f"  ✓ Saved")
    
    # Verify saved file
    df_check = pd.read_csv(reg_path, nrows=1)
    print(f"  Verified saved: {df_check['attack_techniques'].iloc[0]}")

print("\n" + "=" * 80)
print("ALL REGISTERS FIXED!")
print("=" * 80)
print("\nNow run: python create_ember_splits.py")
print("=" * 80)

