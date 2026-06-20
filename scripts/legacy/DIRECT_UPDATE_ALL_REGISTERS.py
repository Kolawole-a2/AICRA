#!/usr/bin/env python3
"""Directly update ALL register files - no dependencies, just pandas."""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("DIRECT UPDATE: ALL REGISTER FILES")
print("=" * 80)

DEFAULT_TECHS_STR = "['T1486', 'T1490', 'T1059', 'T1021', 'T1562']"

registers = [
    ("register/smoke_test_register.csv", "smoke_test"),
    ("register/risk_register_small_ember.csv", "small_ember"),
    ("register/risk_register_full.csv", "full_ember"),
]

for reg_path_str, name in registers:
    reg_path = Path(reg_path_str)
    print(f"\n{name}:")
    
    if not reg_path.exists():
        print(f"  ⚠️  File not found")
        continue
    
    # Load
    df = pd.read_csv(reg_path)
    print(f"  Loaded {len(df)} records")
    
    # Set ALL rows to default techniques
    df['attack_techniques'] = DEFAULT_TECHS_STR
    print(f"  Updated all {len(df)} rows")
    print(f"  Sample: {df['attack_techniques'].iloc[0]}")
    
    # Save
    df.to_csv(reg_path, index=False)
    print(f"  ✓ Saved")
    
    # Verify
    df_check = pd.read_csv(reg_path, nrows=1)
    print(f"  Verified: {df_check['attack_techniques'].iloc[0]}")

print("\n" + "=" * 80)
print("ALL REGISTERS UPDATED!")
print("=" * 80)
print("\nNow run: python create_ember_splits.py")
print("=" * 80)

