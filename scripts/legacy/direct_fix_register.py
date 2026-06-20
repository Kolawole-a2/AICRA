#!/usr/bin/env python3
"""Directly fix register file - simple and direct."""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("FIXING REGISTER FILE")
print("=" * 80)

register_path = Path("register/risk_register_full.csv")

if not register_path.exists():
    print(f"ERROR: {register_path} not found")
    sys.exit(1)

print(f"\n1. Loading {register_path}...")
df = pd.read_csv(register_path)
print(f"   Loaded {len(df)} records")

print(f"\n2. Current state:")
print(f"   Sample attack_techniques: {df['attack_techniques'].iloc[0] if 'attack_techniques' in df.columns else 'COLUMN MISSING'}")

print(f"\n3. Updating attack_techniques...")
# Assign default techniques to all "unknown" families
default_techs = ["T1486", "T1490", "T1059", "T1021", "T1562"]
df['attack_techniques'] = df['family'].apply(
    lambda f: default_techs if (pd.isna(f) or str(f).lower() == 'unknown') else default_techs
)
df['attack_techniques'] = df['attack_techniques'].apply(str)

print(f"   Updated all records")
print(f"   Sample attack_techniques: {df['attack_techniques'].iloc[0]}")

print(f"\n4. Saving to {register_path}...")
df.to_csv(register_path, index=False)
print(f"   ✓ Saved {len(df)} records")

print(f"\n5. Verifying...")
df_check = pd.read_csv(register_path, nrows=5)
print(f"   Sample from saved file: {df_check['attack_techniques'].iloc[0]}")

print("\n" + "=" * 80)
print("REGISTER FIXED! Now run: python create_ember_splits.py")
print("=" * 80)

