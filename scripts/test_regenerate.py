#!/usr/bin/env python3
"""Test script to run regeneration and capture output."""

import sys
from pathlib import Path
import subprocess

repo_root = Path(__file__).parent.parent

# Run the regeneration script
print("Running regenerate_learned_mapping.py...")
print("=" * 80)

result = subprocess.run(
    [sys.executable, str(repo_root / "scripts" / "regenerate_learned_mapping.py")],
    cwd=repo_root,
    capture_output=True,
    text=True,
)

print("STDOUT:")
print(result.stdout)
print()
print("STDERR:")
print(result.stderr)
print()
print("Return code:", result.returncode)
print("=" * 80)

# Check if file exists and get stats
learned_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
if learned_path.exists():
    import pandas as pd
    df = pd.read_csv(learned_path)
    print(f"\nLearned mapping file exists: {learned_path}")
    print(f"  Total pairs: {len(df)}")
    print(f"  Unique techniques: {df['technique_id'].nunique()}")
    print(f"  Unique controls: {df['control_id'].nunique()}")
    
    # Compare with deterministic
    det_path = repo_root / "data" / "mappings" / "deterministic_attack_defense_lookup.csv"
    if det_path.exists():
        det_df = pd.read_csv(det_path)
        det_col = "attack_id" if "attack_id" in det_df.columns else "technique_id"
        det_ctrl_col = "defense_id" if "defense_id" in det_df.columns else "control_id"
        det_pairs = set(zip(det_df[det_col].astype(str), det_df[det_ctrl_col].astype(str)))
        learned_pairs = set(zip(df["technique_id"].astype(str), df["control_id"].astype(str)))
        
        print(f"\nComparison:")
        print(f"  Deterministic pairs: {len(det_pairs)}")
        print(f"  Learned pairs: {len(learned_pairs)}")
        print(f"  Learned-only pairs: {len(learned_pairs - det_pairs)}")
        print(f"  Is broader: {len(learned_pairs) > len(det_pairs) and len(learned_pairs - det_pairs) > 0}")
else:
    print(f"\nERROR: Learned mapping file not found at {learned_path}")
