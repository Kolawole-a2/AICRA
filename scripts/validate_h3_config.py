#!/usr/bin/env python3
"""
Validate H3 experiment configuration before running.

This script checks:
1. Reference pairs file exists and is different from deterministic mapping
2. Learned mapping exists and is different from deterministic mapping
3. All required files are present
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_h3_config(repo_root: Path) -> bool:
    """Validate H3 configuration. Returns True if valid, False otherwise."""
    print("=" * 80)
    print("H3 Configuration Validation")
    print("=" * 80)
    
    all_valid = True
    
    # Check file paths
    det_mapping = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    learned_mapping = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_pairs = repo_root / "d3fend_reference_pairs.csv"
    splits_config = repo_root / "config" / "h3_splits.yaml"
    
    # Check files exist
    print("\n1. Checking file existence...")
    for name, path in [
        ("Deterministic mapping", det_mapping),
        ("Learned mapping", learned_mapping),
        ("Reference pairs", ref_pairs),
        ("Splits config", splits_config),
    ]:
        if path.exists():
            print(f"   ✓ {name}: {path}")
        else:
            print(f"   ✗ {name}: NOT FOUND at {path}")
            all_valid = False
    
    if not all_valid:
        print("\n❌ Some required files are missing!")
        return False
    
    # Check reference pairs vs deterministic
    print("\n2. Checking reference pairs vs deterministic mapping...")
    det_hash = compute_file_hash(det_mapping)
    ref_hash = compute_file_hash(ref_pairs)
    
    print(f"   Deterministic hash: {det_hash[:32]}...")
    print(f"   Reference hash:     {ref_hash[:32]}...")
    
    if det_hash == ref_hash:
        print("   ✗ ERROR: Reference pairs file is IDENTICAL to deterministic mapping!")
        print("   Solution: Run 'python scripts/create_reference_pairs.py'")
        all_valid = False
    else:
        print("   ✓ Reference pairs are different from deterministic mapping")
    
    # Check learned mapping vs deterministic
    print("\n3. Checking learned mapping vs deterministic mapping...")
    try:
        det_df = pd.read_csv(det_mapping)
        lrn_df = pd.read_csv(learned_mapping)
        
        # Normalize column names
        det_cols = []
        if "attack_id" in det_df.columns:
            det_cols = ["attack_id", "defense_id"]
        elif "technique_id" in det_df.columns:
            det_cols = ["technique_id", "control_id"]
        else:
            print("   ✗ ERROR: Cannot identify columns in deterministic mapping")
            all_valid = False
        
        if all_valid and det_cols:
            det_pairs = set(zip(det_df[det_cols[0]], det_df[det_cols[1]]))
            lrn_pairs = set(zip(lrn_df["technique_id"], lrn_df["control_id"]))
            
            print(f"   Deterministic pairs: {len(det_pairs)}")
            print(f"   Learned pairs: {len(lrn_pairs)}")
            print(f"   Intersection: {len(det_pairs & lrn_pairs)}")
            print(f"   Only in deterministic: {len(det_pairs - lrn_pairs)}")
            print(f"   Only in learned: {len(lrn_pairs - det_pairs)}")
            
            if len(det_pairs - lrn_pairs) == 0 and len(lrn_pairs - det_pairs) == 0 and len(det_pairs) > 0:
                print("   ✗ ERROR: Learned mapping is IDENTICAL to deterministic mapping!")
                print("   Solution: Run 'python generate_learned_mapping.py'")
                all_valid = False
            else:
                overlap_pct = (len(det_pairs & lrn_pairs) / len(det_pairs) * 100) if len(det_pairs) > 0 else 0
                print(f"   ✓ Mappings are different (overlap: {overlap_pct:.1f}%)")
    except Exception as e:
        print(f"   ✗ ERROR: Failed to compare mappings: {e}")
        all_valid = False
    
    # Summary
    print("\n" + "=" * 80)
    if all_valid:
        print("✓ All H3 configuration checks PASSED")
        print("=" * 80)
        return True
    else:
        print("❌ H3 configuration validation FAILED")
        print("Please fix the issues above before running H3 experiment.")
        print("=" * 80)
        return False


if __name__ == "__main__":
    repo_root = Path.cwd()
    if len(sys.argv) > 1:
        repo_root = Path(sys.argv[1])
    
    is_valid = validate_h3_config(repo_root)
    sys.exit(0 if is_valid else 1)
