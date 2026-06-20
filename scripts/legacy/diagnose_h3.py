#!/usr/bin/env python3
"""Diagnostic script to check H3 evaluation setup."""

import sys
from pathlib import Path

print("=" * 80)
print("H3 Evaluation Diagnostic")
print("=" * 80)
print()

# Check Python
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Check paths
repo_root = Path(__file__).parent.resolve()
print(f"Repository root: {repo_root}")
print()

# Check required files
required_files = {
    "Config": repo_root / "config" / "h3_splits.yaml",
    "Deterministic": repo_root / "data" / "mappings" / "deterministic_lookup.csv",
    "Learned": repo_root / "data" / "mappings" / "learned_mapping.csv",
    "Reference": repo_root / "d3fend_reference_pairs.csv",
    "Risk scores": repo_root / "risk_scores.csv",
}

print("Checking required files:")
all_exist = True
for name, path in required_files.items():
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {name}: {path}")
    if not exists:
        all_exist = False
print()

if not all_exist:
    print("✗ Some required files are missing!")
    sys.exit(1)

# Try importing
print("Testing imports...")
try:
    from aicra.experiments.h3_evaluation import run_h3_evaluation
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test function call (dry run - just check it can be called)
print()
print("Testing function signature...")
import inspect
sig = inspect.signature(run_h3_evaluation)
print(f"✓ Function signature: {sig}")
print()

# Check output directory
output_dir = repo_root / "results" / "H3_full_evaluation"
print(f"Output directory: {output_dir}")
print(f"  Parent exists: {output_dir.parent.exists()}")
print(f"  Will be created: {output_dir}")
print()

print("=" * 80)
print("Setup looks good! Ready to run evaluation.")
print("=" * 80)
print()
print("To run the evaluation, execute:")
print("  python run_h3_fix.py")
print("  OR")
print("  python run_h3_evaluation.py")
print()
