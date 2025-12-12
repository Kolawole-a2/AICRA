#!/usr/bin/env python3
"""Test H1 and H2 experiments with explicit output."""

import sys
import traceback
from pathlib import Path

print("=" * 80)
print("Testing H1 and H2 Experiments")
print("=" * 80)
print()

# Check data files first
print("Step 1: Checking data files...")
try:
    from aicra.config import Settings
    s = Settings()
    print(f"EMBER directory: {s.ember_dir}")
    print(f"Directory exists: {s.ember_dir.exists()}")
    
    required_files = ['train_features.jsonl', 'train_labels.jsonl', 'test_features.jsonl', 'test_labels.jsonl']
    all_exist = True
    for f in required_files:
        exists = (s.ember_dir / f).exists()
        print(f"  {f}: {'✓' if exists else '✗'}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  WARNING: Some required data files are missing!")
        print("   The experiments will fail if data is not available.")
        print("   Expected location: data/ember2024/")
        print()
        sys.exit(1)
    else:
        print("\n✓ All required data files found!")
        print()
except Exception as e:
    print(f"ERROR checking data files: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test H1
print("Step 2: Testing H1 experiment...")
print("-" * 80)
try:
    from aicra.experiments.h1_classification import run_h1_classification_experiment
    
    output_dir = Path("results/H1_classification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running H1 experiment...")
    print(f"Output directory: {output_dir}")
    
    result = run_h1_classification_experiment(
        output_dir=output_dir,
        model_type="lgbm",
        operational_threshold=0.5,
        use_pe_features=True,
        repo_root=Path.cwd(),
    )
    
    print()
    print("✓ H1 experiment completed successfully!")
    print(f"  AUROC: {result.get('auroc', 'N/A'):.4f}")
    print(f"  PR-AUC: {result.get('pr_auc', 'N/A'):.4f}")
    print(f"  Results saved to: {output_dir}")
    
    # Check output files
    expected_files = ['metrics.json', 'H1_full_results.json', 'summary.md', 'H1_summary.md']
    print("\n  Output files:")
    for f in expected_files:
        exists = (output_dir / f).exists()
        print(f"    {f}: {'✓' if exists else '✗'}")
    
    print()
    
except Exception as e:
    print(f"\n✗ H1 experiment failed: {e}")
    traceback.print_exc()
    print("\nSkipping H2 (depends on H1)...")
    sys.exit(1)

# Test H2
print("Step 3: Testing H2 experiment...")
print("-" * 80)
try:
    from aicra.experiments.h2_calibration_thresholds import run_h2_calibration_thresholds_experiment
    
    output_dir = Path("results/H2_calibration_thresholds")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running H2 experiment...")
    print(f"Output directory: {output_dir}")
    
    result = run_h2_calibration_thresholds_experiment(
        output_dir=output_dir,
        cost_fn=10.0,
        cost_fp=1.0,
        calibration_method="auto",
        repo_root=Path.cwd(),
    )
    
    print()
    print("✓ H2 experiment completed successfully!")
    cal = result.get('calibration', {})
    print(f"  Brier (uncalibrated): {cal.get('brier_uncalibrated', 'N/A'):.4f}")
    print(f"  Brier (calibrated): {cal.get('brier_calibrated', 'N/A'):.4f}")
    print(f"  Results saved to: {output_dir}")
    
    # Check output files
    expected_files = ['metrics.json', 'H2_full_results.json', 'summary.md', 'H2_summary.md']
    print("\n  Output files:")
    for f in expected_files:
        exists = (output_dir / f).exists()
        print(f"    {f}: {'✓' if exists else '✗'}")
    
    print()
    
except Exception as e:
    print(f"\n✗ H2 experiment failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("✓ All experiments completed successfully!")
print("=" * 80)
print()
print("Next step: Generate validation report")
print("  python scripts/generate_praxis_validation_report.py")
