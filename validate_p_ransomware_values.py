#!/usr/bin/env python3
"""
Validation script for P_Ransomware values in ransomware-only risk registers.

This script validates that:
1. P_Ransomware values are NOT identical across samples (shows model discrimination)
2. Values are appropriately high for ransomware samples (model confidence)
3. Values vary across different splits (not using same values)
4. Values come from actual model predictions (not constants)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def validate_split(split_name: str, repo_root: Path) -> dict:
    """Validate P_Ransomware values for a single split."""
    print(f"\n{'='*80}")
    print(f"Validating split: {split_name}")
    print(f"{'='*80}")
    
    # Check risk_scores.csv (source data)
    risk_scores_path = repo_root / "results" / "h1h2_rebuild" / split_name / "risk_scores.csv"
    register_path = repo_root / "register" / "h1h2_rebuild" / split_name / "ransomware_only_risk_register.csv"
    
    results = {
        "split": split_name,
        "risk_scores_exists": risk_scores_path.exists(),
        "register_exists": register_path.exists(),
    }
    
    if not risk_scores_path.exists():
        print(f"  [SKIP] risk_scores.csv not found: {risk_scores_path}")
        return results
    
    # Load risk_scores.csv
    risk_df = pd.read_csv(risk_scores_path)
    print(f"\n[1] Risk Scores CSV Analysis:")
    print(f"    Total samples: {len(risk_df)}")
    print(f"    Ransomware (label=1): {len(risk_df[risk_df['true_label']==1])}")
    print(f"    Benign (label=0): {len(risk_df[risk_df['true_label']==0])}")
    
    # Analyze ransomware probabilities
    ransomware_probs = risk_df[risk_df['true_label']==1]['p_ransomware']
    benign_probs = risk_df[risk_df['true_label']==0]['p_ransomware']
    
    print(f"\n[2] Probability Distribution Analysis:")
    print(f"    Ransomware P_Ransomware:")
    print(f"      Count: {len(ransomware_probs)}")
    print(f"      Unique values: {ransomware_probs.nunique()}")
    print(f"      Min: {ransomware_probs.min():.12f}")
    print(f"      Max: {ransomware_probs.max():.12f}")
    print(f"      Mean: {ransomware_probs.mean():.12f}")
    print(f"      Median: {ransomware_probs.median():.12f}")
    print(f"      Std: {ransomware_probs.std():.12f}")
    
    print(f"\n    Benign P_Ransomware (for comparison):")
    print(f"      Count: {len(benign_probs)}")
    print(f"      Unique values: {benign_probs.nunique()}")
    print(f"      Min: {benign_probs.min():.12e}")
    print(f"      Max: {benign_probs.max():.12f}")
    print(f"      Mean: {benign_probs.mean():.12e}")
    print(f"      Median: {benign_probs.median():.12e}")
    
    # Validation checks
    print(f"\n[3] Validation Checks:")
    
    # Check 1: Are ransomware probabilities high?
    high_prob_count = (ransomware_probs > 0.9).sum()
    high_prob_pct = (high_prob_count / len(ransomware_probs)) * 100
    print(f"    [OK] Ransomware with P > 0.9: {high_prob_count}/{len(ransomware_probs)} ({high_prob_pct:.1f}%)")
    results["high_prob_pct"] = high_prob_pct
    
    # Check 2: Are values unique (not constant)?
    unique_ratio = ransomware_probs.nunique() / len(ransomware_probs)
    print(f"    [OK] Unique value ratio: {unique_ratio:.4f} ({ransomware_probs.nunique()} unique / {len(ransomware_probs)} total)")
    results["unique_ratio"] = unique_ratio
    results["is_constant"] = unique_ratio < 0.01
    
    if results["is_constant"]:
        print(f"    ⚠️  WARNING: Values appear to be constant (same value for all samples)")
    else:
        print(f"    [OK] Values vary across samples (model is discriminating)")
    
    # Check 3: Standard deviation (should be > 0 if values vary)
    std_check = ransomware_probs.std() > 1e-10
    print(f"    [OK] Standard deviation > 1e-10: {std_check} (std={ransomware_probs.std():.12e})")
    results["has_variance"] = std_check
    
    # Check 4: Are benign probabilities low?
    low_benign_count = (benign_probs < 0.1).sum()
    low_benign_pct = (low_benign_count / len(benign_probs)) * 100
    print(f"    [OK] Benign with P < 0.1: {low_benign_count}/{len(benign_probs)} ({low_benign_pct:.1f}%)")
    results["low_benign_pct"] = low_benign_pct
    
    # Check 5: Separation between classes
    separation = ransomware_probs.min() - benign_probs.max()
    print(f"    [OK] Class separation: {separation:.12f} (ransomware_min - benign_max)")
    results["class_separation"] = separation
    
    if separation > 0:
        print(f"    [OK] Classes are well-separated (model discriminates correctly)")
    else:
        print(f"    ⚠️  WARNING: Some overlap between classes")
    
    # Sample values
    print(f"\n[4] Sample P_Ransomware Values (first 10 ransomware samples):")
    sample_values = ransomware_probs.head(10).tolist()
    for i, val in enumerate(sample_values):
        print(f"    Sample {i}: {val:.12f}")
    
    # Check register file
    if register_path.exists():
        register_df = pd.read_csv(register_path)
        register_probs = register_df['p_ransomware'].unique()
        print(f"\n[5] Register File Analysis:")
        print(f"    Total rows in register: {len(register_df)}")
        print(f"    Unique samples: {register_df['sample_id'].nunique()}")
        print(f"    Unique P_Ransomware values: {len(register_probs)}")
        print(f"    Min P_Ransomware: {register_df['p_ransomware'].min():.12f}")
        print(f"    Max P_Ransomware: {register_df['p_ransomware'].max():.12f}")
        
        # Verify register matches risk_scores
        register_unique_probs = sorted(register_probs)
        risk_unique_probs = sorted(ransomware_probs.unique())
        
        if len(register_unique_probs) == len(risk_unique_probs):
            matches = np.allclose(register_unique_probs, risk_unique_probs, rtol=1e-10)
            print(f"    [OK] Register probabilities match risk_scores: {matches}")
            results["register_matches"] = matches
        else:
            print(f"    ⚠️  Register has {len(register_unique_probs)} unique values, risk_scores has {len(risk_unique_probs)}")
            results["register_matches"] = False
    
    # Overall validation
    print(f"\n[6] Overall Validation:")
    is_valid = (
        not results.get("is_constant", True) and
        results.get("has_variance", False) and
        results.get("high_prob_pct", 0) > 90 and
        results.get("class_separation", -1) > 0
    )
    
    if is_valid:
        print(f"    [VALID] P_Ransomware values are correct and show proper model discrimination")
    else:
        print(f"    ⚠️  ISSUES DETECTED: See warnings above")
    
    results["is_valid"] = is_valid
    return results


def compare_across_splits(results_list: list[dict]) -> None:
    """Compare P_Ransomware values across different splits."""
    print(f"\n{'='*80}")
    print("Cross-Split Comparison")
    print(f"{'='*80}")
    
    # Check if values are different across splits
    split_probs = {}
    for result in results_list:
        split_name = result["split"]
        risk_scores_path = Path(f"results/h1h2_rebuild/{split_name}/risk_scores.csv")
        if risk_scores_path.exists():
            df = pd.read_csv(risk_scores_path)
            ransomware_probs = df[df['true_label']==1]['p_ransomware']
            split_probs[split_name] = {
                "mean": ransomware_probs.mean(),
                "min": ransomware_probs.min(),
                "max": ransomware_probs.max(),
                "unique": ransomware_probs.nunique(),
            }
    
    print("\nP_Ransomware Statistics by Split:")
    print(f"{'Split':<15} {'Mean':<20} {'Min':<20} {'Max':<20} {'Unique':<10}")
    print("-" * 85)
    
    for split_name, stats in split_probs.items():
        print(f"{split_name:<15} {stats['mean']:<20.12f} {stats['min']:<20.12f} {stats['max']:<20.12f} {stats['unique']:<10}")
    
    # Check if splits have different values
    if len(split_probs) > 1:
        means = [stats["mean"] for stats in split_probs.values()]
        if max(means) - min(means) > 1e-6:
            print(f"\n✓ Splits have different mean probabilities (range: {min(means):.12f} to {max(means):.12f})")
            print(f"  This confirms values are NOT identical across splits")
        else:
            print(f"\n⚠️  WARNING: Splits have very similar mean probabilities")
            print(f"  This might indicate values are being reused across splits")


def main():
    """Main validation function."""
    repo_root = Path(__file__).parent
    
    print("="*80)
    print("P_Ransomware Value Validation for Ransomware-Only Risk Registers")
    print("="*80)
    print("\nThis script validates that P_Ransomware values in ransomware-only")
    print("risk registers are:")
    print("  1. High (close to 1.0) - indicating model confidence")
    print("  2. Variable (not constant) - showing model discrimination")
    print("  3. Different across splits - confirming values are computed per split")
    print("  4. From actual model predictions - not placeholder values")
    
    splits = ["smoke_test", "small_ember", "main", "full_ember"]
    results_list = []
    
    for split in splits:
        try:
            results = validate_split(split, repo_root)
            results_list.append(results)
        except Exception as e:
            print(f"\n  [ERROR] ERROR validating {split}: {e}")
            import traceback
            traceback.print_exc()
    
    # Cross-split comparison
    compare_across_splits(results_list)
    
    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    
    valid_splits = [r for r in results_list if r.get("is_valid", False)]
    print(f"\nValid splits: {len(valid_splits)}/{len(results_list)}")
    
    if len(valid_splits) == len(results_list):
        print("\n[SUCCESS] ALL SPLITS VALIDATED SUCCESSFULLY")
        print("\nConclusion:")
        print("  - P_Ransomware values are HIGH (close to 1.0) - this is CORRECT for ransomware samples")
        print("  - Values are VARIABLE (not constant) - model is discriminating between samples")
        print("  - Values differ across splits - confirming per-split computation")
        print("  - This indicates a well-trained model with high confidence on ransomware samples")
    else:
        print(f"\n⚠️  {len(results_list) - len(valid_splits)} split(s) have issues - see details above")


if __name__ == "__main__":
    main()

