#!/usr/bin/env python3
"""
Regenerate risk registers with CORRECT probability values from model predictions.

This script:
1. Loads the trained model
2. Loads test data
3. Generates predictions using model.predict_proba()
4. Applies calibration if available
5. Regenerates registers with correct probabilities
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
import traceback
import sys
import os
from pathlib import Path

# Add parent directory to path (scripts/ -> repo root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from aicra.config import Settings, get_settings
from aicra.register import compute_register, write_register, Policy
from aicra.pipelines.data_loader import EMBERDataLoader
from aicra.pipelines.policy import PolicyPipeline
from aicra.utils.validation import assert_non_constant_scores
from aicra.utils.register_validation import validate_register_probabilities, validate_susceptibility_buckets


def main():
    parser = argparse.ArgumentParser(description="Regenerate risk registers with correct probabilities")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()
    
    debug = args.debug
    
    if debug:
        print("[DEBUG] Debug mode enabled", file=sys.stdout)
    
    print("=" * 80, file=sys.stdout)
    print("REGENERATING RISK REGISTERS WITH CORRECT MODEL PREDICTIONS", file=sys.stdout)
    print("=" * 80, file=sys.stdout)
    
    # Print working directory and validate inputs
    cwd = os.getcwd()
    print(f"\n[0] Working directory: {cwd}", file=sys.stdout)
    if debug:
        print(f"[DEBUG] Absolute working directory: {os.path.abspath(cwd)}", file=sys.stdout)
    
    # Step 1: Find working model
    print("\n[1] Finding trained model...", file=sys.stdout)
    model_paths = [
        Path("models/lightgbm_small_ember.joblib"),
        Path("artifacts/models/lightgbm_small_ember.joblib"),
        Path("models/bagged_lightgbm.joblib"),
        Path("artifacts/models/bagged_lightgbm.joblib"),
    ]
    
    # Print and validate model paths
    if debug:
        print(f"[DEBUG] Checking model paths:", file=sys.stdout)
        for path in model_paths:
            abs_path = os.path.abspath(path)
            exists = path.exists()
            print(f"[DEBUG]   {path} -> {abs_path} (exists: {exists})", file=sys.stdout)

    model = None
    model_path = None
    for path in model_paths:
        if path.exists():
            try:
                abs_path = os.path.abspath(path)
                if debug:
                    print(f"[DEBUG] Loading model from: {abs_path}", file=sys.stdout)
                model = joblib.load(path)
                model_path = path
                print(f"  [OK] Found model: {path}", file=sys.stdout)
                break
            except Exception as e:
                print(f"  [ERROR] Could not load {path}: {e}", file=sys.stderr)
    
    if model is None:
        error_msg = "ERROR: No working model found!"
        print(f"  {error_msg}", file=sys.stderr)
        print("  Please run: python -m aicra.run-test --phase small_ember", file=sys.stderr)
        print("  This will train and save the model needed for regeneration.", file=sys.stderr)
        # Assert model path exists before failing
        missing_paths = [str(p) for p in model_paths if not p.exists()]
        raise FileNotFoundError(f"No model found. Checked paths: {missing_paths}")

    # Step 2: Load settings
    print("\n[2] Loading settings...", file=sys.stdout)
    try:
        settings = Settings()
        print(f"  [OK] Settings loaded", file=sys.stdout)
    except Exception as e:
        print(f"  [ERROR] Could not load settings: {e}", file=sys.stderr)
        print("  Using default settings...", file=sys.stdout)
        settings = get_settings()
    
    # Step 3: Load policy
    print("\n[3] Loading policy...", file=sys.stdout)
    def load_policy() -> Policy:
        """Load policy from file or use defaults."""
        policy_paths = [
            Path("policies/policy.json"),
            Path("artifacts/policy.json"),
            Path("artifacts/policy_small_ember.json"),
            Path("artifacts/policy_full.json"),
        ]
        
        if debug:
            print(f"[DEBUG] Checking policy paths:", file=sys.stdout)
            for policy_path in policy_paths:
                abs_path = os.path.abspath(policy_path)
                exists = policy_path.exists()
                print(f"[DEBUG]   {policy_path} -> {abs_path} (exists: {exists})", file=sys.stdout)
        
        for policy_path in policy_paths:
            if policy_path.exists():
                try:
                    abs_path = os.path.abspath(policy_path)
                    if debug:
                        print(f"[DEBUG] Loading policy from: {abs_path}", file=sys.stdout)
                    with open(policy_path, encoding="utf-8") as f:
                        policy_data = json.load(f)
                    print(f"  [OK] Loaded policy from {policy_path}", file=sys.stdout)
                    return Policy(
                        threshold=policy_data.get("threshold", 0.5),
                        cost_false_negative=policy_data.get("cost_false_negative", 1.0),
                        cost_false_positive=policy_data.get("cost_false_positive", 0.1),
                        impact_default=policy_data.get("impact_default", 5000000.0),
                    )
                except Exception as e:
                    print(f"  [ERROR] Could not load policy from {policy_path}: {e}", file=sys.stderr)
                    continue
        
        # Default policy with $5M impact
        print("  [OK] Using default policy with $5M impact", file=sys.stdout)
        return Policy(
            threshold=0.5,
            cost_false_negative=1.0,
            cost_false_positive=0.1,
            impact_default=5000000.0
        )
    
    policy = load_policy()
    print(f"  Policy: Threshold={policy.threshold}, Impact=${policy.impact_default:,.2f}", file=sys.stdout)
    
    # Step 4: Find data directory
    print("\n[4] Finding data directory...", file=sys.stdout)
    try:
        from aicra.utils.data_paths import get_ember2024_dir
        data_dir = get_ember2024_dir()
        jsonl_files = list(data_dir.glob("*.jsonl"))
        if jsonl_files:
            abs_data_dir = os.path.abspath(data_dir)
            print(f"  [OK] Found data directory: {data_dir} ({len(jsonl_files)} JSONL files)", file=sys.stdout)
            if debug:
                print(f"[DEBUG] Absolute data directory: {abs_data_dir}", file=sys.stdout)
                print(f"[DEBUG] JSONL files:", file=sys.stdout)
                for f in jsonl_files[:5]:  # Show first 5
                    print(f"[DEBUG]   {os.path.abspath(f)}", file=sys.stdout)
                if len(jsonl_files) > 5:
                    print(f"[DEBUG]   ... and {len(jsonl_files) - 5} more", file=sys.stdout)
        else:
            print(f"  [WARNING] Data directory found but no JSONL files: {data_dir}", file=sys.stderr)
            data_dir = None
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}", file=sys.stderr)
        data_dir = None
    
    # Fallback: try other common locations
    if data_dir is None:
        possible_data_dirs = [
            settings.data_dir / "ember2024_real",
            settings.data_dir / "ember2024",
            Path("data/ember2024_real"),
            Path("data/ember2024"),
        ]
        
        if debug:
            print(f"[DEBUG] Trying fallback data directories:", file=sys.stdout)
            for d in possible_data_dirs:
                abs_d = os.path.abspath(d)
                exists = d.exists()
                jsonl_count = len(list(d.glob("*.jsonl"))) if exists else 0
                print(f"[DEBUG]   {d} -> {abs_d} (exists: {exists}, jsonl: {jsonl_count})", file=sys.stdout)
        
        for d in possible_data_dirs:
            if d.exists():
                jsonl_files = list(d.glob("*.jsonl"))
                if jsonl_files:
                    data_dir = d
                    abs_data_dir = os.path.abspath(data_dir)
                    print(f"  [OK] Found data directory: {data_dir} ({len(jsonl_files)} JSONL files)", file=sys.stdout)
                    if debug:
                        print(f"[DEBUG] Absolute data directory: {abs_data_dir}", file=sys.stdout)
                    break
    
    if data_dir is None:
        error_msg = "Data directory not found!"
        print(f"  [ERROR] {error_msg}", file=sys.stderr)
        print(f"  Please ensure EMBER data is available.", file=sys.stderr)
        print(f"  You can set AICRA_EMBER2024_DIR environment variable.", file=sys.stderr)
        raise FileNotFoundError("Data directory not found. Checked paths but none contained JSONL files.")
    
    # Initialize data loader
    data_loader = EMBERDataLoader(settings)

    # Step 5: Regenerate registers with correct probabilities
    def regenerate_register_with_model(phase: str, sample_size: int | None = None) -> bool:
        """Regenerate a register file with correct model predictions."""
        print(f"\n{'='*80}", file=sys.stdout)
        print(f"REGENERATING: {phase.upper()}", file=sys.stdout)
        print(f"{'='*80}", file=sys.stdout)
        
        try:
            # Load test data
            print(f"\n  [1] Loading test data from {data_dir}...", file=sys.stdout)
            if debug:
                abs_data_dir = os.path.abspath(data_dir)
                print(f"[DEBUG] Loading from absolute path: {abs_data_dir}", file=sys.stdout)
            
            features_df, labels_series, families_series, metadata = data_loader.load_ember_data(
                data_dir=str(data_dir),
                sample_size=sample_size,
                seed=42,
                phase=phase
            )
            
            print(f"      Loaded {len(features_df)} samples", file=sys.stdout)
            print(f"      Label distribution: {labels_series.value_counts().to_dict()}", file=sys.stdout)
            
            # VALIDATION A: Label sanity check
            print(f"\n  [VALIDATION A] Label sanity check...", file=sys.stdout)
            print(f"      DataFrame shape: {features_df.shape}", file=sys.stdout)
            print(f"      Labels shape: {labels_series.shape}", file=sys.stdout)
            
            # Check label column presence (labels_series is already the label column)
            label_values = labels_series.values
            unique_labels = np.unique(label_values)
            print(f"      Unique label values: {unique_labels}", file=sys.stdout)
            print(f"      Label value counts: {pd.Series(label_values).value_counts().to_dict()}", file=sys.stdout)
            
            # Validate labels are {0, 1}
            allowed_labels = {0, 1}
            if not set(unique_labels).issubset(allowed_labels):
                raise ValueError(
                    f"Invalid label values found: {unique_labels}. "
                    f"Allowed values are {allowed_labels}. "
                    f"Check that labels are binary (0=benign, 1=ransomware)."
                )
            
            label_counts = pd.Series(label_values).value_counts()
            print(f"      [OK] Label validation passed: {len(label_counts)} classes", file=sys.stdout)
            for label_val, count in label_counts.items():
                print(f"        Label {label_val}: {count} samples", file=sys.stdout)
        
            # Generate predictions
            print(f"\n  [2] Generating model predictions...", file=sys.stdout)
            y_prob = model.predict_proba(features_df.values)
            
            # Ensure y_prob is 2D and extract positive class probabilities
            if y_prob.ndim == 1:
                y_prob = np.column_stack([1 - y_prob, y_prob])
            y_prob_raw = y_prob[:, 1]  # Raw p(ransomware)
            
            print(f"      Raw predictions: min={y_prob_raw.min():.6f}, max={y_prob_raw.max():.6f}, mean={y_prob_raw.mean():.6f}", file=sys.stdout)
            if debug:
                print(f"[DEBUG] Raw prediction shape: {y_prob_raw.shape}", file=sys.stdout)
                print(f"[DEBUG] Raw prediction dtype: {y_prob_raw.dtype}", file=sys.stdout)
            
            # Apply calibration if available
            calibrator_path = settings.models_dir / f"calibrator_{phase}.joblib"
            if calibrator_path.exists():
                abs_calibrator_path = os.path.abspath(calibrator_path)
                if debug:
                    print(f"[DEBUG] Loading calibrator from: {abs_calibrator_path}", file=sys.stdout)
                print(f"\n  [3] Applying calibration from {calibrator_path}...", file=sys.stdout)
                calibrator = joblib.load(calibrator_path)
                probabilities = calibrator.transform(y_prob_raw)
                print(f"      Calibrated probabilities: min={probabilities.min():.6f}, max={probabilities.max():.6f}, mean={probabilities.mean():.6f}", file=sys.stdout)
            else:
                if debug:
                    print(f"[DEBUG] No calibrator found at: {os.path.abspath(calibrator_path)}", file=sys.stdout)
                print(f"\n  [3] No calibrator found, using raw probabilities...", file=sys.stdout)
                probabilities = y_prob_raw.clip(0.0, 1.0)
            
            # VALIDATION B: Probability sanity check
            print(f"\n  [VALIDATION B] Probability sanity check...", file=sys.stdout)
            print(f"      Probability array shape: {probabilities.shape}", file=sys.stdout)
            print(f"      Probability dtype: {probabilities.dtype}", file=sys.stdout)
            
            # Check probability range [0, 1]
            prob_min = float(probabilities.min())
            prob_max = float(probabilities.max())
            print(f"      Probability range: [{prob_min:.6f}, {prob_max:.6f}]", file=sys.stdout)
            
            if prob_min < 0.0 or prob_max > 1.0:
                raise ValueError(
                    f"Probabilities out of range [0, 1]. "
                    f"Min: {prob_min:.6f}, Max: {prob_max:.6f}. "
                    f"Probabilities must be in [0, 1]."
                )
            
            print(f"      [OK] Probability range validation passed", file=sys.stdout)
            
            # Validate predictions
            assert_non_constant_scores(probabilities, phase, min_unique=5, min_std=1e-6)
        
            # Check prediction quality by label
            benign_mask = labels_series.values == 0
            ransomware_mask = labels_series.values == 1
            
            benign_probs = probabilities[benign_mask]
            ransomware_probs = probabilities[ransomware_mask]
            
            # VALIDATION C: Probability direction sanity check (CRITICAL)
            print(f"\n  [VALIDATION C] Probability direction sanity check (CRITICAL)...", file=sys.stdout)
            mean_prob_neg = float(benign_probs.mean())
            mean_prob_pos = float(ransomware_probs.mean())
            print(f"      Mean probability for benign (label=0): {mean_prob_neg:.6f}", file=sys.stdout)
            print(f"      Mean probability for ransomware (label=1): {mean_prob_pos:.6f}", file=sys.stdout)
            
            if mean_prob_neg > mean_prob_pos:
                error_msg = (
                    f"Probability direction appears inverted (benign has higher mean probability than ransomware). "
                    f"Benign mean: {mean_prob_neg:.6f}, Ransomware mean: {mean_prob_pos:.6f}. "
                    f"Check you are using P(ransomware)=proba[:,1] not P(benign)."
                )
                print(f"      [ERROR] {error_msg}", file=sys.stderr)
                raise ValueError(error_msg)
            
            print(f"      [OK] Probability direction validation passed", file=sys.stdout)
            
            print(f"\n  [4] Prediction quality check:", file=sys.stdout)
            print(f"      Benign (label=0): mean={benign_probs.mean():.6f}, median={np.median(benign_probs):.6f}, max={benign_probs.max():.6f}", file=sys.stdout)
            print(f"      Ransomware (label=1): mean={ransomware_probs.mean():.6f}, median={np.median(ransomware_probs):.6f}, min={ransomware_probs.min():.6f}", file=sys.stdout)
            
            # Check for problematic cases
            benign_high = np.sum(benign_probs > 0.66)
            ransomware_low = np.sum(ransomware_probs < 0.33)
            
            print(f"      Benign with HIGH prob (>0.66): {benign_high}/{len(benign_probs)} ({benign_high/len(benign_probs)*100:.1f}%)", file=sys.stdout)
            print(f"      Ransomware with LOW prob (<0.33): {ransomware_low}/{len(ransomware_probs)} ({ransomware_low/len(ransomware_probs)*100:.1f}%)", file=sys.stdout)
            
            if benign_high > len(benign_probs) * 0.1:
                print(f"      [WARNING] {benign_high/len(benign_probs)*100:.1f}% of benign samples have high probability!", file=sys.stderr)
            if ransomware_low > len(ransomware_probs) * 0.1:
                print(f"      [WARNING] {ransomware_low/len(ransomware_probs)*100:.1f}% of ransomware samples have low probability!", file=sys.stderr)
        
            # Create register dataframe
            print(f"\n  [5] Creating register dataframe...", file=sys.stdout)
            register_df = pd.DataFrame({
                "family": families_series.values,
                "probability": probabilities.clip(0.0, 1.0),
                "label": labels_series.values.astype(int),
            })
            
            if debug:
                print(f"[DEBUG] Register dataframe shape: {register_df.shape}", file=sys.stdout)
                print(f"[DEBUG] Register dataframe columns: {list(register_df.columns)}", file=sys.stdout)
                print(f"[DEBUG] Register dataframe head(3):", file=sys.stdout)
                print(register_df[["family", "probability", "label"]].head(3).to_string(), file=sys.stdout)
            
            # Compute register (this will calculate susceptibility, buckets, expected_loss)
            print(f"\n  [6] Computing register fields...", file=sys.stdout)
            register_df = compute_register(register_df, policy)
            
            # VALIDATION D: Bucketing sanity check
            print(f"\n  [VALIDATION D] Bucketing sanity check...", file=sys.stdout)
            # Bucketing thresholds from compute_register: Low [0, 0.33], Medium (0.33, 0.66], High (0.66, 1.0]
            thresholds = [0.0, 0.33, 0.66, 1.0]
            print(f"      Bucketing thresholds: Low [0.0, 0.33], Medium (0.33, 0.66], High (0.66, 1.0]", file=sys.stdout)
            print(f"      Comparison: Low uses <=0.33, Medium uses >0.33 and <=0.66, High uses >0.66", file=sys.stdout)
            
            if "susceptibility_bucket" not in register_df.columns:
                raise ValueError("Missing 'susceptibility_bucket' column after compute_register")
            
            # Cross-tabulation: bucket counts overall and by label
            bucket_counts = register_df["susceptibility_bucket"].value_counts()
            print(f"      Bucket counts (overall):", file=sys.stdout)
            for bucket, count in bucket_counts.items():
                print(f"        {bucket}: {count}", file=sys.stdout)
            
            # Cross-tab by label
            if "label" in register_df.columns:
                cross_tab = pd.crosstab(register_df["label"], register_df["susceptibility_bucket"], margins=True)
                print(f"      Bucket counts by label (cross-tabulation):", file=sys.stdout)
                print(cross_tab.to_string(), file=sys.stdout)
                
                # Check for benign in High bucket
                benign_df = register_df[register_df["label"] == 0]
                benign_high_count = (benign_df["susceptibility_bucket"] == "High").sum()
                benign_total = len(benign_df)
                benign_high_pct = (benign_high_count / benign_total * 100) if benign_total > 0 else 0.0
                
                print(f"      Benign samples in High bucket: {benign_high_count}/{benign_total} ({benign_high_pct:.2f}%)", file=sys.stdout)
                
                if benign_high_pct > 2.0:
                    print(f"      [WARNING] {benign_high_pct:.2f}% of benign samples are in High bucket (>2% threshold)", file=sys.stderr)
                    if debug:
                        # Show top 20 benign rows by probability
                        benign_high_rows = benign_df[benign_df["susceptibility_bucket"] == "High"].nlargest(20, "probability")
                        print(f"[DEBUG] Top 20 benign rows in High bucket by probability:", file=sys.stdout)
                        print(benign_high_rows[["family", "probability", "susceptibility", "susceptibility_bucket"]].to_string(), file=sys.stdout)
            
            print(f"      [OK] Bucketing validation completed", file=sys.stdout)
        
            # Enrich with prescriptive controls
            try:
                print(f"\n  [7] Enriching with prescriptive controls...", file=sys.stdout)
                policy_pipeline = PolicyPipeline(settings, skip_mlflow=True)
                register_df = policy_pipeline.enrich_register_with_controls(register_df)
            except Exception as e:
                print(f"      WARNING: Could not enrich with controls: {e}", file=sys.stderr)
            
            # Validate register probabilities
            print(f"\n  [8] Validating register probabilities...", file=sys.stdout)
            try:
                validate_register_probabilities(register_df, phase)
                print(f"      [OK] Probability validation passed", file=sys.stdout)
            except ValueError as e:
                print(f"      [ERROR] Probability validation failed: {e}", file=sys.stderr)
                raise
            
            # Validate susceptibility buckets
            try:
                validate_susceptibility_buckets(register_df, phase)
                print(f"      [OK] Bucket validation passed", file=sys.stdout)
            except ValueError as e:
                print(f"      [ERROR] Bucket validation failed: {e}", file=sys.stderr)
                raise
            
            # Verify final results
            print(f"\n  [9] Final register statistics...", file=sys.stdout)
            benign_final = register_df[register_df["label"] == 0]
            ransomware_final = register_df[register_df["label"] == 1]
            
            print(f"      Benign - Low bucket: {(benign_final['susceptibility_bucket'] == 'Low').sum()}/{len(benign_final)}", file=sys.stdout)
            print(f"      Benign - Medium bucket: {(benign_final['susceptibility_bucket'] == 'Medium').sum()}/{len(benign_final)}", file=sys.stdout)
            print(f"      Benign - High bucket: {(benign_final['susceptibility_bucket'] == 'High').sum()}/{len(benign_final)}", file=sys.stdout)
            print(f"      Ransomware - Low bucket: {(ransomware_final['susceptibility_bucket'] == 'Low').sum()}/{len(ransomware_final)}", file=sys.stdout)
            print(f"      Ransomware - Medium bucket: {(ransomware_final['susceptibility_bucket'] == 'Medium').sum()}/{len(ransomware_final)}", file=sys.stdout)
            print(f"      Ransomware - High bucket: {(ransomware_final['susceptibility_bucket'] == 'High').sum()}/{len(ransomware_final)}", file=sys.stdout)
            
            # Save register
            print(f"\n  [10] Saving register...", file=sys.stdout)
            latest_path, archived_path = write_register(register_df, name=phase)
            print(f"      [OK] Register saved", file=sys.stdout)
            print(f"        Latest: {os.path.abspath(latest_path)}", file=sys.stdout)
            print(f"        Archived: {os.path.abspath(archived_path)}", file=sys.stdout)
            # Also check register_dir (write_register writes there too)
            register_csv = settings.register_dir / f"{phase}.csv"
            register_json = settings.register_dir / f"{phase}.json"
            if register_csv.exists():
                print(f"        Register CSV: {os.path.abspath(register_csv)}", file=sys.stdout)
            if register_json.exists():
                print(f"        Register JSON: {os.path.abspath(register_json)}", file=sys.stdout)
            
            return True
            
        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return False

    # Regenerate registers
    regeneration_results = {}
    
    # Small EMBER (2,000 samples)
    regeneration_results["small_ember"] = regenerate_register_with_model("small_ember", sample_size=2000)
    
    # Full EMBER (all samples)
    regeneration_results["full_ember"] = regenerate_register_with_model("full_ember", sample_size=None)
    
    # Summary
    print("\n" + "=" * 80, file=sys.stdout)
    print("REGENERATION SUMMARY", file=sys.stdout)
    print("=" * 80, file=sys.stdout)
    for phase, success in regeneration_results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"  {phase}: {status}", file=sys.stdout)
    
    successful = sum(1 for s in regeneration_results.values() if s)
    total = len(regeneration_results)
    
    if successful == total:
        print(f"\n[SUCCESS] All {successful}/{total} registers regenerated with correct model predictions!", file=sys.stdout)
        print("\nThe registers now have:", file=sys.stdout)
        print("  - Correct probability values from model.predict_proba()", file=sys.stdout)
        print("  - Calibrated probabilities (if calibrator available)", file=sys.stdout)
        print("  - Proper susceptibility scores (probability.clip(0,1))", file=sys.stdout)
        print("  - Correct susceptibility buckets (Low/Medium/High)", file=sys.stdout)
        print("  - Correct expected loss (susceptibility × $5,000,000)", file=sys.stdout)
        print("\nBenign samples should now have LOW susceptibility,", file=sys.stdout)
        print("and ransomware samples should have HIGH susceptibility.", file=sys.stdout)
    else:
        print(f"\n[WARNING] {total - successful}/{total} register(s) failed to regenerate", file=sys.stderr)
    
    print("=" * 80, file=sys.stdout)
    
    # Print output locations
    print("\n[OUTPUT LOCATIONS]", file=sys.stdout)
    register_dir = Path("register")
    artifacts_dir = settings.artifacts_dir
    print(f"  Register directory: {os.path.abspath(register_dir)}", file=sys.stdout)
    print(f"  Artifacts directory: {os.path.abspath(artifacts_dir)}", file=sys.stdout)
    if register_dir.exists():
        register_files = list(register_dir.glob("risk_register_*.csv"))
        if register_files:
            print(f"  Generated register files:", file=sys.stdout)
            for f in register_files:
                print(f"    -> {os.path.abspath(f)}", file=sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
