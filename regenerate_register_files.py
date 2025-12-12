#!/usr/bin/env python3
"""
Regenerate register files with proper model predictions.

This script:
1. Loads the working small_ember model
2. Loads EMBER data for full_ember and main
3. Generates proper predictions with variance
4. Creates new register files with varied probabilities
5. Validates the output
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import json
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from aicra.pipelines.data_loader import EMBERDataLoader
from aicra.config import Settings
from aicra.utils.validation import assert_non_constant_scores
from aicra.register import compute_register, write_register, Policy

print("=" * 80)
print("REGENERATE REGISTER FILES WITH PROPER MODEL PREDICTIONS")
print("=" * 80)

# Step 1: Check for working model
print("\n[1] Checking for working model...")
model_paths = [
    Path("models/lightgbm_small_ember.joblib"),
    Path("artifacts/models/lightgbm_small_ember.joblib"),
    Path("models/bagged_lightgbm.joblib"),
    Path("artifacts/models/bagged_lightgbm.joblib"),
]

model = None
model_path = None
for path in model_paths:
    if path.exists():
        try:
            model = joblib.load(path)
            model_path = path
            print(f"  Found model: {path}")
            break
        except Exception as e:
            print(f"  Could not load {path}: {e}")

if model is None:
    print("  ERROR: No working model found!")
    print("  Please run: python -m aicra.run-test --phase small_ember")
    print("  This will train and save the model needed for regeneration.")
    sys.exit(1)

# Step 2: Load settings
print("\n[2] Loading settings...")
try:
    settings = Settings()
    print(f"  Settings loaded")
except Exception as e:
    print(f"  Could not load settings: {e}")
    print("  Using default settings...")
    from aicra.config import get_settings
    settings = get_settings()

# Step 3: Check data directory
print("\n[3] Checking EMBER data directory...")
# Try ember2024_real first (full dataset), then fallback to other paths
data_dir = None
alt_paths = [Path("data/ember2024_real"), Path("data/ember2024"), settings.data_dir or Path("data/ember2024")]
for alt_path in alt_paths:
    if alt_path and alt_path.exists():
        jsonl_files = list(alt_path.glob("*.jsonl"))
        if jsonl_files:
            data_dir = alt_path
            print(f"  Found {len(jsonl_files)} JSONL files in {data_dir}")
            break

if data_dir is None:
    print(f"  ERROR: No EMBER data directory with JSONL files found.")
    print(f"  Tried: data/ember2024_real, data/ember2024, {settings.data_dir}")
    sys.exit(1)

# Step 4: Load data loader
print("\n[4] Initializing data loader...")
data_loader = EMBERDataLoader(settings)

# Step 5: Create default policy
print("\n[5] Creating default policy...")
policy = Policy(
    threshold=0.5,
    cost_false_negative=1.0,
    cost_false_positive=0.1,
    impact_default=100000.0
)

# Step 6: Regenerate full_ember register
print("\n[6] Regenerating full_ember register...")
output_path = None
try:
    # Load all data (no sampling) - use the full EMBER dataset
    print(f"  Loading data from {data_dir}...")
    features_df, labels_series, families_series, metadata = data_loader.load_ember_data(
        data_dir=str(data_dir),
        sample_size=None,  # Load all
        seed=42,
        phase="full_ember"
    )
    
    print(f"  Loaded {len(features_df)} samples")
    print(f"  Features: {len(features_df.columns)} columns")
    
    # Generate predictions
    print(f"  Generating predictions for {len(features_df):,} samples (this may take a while)...")
    print(f"  Processing in batches...")
    
    # Process in batches to show progress
    batch_size = 10000
    n_batches = (len(features_df) + batch_size - 1) // batch_size
    probabilities = np.zeros(len(features_df))
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(features_df))
        batch_features = features_df.iloc[start_idx:end_idx].values
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(batch_features)
            if y_prob.ndim == 1:
                y_prob = np.column_stack([1 - y_prob, y_prob])
            probabilities[start_idx:end_idx] = y_prob[:, 1]
        elif hasattr(model, 'predict'):
            probabilities[start_idx:end_idx] = model.predict(batch_features)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"    Processed batch {batch_idx + 1}/{n_batches} ({end_idx:,}/{len(features_df):,} samples)")
    
    # Validate predictions
    print(f"  Prediction stats: mean={probabilities.mean():.6f}, std={probabilities.std():.6f}, unique={np.unique(probabilities).size}")
    assert_non_constant_scores(probabilities, "full_ember", min_unique=5, min_std=1e-6)
    
    # Create register dataframe
    register_df = pd.DataFrame({
        "family": families_series.values,
        "probability": probabilities.clip(0.0, 1.0),
        "label": labels_series.values.astype(int),
    })
    
    if False:  # Removed fallback - we always use the model now
        # Fallback: Use small_ember register as reference distribution
        print("  Using small_ember register as reference distribution...")
        small_ember_path = Path("register/risk_register_small_ember.csv")
        if not small_ember_path.exists():
            raise FileNotFoundError(f"small_ember register not found: {small_ember_path}")
        
        df_small = pd.read_csv(small_ember_path)
        ref_probs = df_small["probability"].values
        
        print(f"  Reference distribution: mean={ref_probs.mean():.6f}, std={ref_probs.std():.6f}, unique={np.unique(ref_probs).size}")
        
        # Load existing full_ember register to get structure
        full_path = Path("register/risk_register_full.csv")
        if not full_path.exists():
            raise FileNotFoundError(f"full_ember register not found: {full_path}")
        
        register_df = pd.read_csv(full_path)
        n_rows = len(register_df)
        
        # Sample probabilities from reference distribution
        np.random.seed(42)
        new_probs = np.random.choice(ref_probs, size=n_rows, replace=True)
        register_df["probability"] = new_probs.clip(0.0, 1.0)
        
        print(f"  Generated {n_rows} probabilities from reference distribution")
        print(f"  New probability stats: mean={new_probs.mean():.6f}, std={new_probs.std():.6f}, unique={np.unique(new_probs).size}")
        assert_non_constant_scores(new_probs, "full_ember", min_unique=5, min_std=1e-6)
    
    # Recompute register fields (susceptibility, expected_loss, etc.)
    register_df = compute_register(register_df, policy)
    
    # Enrich with prescriptive controls (skip if it fails)
    try:
        from aicra.pipelines.policy import PolicyPipeline
        policy_pipeline = PolicyPipeline(settings, skip_mlflow=True)
        register_df = policy_pipeline.enrich_register_with_controls(register_df)
    except Exception as e:
        print(f"  WARNING: Could not enrich with controls: {e}")
        print(f"  Continuing without prescriptive controls...")
    
    # Write register
    register_name = "risk_register_full"
    write_register(register_df, name=register_name)
    
    # Validate output
    output_path = Path("register") / f"{register_name}.csv"
    if output_path.exists():
        df_check = pd.read_csv(output_path)
        probs_check = df_check["probability"]
        print(f"  Created: {output_path}")
        print(f"    Rows: {len(register_df)}, Std: {probs_check.std():.6f}, Unique: {probs_check.nunique()}")
        assert_non_constant_scores(probs_check, "full_ember_register", min_unique=5, min_std=1e-6)
        print(f"  Validation passed!")
    else:
        print(f"  WARNING: Register file not found at expected path: {output_path}")
        output_path = None
    
except Exception as e:
    print(f"  ERROR: Error regenerating full_ember: {e}")
    import traceback
    traceback.print_exc()
    output_path = None

# Step 7: Regenerate main register (sample from full_ember)
print("\n[7] Regenerating main register (sampling from full_ember)...")
try:
    # Sample 10,000 rows from the full register we just created
    if output_path and output_path.exists():
        df_full = pd.read_csv(output_path)
        df_main = df_full.sample(n=10000, random_state=42).reset_index(drop=True)
        
        # Validate probabilities
        probs_main = df_main["probability"]
        print(f"  Sampled {len(df_main)} rows")
        print(f"  Probability stats: mean={probs_main.mean():.6f}, std={probs_main.std():.6f}, unique={probs_main.nunique()}")
        assert_non_constant_scores(probs_main, "main", min_unique=5, min_std=1e-6)
        
        # Write main register
        register_name = "risk_register_main"
        write_register(df_main, name=register_name)
        
        # Validate output
        main_output_path = Path("register") / f"{register_name}.csv"
        if main_output_path.exists():
            df_check = pd.read_csv(main_output_path)
            probs_check = df_check["probability"]
            print(f"  Created: {main_output_path}")
            print(f"    Rows: {len(df_main)}, Std: {probs_check.std():.6f}, Unique: {probs_check.nunique()}")
            assert_non_constant_scores(probs_check, "main_register", min_unique=5, min_std=1e-6)
            print(f"  Validation passed!")
        else:
            print(f"  WARNING: Register file not found at expected path: {main_output_path}")
    else:
        print(f"  ERROR: Cannot create main register - full_ember register not found")
    
except Exception as e:
    print(f"  ERROR: Error regenerating main: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("REGISTER REGENERATION COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("  1. Regenerate H3 splits: python create_ember_splits.py")
print("  2. Re-run H3 evaluation: python -m aicra.experiments.h3_evaluation --config config/h3_splits.yaml")
print("=" * 80)

