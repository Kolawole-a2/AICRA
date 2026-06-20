#!/usr/bin/env python3
"""
Regenerate main and full_ember risk scores using the working small_ember model.

This script:
1. Loads the working small_ember model
2. Loads EMBER data for main and full_ember
3. Generates proper predictions with variance
4. Creates new risk_scores.csv files
5. Validates the output
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aicra.config import Settings
from aicra.pipelines.data_loader import EMBERDataLoader
from aicra.utils.validation import assert_non_constant_scores, validate_risk_scores_file

print("=" * 80)
print("REGENERATE MAIN AND FULL_EMBER RISK SCORES")
print("=" * 80)

# Step 1: Check for working model
print("\n[1] Checking for working model...")
model_paths = [
    Path("models/lightgbm_small_ember.joblib"),
    Path("artifacts/models/lightgbm_small_ember.joblib"),
]

model = None
model_path = None
for path in model_paths:
    if path.exists():
        try:
            model = joblib.load(path)
            model_path = path
            print(f"  ✓ Found model: {path}")
            break
        except Exception as e:
            print(f"  ⚠️  Could not load {path}: {e}")

if model is None:
    print("  ❌ No working model found!")
    print("  Please run: python -m aicra.run-test --phase small_ember")
    print("  This will train and save the model needed for regeneration.")
    sys.exit(1)

# Step 2: Load settings
print("\n[2] Loading settings...")
try:
    settings = Settings()
    print("  ✓ Settings loaded")
except Exception as e:
    print(f"  ⚠️  Could not load settings: {e}")
    print("  Using default settings...")
    from aicra.config import get_settings

    settings = get_settings()

# Step 3: Check data directory
print("\n[3] Checking EMBER data directory...")
data_dir = settings.data_dir or Path("data/ember2024")
if not data_dir.exists():
    print(f"  ⚠️  Data directory not found: {data_dir}")
    print(f"  Please provide EMBER-2024 JSONL files in: {data_dir}")
    print("  Or set data_dir in config/settings.yaml")
    sys.exit(1)

jsonl_files = list(Path(data_dir).glob("*.jsonl"))
if not jsonl_files:
    print(f"  ⚠️  No JSONL files found in: {data_dir}")
    sys.exit(1)

print(f"  ✓ Found {len(jsonl_files)} JSONL files")

# Step 4: Load data loader
print("\n[4] Initializing data loader...")
data_loader = EMBERDataLoader(settings)

# Step 5: Regenerate main split (10,000 samples)
print("\n[5] Regenerating main split...")
try:
    # Load 10,000 samples
    features_df, labels_series, families_series, metadata = data_loader.load_ember_data(
        data_dir=str(data_dir), sample_size=10000, seed=42, phase="main"
    )

    print(f"  Loaded {len(features_df)} samples")
    print(f"  Features: {len(features_df.columns)} columns")

    # Generate predictions
    print("  Generating predictions...")
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(features_df.values)
        if y_prob.ndim == 1:
            y_prob = np.column_stack([1 - y_prob, y_prob])
        risk_scores = y_prob[:, 1]  # Positive class probability
    elif hasattr(model, "predict"):
        # LightGBM Booster
        risk_scores = model.predict(features_df.values)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    # Validate predictions
    print(
        f"  Prediction stats: mean={risk_scores.mean():.6f}, std={risk_scores.std():.6f}, unique={np.unique(risk_scores).size}"
    )
    assert_non_constant_scores(risk_scores, "main", min_unique=5, min_std=1e-6)

    # Create H3 format DataFrame
    h3_main = pd.DataFrame(
        {
            "asset_id": [f"asset_{i:04d}" for i in range(len(features_df))],
            "risk_score": risk_scores.clip(0.0, 1.0),
            "predicted_label": (risk_scores >= 0.5).astype(int),
            "true_label": labels_series.values.astype(int),
            "technique_id": "T1486",  # Default, will be enriched later if needed
        }
    )

    # Extract technique IDs from families if possible
    # (This is a simplified version - you may need to enrich with actual ATT&CK mappings)
    h3_main["technique_id"] = (
        h3_main["technique_id"].fillna("T1486").replace("", "T1486")
    )

    # Save
    output_path = Path("results/main/risk_scores.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h3_main.to_csv(output_path, index=False)

    # Validate output
    validate_risk_scores_file(output_path, "main")
    print(f"  ✓ Created: {output_path}")
    print(
        f"    Rows: {len(h3_main)}, Std: {h3_main['risk_score'].std():.6f}, Unique: {h3_main['risk_score'].nunique()}"
    )

except Exception as e:
    print(f"  ❌ Error regenerating main: {e}")
    import traceback

    traceback.print_exc()

# Step 6: Regenerate full_ember split (all data)
print("\n[6] Regenerating full_ember split...")
try:
    # Load all data (no sampling)
    features_df, labels_series, families_series, metadata = data_loader.load_ember_data(
        data_dir=str(data_dir),
        sample_size=None,  # Load all
        seed=42,
        phase="full_ember",
    )

    print(f"  Loaded {len(features_df)} samples")
    print(f"  Features: {len(features_df.columns)} columns")

    # Generate predictions
    print("  Generating predictions (this may take a while for large datasets)...")
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(features_df.values)
        if y_prob.ndim == 1:
            y_prob = np.column_stack([1 - y_prob, y_prob])
        risk_scores = y_prob[:, 1]
    elif hasattr(model, "predict"):
        risk_scores = model.predict(features_df.values)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    # Validate predictions
    print(
        f"  Prediction stats: mean={risk_scores.mean():.6f}, std={risk_scores.std():.6f}, unique={np.unique(risk_scores).size}"
    )
    assert_non_constant_scores(risk_scores, "full_ember", min_unique=5, min_std=1e-6)

    # Create H3 format DataFrame
    h3_full = pd.DataFrame(
        {
            "asset_id": [f"asset_{i:04d}" for i in range(len(features_df))],
            "risk_score": risk_scores.clip(0.0, 1.0),
            "predicted_label": (risk_scores >= 0.5).astype(int),
            "true_label": labels_series.values.astype(int),
            "technique_id": "T1486",  # Default
        }
    )

    h3_full["technique_id"] = (
        h3_full["technique_id"].fillna("T1486").replace("", "T1486")
    )

    # Save
    output_path = Path("results/full_ember/risk_scores.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h3_full.to_csv(output_path, index=False)

    # Validate output
    validate_risk_scores_file(output_path, "full_ember")
    print(f"  ✓ Created: {output_path}")
    print(
        f"    Rows: {len(h3_full)}, Std: {h3_full['risk_score'].std():.6f}, Unique: {h3_full['risk_score'].nunique()}"
    )

except Exception as e:
    print(f"  ❌ Error regenerating full_ember: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("✓ REGENERATION COMPLETE")
print("=" * 80)
print("\nNext steps:")
print("  1. Verify the new risk_scores.csv files have proper variance")
print("  2. Run H3 evaluation: python -m aicra.experiments.h3_evaluation")
print("=" * 80)
