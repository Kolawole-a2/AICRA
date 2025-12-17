#!/usr/bin/env python3
"""
H1/H2 Rebuild – Phase 2: Train Model, Calibrate, and Emit Per-Sample Risk Scores

This script:
- Loads EMBER-2024 using `load_ember_2024(time_ordered=True)`
- Trains a LightGBM model via `TrainingPipeline` (H1-style)
- Calibrates probabilities via `CalibrationPipeline` (H2-style, auto method)
- Generates calibrated per-sample probabilities for the time-ordered test set
- Slices the test set into four evaluation splits:
    - smoke_test, small_ember, main, full_ember
- Writes one `risk_scores.csv` per split under:
    - results/h1h2_rebuild/<split>/risk_scores.csv

Schema (`risk_scores.csv`):
- sample_id: matches the ID in the manifest (same convention)
- true_label: 0 = benign, 1 = ransomware
- family: malware family name (lower-cased)
- p_ransomware: calibrated probability of ransomware
- predicted_label: 1 if p_ransomware >= global_cost_optimal_threshold, else 0
- split_part: "test" (for future compatibility)

Notes:
- This pipeline is separate from canonical H1/H2 experiments and does not modify
  `results/H1_classification/*` or `results/H2_calibration_thresholds/*`.
- H3 code and results are not touched; we only consume EMBER-2024 and AICRA
  training/calibration pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from aicra.config import Settings
from aicra.core.data import Dataset, load_ember_2024
from aicra.core.evaluation import cost_sensitive_threshold
from aicra.pipelines.calibration import CalibrationPipeline
from aicra.pipelines.training import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _make_split_indices(n: int) -> dict[str, np.ndarray]:
    """Return row index selections for each split based on full dataset size n."""
    full_idx = np.arange(n)
    main_n = min(10_000, n)
    small_n = min(2_000, main_n)
    smoke_n = min(200, small_n)

    return {
        "full_ember": full_idx,
        "main": np.arange(main_n),
        "small_ember": np.arange(small_n),
        "smoke_test": np.arange(smoke_n),
    }


def train_and_score(output_root: Path | None = None) -> None:
    """Train H1/H2-style model and emit per-sample scores for all splits."""
    if output_root is None:
        output_root = Path("results") / "h1h2_rebuild"

    settings = Settings()

    logger.info("=" * 80)
    logger.info("H1/H2 Rebuild – Phase 2: Train, Calibrate, and Score")
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Load data (time-ordered split) and combine train+test for full_ember
    # -------------------------------------------------------------------------
    logger.info("Loading EMBER-2024 with time-ordered split...")
    train_ds, test_ds = load_ember_2024(time_ordered=True)
    n_train = len(train_ds.features)
    n_test = len(test_ds.features)
    logger.info(f"Train samples: {n_train}, Test samples: {n_test}")

    # Full dataset (train + test) for full_ember evaluation
    all_features = pd.concat([train_ds.features, test_ds.features], ignore_index=True)
    all_labels = pd.concat([train_ds.labels, test_ds.labels], ignore_index=True)
    all_families = pd.concat([train_ds.families, test_ds.families], ignore_index=True)
    all_timestamps = pd.concat(
        [train_ds.timestamps, test_ds.timestamps], ignore_index=True
    )
    n_all = len(all_features)
    logger.info(f"Full dataset (train+test) size for full_ember: {n_all} samples")

    # -------------------------------------------------------------------------
    # Train H1 model via TrainingPipeline
    # -------------------------------------------------------------------------
    logger.info("Training LightGBM model (H1-style)...")
    training_pipeline = TrainingPipeline(settings)
    model_path = training_pipeline.run(
        train_data=train_ds,
        model_type="lgbm",
        model_name="h1h2_rebuild_lgbm",
        experiment_name="H1H2_Rebuild",
        seeds=5,
        is_smoke_test=False,
    )

    import joblib

    model = joblib.load(model_path)
    logger.info(f"Loaded trained model from {model_path}")

    # -------------------------------------------------------------------------
    # Prepare train/val split for calibration (10% of train for val)
    # -------------------------------------------------------------------------
    val_split = 0.1
    split_idx = int(n_train * (1 - val_split))

    train_cal = Dataset(
        features=train_ds.features.iloc[:split_idx].reset_index(drop=True),
        labels=train_ds.labels.iloc[:split_idx].reset_index(drop=True),
        families=train_ds.families.iloc[:split_idx].reset_index(drop=True),
        timestamps=train_ds.timestamps.iloc[:split_idx].reset_index(drop=True),
    )

    val_cal = Dataset(
        features=train_ds.features.iloc[split_idx:].reset_index(drop=True),
        labels=train_ds.labels.iloc[split_idx:].reset_index(drop=True),
        families=train_ds.families.iloc[split_idx:].reset_index(drop=True),
        timestamps=train_ds.timestamps.iloc[split_idx:].reset_index(drop=True),
    )

    logger.info(
        f"Calibration split: train={len(train_cal.features)}, val={len(val_cal.features)}"
    )

    # -------------------------------------------------------------------------
    # Generate uncalibrated probabilities
    # -------------------------------------------------------------------------
    X_train = train_cal.features.values
    X_val = val_cal.features.values
    X_all = all_features.values

    X_train_df = pd.DataFrame(X_train)
    X_val_df = pd.DataFrame(X_val)
    X_all_df = pd.DataFrame(X_all)

    prob_train = model.predict_proba(X_train_df)
    prob_val = model.predict_proba(X_val_df)
    prob_all = model.predict_proba(X_all_df)

    # Handle BaggedLightGBM (1D) vs standard sklearn (2D)
    if prob_train.ndim == 1:
        y_prob_train = prob_train
        y_prob_val = prob_val
        y_prob_all = prob_all
    else:
        y_prob_train = prob_train[:, 1]
        y_prob_val = prob_val[:, 1]
        y_prob_all = prob_all[:, 1]

    y_true_train = train_cal.labels.values
    y_true_val = val_cal.labels.values
    y_true_all = all_labels.values

    # -------------------------------------------------------------------------
    # Calibrate via CalibrationPipeline (auto method)
    # -------------------------------------------------------------------------
    logger.info("Calibrating probabilities (H2-style, auto method)...")
    calibration_pipeline = CalibrationPipeline(settings)
    calibrator = calibration_pipeline.run(
        train_data=Dataset(
            features=pd.DataFrame(X_train),
            labels=pd.Series(y_true_train),
        ),
        val_data=Dataset(
            features=pd.DataFrame(X_val),
            labels=pd.Series(y_true_val),
        ),
        y_prob_train=y_prob_train,
        y_prob_val=y_prob_val,
        method="auto",
        skip_mlflow=True,
    )

    y_prob_all_calibrated = calibrator.transform(y_prob_all)

    # Save models under a separate namespace
    models_dir = Path("models") / "h1h2_rebuild"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "h1_model.joblib")
    joblib.dump(calibrator, models_dir / "h2_calibrator.joblib")
    logger.info(f"Saved rebuild models under {models_dir}")

    # -------------------------------------------------------------------------
    # Compute a global cost-sensitive threshold (FN≫FP, banking-style)
    # -------------------------------------------------------------------------
    cost_fn = 100.0
    cost_fp = 1.0
    global_threshold = cost_sensitive_threshold(
        y_true_all, y_prob_all_calibrated, cost_fn=cost_fn, cost_fp=cost_fp
    )
    logger.info(
        f"Global cost-sensitive threshold (FN={cost_fn}, FP={cost_fp}): {global_threshold:.4f}"
    )

    # -------------------------------------------------------------------------
    # Emit per-sample scores for each split from full dataset (train+test)
    # -------------------------------------------------------------------------
    splits = _make_split_indices(n_all)

    for split_name, idx in splits.items():
        logger.info(f"\nWriting risk_scores for split '{split_name}' ({len(idx)} samples)")
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        families = all_families.iloc[idx].fillna("unknown").astype(str).str.lower()
        labels = all_labels.iloc[idx].astype(int).values
        probs = y_prob_all_calibrated[idx]

        df = pd.DataFrame(
            {
                "sample_id": [f"{split_name}_{i:06d}" for i in range(len(idx))],
                "true_label": labels,
                "family": families.values,
                "p_ransomware": probs,
                "predicted_label": (probs >= global_threshold).astype(int),
                "split_part": "test",
            }
        )

        out_path = split_dir / "risk_scores.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"  ✓ Wrote {out_path} ({len(df)} rows)")

    logger.info("\nAll splits scored successfully.")


def main() -> None:
    repo_root = Path(__file__).parent.parent.parent
    out_root = repo_root / "results" / "h1h2_rebuild"
    train_and_score(out_root)


if __name__ == "__main__":
    main()


