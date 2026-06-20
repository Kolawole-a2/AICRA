#!/usr/bin/env python3
"""
H1/H2 Rebuild – Phase 1: Build Split Manifests

This script creates simple per-sample manifests for four evaluation splits:
- smoke_test
- small_ember
- main
- full_ember

The splits are derived from the **full EMBER-2024 dataset** by concatenating
the time-ordered train and test sets returned by
`aicra.core.data.load_ember_2024(time_ordered=True)`:

- full_ember: all train + test samples (time-ordered)
- main:       first 10,000 samples from full_ember (or all if fewer)
- small_ember: first 2,000 samples from main
- smoke_test:  first 200 samples from small_ember

Outputs:
- artifacts/h1h2_rebuild/splits/<split>.manifest.csv

Schema:
- `sample_id`: stable identifier within the split (e.g., "<split>_000000")
- `split`:     split name (smoke_test, small_ember, main, full_ember)
- `true_label`: 0 = benign, 1 = ransomware
- `family`:   malware family name (from EMBER-2024, lower-cased)
- `timestamp`: ISO timestamp string

This script is read-only with respect to H3 code/data/configs; it only uses
`aicra.core.data.load_ember_2024` and the `data/ember2024` JSONL files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aicra.core.data import load_ember_2024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _make_split_indices(n: int) -> dict[str, pd.Index]:
    """Return row index selections for each split based on full dataset size n."""
    full_idx = pd.RangeIndex(0, n)
    main_n = min(10_000, n)
    small_n = min(2_000, main_n)
    smoke_n = min(200, small_n)

    main_idx = pd.RangeIndex(0, main_n)
    small_idx = pd.RangeIndex(0, small_n)
    smoke_idx = pd.RangeIndex(0, smoke_n)

    return {
        "full_ember": full_idx,
        "main": main_idx,
        "small_ember": small_idx,
        "smoke_test": smoke_idx,
    }


def build_manifests(output_root: Path | None = None) -> None:
    """Build manifests for all splits."""
    if output_root is None:
        output_root = Path("artifacts") / "h1h2_rebuild" / "splits"

    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("H1/H2 Rebuild – Phase 1: Building split manifests")
    logger.info("=" * 80)

    # Load EMBER-2024 with time-ordered split and combine train + test
    logger.info("Loading EMBER-2024 with time-ordered train + test...")
    train_ds, test_ds = load_ember_2024(time_ordered=True)

    # Combine train + test into a full dataset
    all_features = pd.concat([train_ds.features, test_ds.features], ignore_index=True)
    all_labels = pd.concat([train_ds.labels, test_ds.labels], ignore_index=True)
    all_families = pd.concat([train_ds.families, test_ds.families], ignore_index=True)
    all_timestamps = pd.concat(
        [train_ds.timestamps, test_ds.timestamps], ignore_index=True
    )

    n_all = len(all_features)
    logger.info(f"Loaded EMBER-2024 full dataset (train+test): {n_all} samples")
    print(f"[h1h2_rebuild/build_split_manifests] n_all={n_all}")

    splits = _make_split_indices(n_all)

    for split_name, idx in splits.items():
        logger.info(
            f"\nBuilding manifest for split '{split_name}' with {len(idx)} samples"
        )

        # Subset arrays from full dataset
        labels = all_labels.iloc[idx].reset_index(drop=True)
        families = all_families.iloc[idx].fillna("unknown").astype(str).str.lower()
        timestamps = all_timestamps.iloc[idx].reset_index(drop=True)

        df = pd.DataFrame(
            {
                "sample_id": [f"{split_name}_{i:06d}" for i in range(len(idx))],
                "split": split_name,
                "true_label": labels.astype(int),
                "family": families,
                "timestamp": timestamps.astype("datetime64[ns]").dt.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
            }
        )

        # Basic validation
        if not df["true_label"].isin([0, 1]).all():
            raise ValueError(f"{split_name}: true_label must be in {{0,1}}")

        out_path = output_root / f"{split_name}.manifest.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"  ✓ Wrote manifest: {out_path} ({len(df)} rows)")

    logger.info("\nAll manifests built successfully.")


def main() -> None:
    """CLI entry point."""
    repo_root = Path(__file__).parent.parent.parent
    out_root = repo_root / "artifacts" / "h1h2_rebuild" / "splits"
    build_manifests(out_root)


if __name__ == "__main__":
    main()
