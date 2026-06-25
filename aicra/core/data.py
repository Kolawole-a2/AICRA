from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get_settings


@dataclass
class Dataset:
    features: pd.DataFrame
    labels: pd.Series
    families: pd.Series
    timestamps: pd.Series

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        families: pd.Series = None,
        timestamps: pd.Series = None,
    ):
        """Initialize Dataset with backward compatibility."""
        self.features = features
        self.labels = labels

        # Handle backward compatibility
        if families is None:
            self.families = pd.Series(["unknown"] * len(features))
        else:
            self.families = families

        if timestamps is None:
            self.timestamps = pd.Series(pd.Timestamp("2024-01-01")).repeat(
                len(features)
            )
        else:
            self.timestamps = timestamps


def _load_jsonl_pair(features_path: Path, labels_path: Path) -> Dataset:
    X = pd.read_json(features_path, lines=True)
    y = pd.read_json(labels_path, lines=True)
    if "label" in y.columns:
        labels = y["label"].astype(int)
    else:
        labels = y.squeeze().astype(int)
    families = X.get("family", pd.Series(["unknown"] * len(X)))
    timestamps = pd.to_datetime(
        X.get("timestamp", pd.Series(pd.Timestamp("2024-01-01")).repeat(len(X)))
    )
    feature_cols = [
        c
        for c in X.columns
        if c.startswith("feature_") or c.startswith("byte_") or c.startswith("pe_")
    ]
    features = X[feature_cols].astype(float)
    return Dataset(
        features=features, labels=labels, families=families, timestamps=timestamps
    )


def _synthetic_dataset(
    n: int = 5000, d: int = 256, seed: int = 0
) -> tuple[Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="H")
    families = rng.choice(
        ["lockbit", "blackcat", "benign"], size=n, p=[0.15, 0.1, 0.75]
    )
    labels = (families != "benign").astype(int)
    means = np.where(labels[:, None] == 1, 0.3, 0.0)
    X = rng.normal(loc=means, scale=1.0, size=(n, d)).astype(np.float32)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(d)])
    ds = Dataset(
        features=df,
        labels=pd.Series(labels),
        families=pd.Series(families),
        timestamps=pd.Series(timestamps),
    )
    split_idx = int(n * 0.8)
    train = Dataset(
        df.iloc[:split_idx].reset_index(drop=True),
        ds.labels.iloc[:split_idx].reset_index(drop=True),
        ds.families.iloc[:split_idx].reset_index(drop=True),
        ds.timestamps.iloc[:split_idx].reset_index(drop=True),
    )
    test = Dataset(
        df.iloc[split_idx:].reset_index(drop=True),
        ds.labels.iloc[split_idx:].reset_index(drop=True),
        ds.families.iloc[split_idx:].reset_index(drop=True),
        ds.timestamps.iloc[split_idx:].reset_index(drop=True),
    )
    return train, test


def load_ember_2024(
    time_ordered: bool = False,
    train_time_end: pd.Timestamp | None = None,
    test_time_start: pd.Timestamp | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Load EMBER-2024 dataset with optional time-ordered split.

    Args:
        time_ordered: If True, split by timestamp to ensure temporal ordering.
        train_time_end: Maximum timestamp for training set (if None, uses 80% chronologically).
        test_time_start: Minimum timestamp for test set (if None, uses data after train_time_end).

    Returns:
        (train_dataset, test_dataset)
    """
    import logging

    logger = logging.getLogger(__name__)

    settings = get_settings()
    train_feat = settings.ember_dir / "train_features.jsonl"
    train_lab = settings.ember_dir / "train_labels.jsonl"
    test_feat = settings.ember_dir / "test_features.jsonl"
    test_lab = settings.ember_dir / "test_labels.jsonl"

    if all(p.exists() for p in [train_feat, train_lab, test_feat, test_lab]):
        train = _load_jsonl_pair(train_feat, train_lab)
        test = _load_jsonl_pair(test_feat, test_lab)

        # Combine train and test for time-ordered split if requested
        if time_ordered:
            # Combine all data
            all_features = pd.concat([train.features, test.features], ignore_index=True)
            all_labels = pd.concat([train.labels, test.labels], ignore_index=True)
            all_families = (
                pd.concat([train.families, test.families], ignore_index=True)
                if train.families is not None
                else None
            )
            all_timestamps = pd.concat(
                [train.timestamps, test.timestamps], ignore_index=True
            )

            # Sort by timestamp
            sort_idx = all_timestamps.argsort()
            all_features = all_features.iloc[sort_idx].reset_index(drop=True)
            all_labels = all_labels.iloc[sort_idx].reset_index(drop=True)
            all_families = (
                all_families.iloc[sort_idx].reset_index(drop=True)
                if all_families is not None
                else None
            )
            all_timestamps = all_timestamps.iloc[sort_idx].reset_index(drop=True)

            n_rows = len(all_features)
            split_idx = int(n_rows * 0.8)

            if train_time_end is None and test_time_start is None:
                # Default H1 path: index-based 80/20 after sort (no boundary duplicates).
                train_features = all_features.iloc[:split_idx]
                train_labels = all_labels.iloc[:split_idx]
                train_families = (
                    all_families.iloc[:split_idx] if all_families is not None else None
                )
                train_timestamps = all_timestamps.iloc[:split_idx]

                test_features = all_features.iloc[split_idx:]
                test_labels = all_labels.iloc[split_idx:]
                test_families = (
                    all_families.iloc[split_idx:] if all_families is not None else None
                )
                test_timestamps = all_timestamps.iloc[split_idx:]
            else:
                if train_time_end is None:
                    train_time_end = all_timestamps.iloc[split_idx - 1]
                if test_time_start is None:
                    after_train = all_timestamps > train_time_end
                    if not after_train.any():
                        raise ValueError(
                            "No test samples after train_time_end="
                            f"{train_time_end}. Choose an earlier cutoff."
                        )
                    test_time_start = all_timestamps.loc[after_train].min()

                if train_time_end >= test_time_start:
                    raise ValueError(
                        "Invalid temporal split: train_time_end must be strictly "
                        f"before test_time_start ({train_time_end} >= {test_time_start})."
                    )

                train_mask = all_timestamps <= train_time_end
                test_mask = all_timestamps >= test_time_start

                train_features = all_features[train_mask]
                train_labels = all_labels[train_mask]
                train_families = (
                    all_families[train_mask] if all_families is not None else None
                )
                train_timestamps = all_timestamps[train_mask]

                test_features = all_features[test_mask]
                test_labels = all_labels[test_mask]
                test_families = (
                    all_families[test_mask] if all_families is not None else None
                )
                test_timestamps = all_timestamps[test_mask]

            train = Dataset(
                features=train_features.reset_index(drop=True),
                labels=train_labels.reset_index(drop=True),
                families=(
                    train_families.reset_index(drop=True)
                    if train_families is not None
                    else None
                ),
                timestamps=train_timestamps.reset_index(drop=True),
            )

            test = Dataset(
                features=test_features.reset_index(drop=True),
                labels=test_labels.reset_index(drop=True),
                families=(
                    test_families.reset_index(drop=True)
                    if test_families is not None
                    else None
                ),
                timestamps=test_timestamps.reset_index(drop=True),
            )

            logger.info(
                f"Time-ordered split: train={len(train.features)} (max_ts={train.timestamps.max()}), "
                f"test={len(test.features)} (min_ts={test.timestamps.min()})"
            )

        return train, test

    raise FileNotFoundError(
        "EMBER-2024 files not found. Expected jsonl pairs under data/ember2024/. "
        "Use scripts download_ember.py and feature_extractor.py to fetch and prepare real data."
    )
