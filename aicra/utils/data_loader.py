"""
Data loading utilities for hypothesis experiments.

This module provides data loading functions that support train/val/test splits
for H1, H2, and H3 experiments.
"""

from __future__ import annotations

import numpy as np

from ..config import get_settings
from ..core.data import Dataset, _load_jsonl_pair


def load_ember_2024(
    return_val: bool = False,
    val_split: float = 0.1,
    seed: int = 42,
    time_ordered: bool = True,
    stratified: bool = False,
) -> tuple[Dataset, Dataset] | tuple[Dataset, Dataset, Dataset]:
    """
    Load EMBER-2024 dataset with optional validation split.

    Args:
        return_val: If True, return train/val/test. If False, return train/test only.
        val_split: Fraction of training data to use for validation (only if return_val=True).
        seed: Random seed for deterministic splitting (used only if time_ordered=False and stratified=False).
        time_ordered: If True, split by timestamp to avoid temporal leakage. If False, use random or stratified split.
        stratified: If True, use stratified split to preserve class distribution.
                   Can be combined with time_ordered=True for stratified sampling within time windows.
                   Requires return_val=True.

    Returns:
        If return_val=False: (train_data, test_data)
        If return_val=True: (train_data, val_data, test_data)

    Raises:
        FileNotFoundError: If EMBER-2024 files are not found.
        ValueError: If stratified=True but return_val=False (stratified split requires validation split).
    """
    settings = get_settings()
    train_feat = settings.ember_dir / "train_features.jsonl"
    train_lab = settings.ember_dir / "train_labels.jsonl"
    test_feat = settings.ember_dir / "test_features.jsonl"
    test_lab = settings.ember_dir / "test_labels.jsonl"

    if not all(p.exists() for p in [train_feat, train_lab, test_feat, test_lab]):
        raise FileNotFoundError(
            "EMBER-2024 files not found. Expected jsonl pairs under data/ember2024/. "
            "Use scripts download_ember.py and feature_extractor.py to fetch and prepare real data."
        )

    # Load train and test
    train = _load_jsonl_pair(train_feat, train_lab)
    test = _load_jsonl_pair(test_feat, test_lab)

    if not return_val:
        return train, test

    # Split training data into train/val
    n_train = len(train.features)

    if time_ordered and train.timestamps is not None:
        # Time-ordered split: sort by timestamp and split chronologically
        sorted_indices = train.timestamps.argsort()
        split_point = int(n_train * (1 - val_split))

        if stratified:
            # Combined stratified + time-ordered split
            # Strategy: Sort by time, then perform stratified sampling within time windows
            # to preserve both temporal ordering AND class distribution
            from sklearn.model_selection import train_test_split

            sorted_labels = train.labels.values[sorted_indices]

            # Use stratified split on sorted indices to maintain class balance
            # This will try to preserve class distribution while respecting time order
            train_indices_sorted, val_indices_sorted = train_test_split(
                np.arange(n_train),
                test_size=val_split,
                stratify=sorted_labels,
                random_state=seed,
            )

            # Map back to original indices
            train_indices_temp = sorted_indices[train_indices_sorted]
            val_indices_temp = sorted_indices[val_indices_sorted]

            # Verify time ordering is maintained
            train_timestamps = train.timestamps.values[train_indices_temp]
            val_timestamps = train.timestamps.values[val_indices_temp]

            if len(train_timestamps) > 0 and len(val_timestamps) > 0:
                max_train_ts = train_timestamps.max()
                min_val_ts = val_timestamps.min()

                if max_train_ts < min_val_ts:
                    # Time ordering is preserved - use stratified result
                    train_indices = train_indices_temp
                    val_indices = val_indices_temp
                else:
                    # Time ordering violated - use time-windowed stratified approach
                    # Split into time windows and do stratified sampling within each
                    n_windows = max(10, int(1 / val_split))  # Adaptive window count
                    window_size = n_train // n_windows

                    train_indices_list = []
                    val_indices_list = []

                    for i in range(n_windows):
                        start_idx = i * window_size
                        end_idx = (
                            (i + 1) * window_size if i < n_windows - 1 else n_train
                        )
                        window_indices = sorted_indices[start_idx:end_idx]
                        window_labels = train.labels.values[window_indices]

                        if (
                            len(window_indices) > 1
                            and len(np.unique(window_labels)) > 1
                        ):
                            # Stratified split within window
                            win_train, win_val = train_test_split(
                                np.arange(len(window_indices)),
                                test_size=val_split,
                                stratify=window_labels,
                                random_state=seed + i,  # Different seed per window
                            )
                            train_indices_list.extend(window_indices[win_train])
                            val_indices_list.extend(window_indices[win_val])
                        else:
                            # Simple split for small/uniform windows
                            win_split = int(len(window_indices) * (1 - val_split))
                            train_indices_list.extend(window_indices[:win_split])
                            val_indices_list.extend(window_indices[win_split:])

                    train_indices = np.array(train_indices_list)
                    val_indices = np.array(val_indices_list)
            else:
                train_indices = train_indices_temp
                val_indices = val_indices_temp
        else:
            # Pure time-ordered split (no stratification)
            train_indices = sorted_indices[:split_point]
            val_indices = sorted_indices[split_point:]
    elif stratified:
        # Stratified split: preserve class distribution (H1 requirement)
        if not return_val:
            raise ValueError(
                "stratified=True requires return_val=True (stratified split needs validation set)"
            )
        from sklearn.model_selection import train_test_split

        indices = np.arange(n_train)
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_split,
            stratify=train.labels.values,
            random_state=seed,
        )
    else:
        # Random split (for backward compatibility)
        rng = np.default_rng(seed)
        indices = np.arange(n_train)
        rng.shuffle(indices)
        val_size = int(n_train * val_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

    # Create validation dataset
    val = Dataset(
        features=train.features.iloc[val_indices].reset_index(drop=True),
        labels=train.labels.iloc[val_indices].reset_index(drop=True),
        families=(
            train.families.iloc[val_indices].reset_index(drop=True)
            if train.families is not None
            else None
        ),
        timestamps=(
            train.timestamps.iloc[val_indices].reset_index(drop=True)
            if train.timestamps is not None
            else None
        ),
    )

    # Create reduced training dataset
    train_reduced = Dataset(
        features=train.features.iloc[train_indices].reset_index(drop=True),
        labels=train.labels.iloc[train_indices].reset_index(drop=True),
        families=(
            train.families.iloc[train_indices].reset_index(drop=True)
            if train.families is not None
            else None
        ),
        timestamps=(
            train.timestamps.iloc[train_indices].reset_index(drop=True)
            if train.timestamps is not None
            else None
        ),
    )

    return train_reduced, val, test
