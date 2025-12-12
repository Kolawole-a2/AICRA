"""
Data loading utilities for hypothesis experiments.

This module provides data loading functions that support train/val/test splits
for H1, H2, and H3 experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..core.data import Dataset, _load_jsonl_pair
from ..config import get_settings


def load_ember_2024(
    return_val: bool = False,
    val_split: float = 0.1,
    seed: int = 42,
    time_ordered: bool = True,
    stratified: bool = False,
) -> Tuple[Dataset, Dataset] | Tuple[Dataset, Dataset, Dataset]:
    """
    Load EMBER-2024 dataset with optional validation split.
    
    Args:
        return_val: If True, return train/val/test. If False, return train/test only.
        val_split: Fraction of training data to use for validation (only if return_val=True).
        seed: Random seed for deterministic splitting (used only if time_ordered=False and stratified=False).
        time_ordered: If True, split by timestamp to avoid temporal leakage. If False, use random or stratified split.
        stratified: If True, use stratified split to preserve class distribution. 
                   Only used if time_ordered=False. Requires return_val=True.
        
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
        train_indices = sorted_indices[:split_point]
        val_indices = sorted_indices[split_point:]
    elif stratified:
        # Stratified split: preserve class distribution (H1 requirement)
        if not return_val:
            raise ValueError("stratified=True requires return_val=True (stratified split needs validation set)")
        from sklearn.model_selection import train_test_split
        indices = np.arange(n_train)
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_split,
            stratify=train.labels.values,
            random_state=seed
        )
    else:
        # Random split (for backward compatibility)
        rng = np.random.default_rng(seed)
        indices = np.arange(n_train)
        rng.shuffle(indices)
        val_size = int(n_train * val_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
    
    # Create validation dataset
    val = Dataset(
        features=train.features.iloc[val_indices].reset_index(drop=True),
        labels=train.labels.iloc[val_indices].reset_index(drop=True),
        families=train.families.iloc[val_indices].reset_index(drop=True) if train.families is not None else None,
        timestamps=train.timestamps.iloc[val_indices].reset_index(drop=True) if train.timestamps is not None else None,
    )
    
    # Create reduced training dataset
    train_reduced = Dataset(
        features=train.features.iloc[train_indices].reset_index(drop=True),
        labels=train.labels.iloc[train_indices].reset_index(drop=True),
        families=train.families.iloc[train_indices].reset_index(drop=True) if train.families is not None else None,
        timestamps=train.timestamps.iloc[train_indices].reset_index(drop=True) if train.timestamps is not None else None,
    )
    
    return train_reduced, val, test
