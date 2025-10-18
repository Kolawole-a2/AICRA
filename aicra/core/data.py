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
    return Dataset(features=features, labels=labels, families=families, timestamps=timestamps)


def _synthetic_dataset(n: int = 5000, d: int = 256, seed: int = 0) -> tuple[Dataset, Dataset]:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="H")
    families = rng.choice(["lockbit", "blackcat", "benign"], size=n, p=[0.15, 0.1, 0.75])
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


def load_ember_2024() -> tuple[Dataset, Dataset]:
    settings = get_settings()
    train_feat = settings.ember_dir / "train_features.jsonl"
    train_lab = settings.ember_dir / "train_labels.jsonl"
    test_feat = settings.ember_dir / "test_features.jsonl"
    test_lab = settings.ember_dir / "test_labels.jsonl"
    if all(p.exists() for p in [train_feat, train_lab, test_feat, test_lab]):
        train = _load_jsonl_pair(train_feat, train_lab)
        test = _load_jsonl_pair(test_feat, test_lab)
        return train, test
    raise FileNotFoundError(
        "EMBER-2024 files not found. Expected jsonl pairs under data/ember2024/. "
        "Use scripts download_ember.py and feature_extractor.py to fetch and prepare real data."
    )
