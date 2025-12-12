"""Generate deterministic ATT&CK→D3FEND lookup table.

This module provides functions to build and save the deterministic lookup table
from MITRE ATT&CK and D3FEND catalogs and edges.
"""

from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_deterministic_lookup(path: Path) -> pd.DataFrame:
    """
    Load deterministic lookup table from CSV.
    
    Args:
        path: Path to deterministic_lookup.csv
        
    Returns:
        DataFrame with deterministic mappings
    """
    LOGGER.info(f"Loading deterministic lookup from {path}")
    df = pd.read_csv(path)
    
    # Normalize column names if needed
    if "attack_id" in df.columns:
        df = df.rename(columns={"attack_id": "technique_id"})
    if "defense_id" in df.columns:
        df = df.rename(columns={"defense_id": "control_id"})
    
    LOGGER.info(f"Loaded {len(df)} deterministic mappings")
    return df


def save_deterministic_lookup(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save deterministic lookup table to CSV.
    
    Args:
        df: DataFrame with deterministic mappings
        out_path: Output path for CSV file
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    LOGGER.info(f"Saved deterministic lookup to {out_path}")

