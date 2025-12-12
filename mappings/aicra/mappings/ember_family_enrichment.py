"""Enrich EMBER dataset with ATT&CK technique mappings based on malware family."""

from __future__ import annotations

from pathlib import Path
import logging
import pandas as pd


def load_family_attack_map(path: Path) -> pd.DataFrame:
    """
    Load family→ATT&CK mapping from CSV.
    
    Args:
        path: Path to family_attack_map.csv
        
    Returns:
        DataFrame with columns: family_name, attack_id, confidence
    """
    logging.info(f"Loading family→ATT&CK mapping from {path}...")
    
    if not path.exists():
        raise FileNotFoundError(f"Family attack map not found at {path}")
    
    df = pd.read_csv(path)
    
    # Validate columns
    required_cols = ["family_name", "attack_id", "confidence"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Family attack map missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )
    
    logging.info(f"Loaded {len(df)} family→ATT&CK mappings")
    return df


def enrich_ember_with_attacks(
    ember_df: pd.DataFrame,
    family_attack_map: pd.DataFrame,
    family_col: str = "family",
) -> pd.DataFrame:
    """
    Left-join EMBER dataframe with family→ATT&CK mapping.
    
    Args:
        ember_df: EMBER dataset DataFrame
        family_attack_map: DataFrame with family_name, attack_id, confidence columns
        family_col: Name of the family column in ember_df
        
    Returns:
        Enriched DataFrame with additional columns: attack_id, confidence
    """
    if family_col not in ember_df.columns:
        raise ValueError(
            f"Family column '{family_col}' not found in EMBER DataFrame. "
            f"Available columns: {list(ember_df.columns)}"
        )
    
    # Perform left join on family name
    enriched_df = ember_df.merge(
        family_attack_map,
        left_on=family_col,
        right_on="family_name",
        how="left",
    )
    
    # Calculate percentage of rows with missing attack_id
    total_rows = len(enriched_df)
    missing_attack_id = enriched_df["attack_id"].isna().sum()
    pct_missing = (missing_attack_id / total_rows * 100) if total_rows > 0 else 0.0
    
    logging.info(
        f"Enriched {total_rows} EMBER samples with ATT&CK mappings. "
        f"{missing_attack_id} rows ({pct_missing:.2f}%) missing attack_id"
    )
    
    return enriched_df



