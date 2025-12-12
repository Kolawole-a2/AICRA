"""Build deterministic ATT&CK→D3FEND lookup table from MITRE catalogs and edges."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import logging
import pandas as pd


def load_attack_catalog(path: Path) -> pd.DataFrame:
    """
    Load ATT&CK catalog from CSV.
    
    Args:
        path: Path to attack_catalog.csv
        
    Returns:
        DataFrame with attack_id and attack_name columns
    """
    logging.info(f"Loading ATT&CK catalog from {path}...")
    df = pd.read_csv(path)
    logging.info(f"Loaded {len(df)} ATT&CK techniques")
    return df


def load_defense_catalog(path: Path) -> pd.DataFrame:
    """
    Load D3FEND defense catalog from CSV.
    
    Args:
        path: Path to defense_catalog.csv
        
    Returns:
        DataFrame with defense_id and defense_name columns
    """
    logging.info(f"Loading D3FEND defense catalog from {path}...")
    df = pd.read_csv(path)
    logging.info(f"Loaded {len(df)} D3FEND defenses")
    return df


def load_attack_defense_edges(path: Path) -> pd.DataFrame:
    """
    Load ATT&CK↔D3FEND edges from CSV.
    
    Args:
        path: Path to attack_defense_edges.csv
        
    Returns:
        DataFrame with attack_id, defense_id, and source columns
    """
    logging.info(f"Loading ATT&CK↔D3FEND edges from {path}...")
    df = pd.read_csv(path)
    logging.info(f"Loaded {len(df)} edges")
    return df


def build_deterministic_lookup(
    attack_catalog_path: Path,
    defense_catalog_path: Path,
    edges_path: Path,
    ransomware_only: bool = True,
) -> pd.DataFrame:
    """
    Build deterministic ATT&CK→D3FEND lookup table.
    
    Load all three catalogs, merge edges with attack catalog on attack_id,
    merge that result with defense catalog on defense_id.
    Optionally filter to ransomware-related ATT&CK techniques only.
    Create a DataFrame with columns:
    - attack_id
    - attack_name
    - defense_id
    - defense_name
    - is_correct (int, always 1)
    - source (string; prefer the source column from edges, or "mitre_d3fend" if missing)
    
    Args:
        attack_catalog_path: Path to attack_catalog.csv
        defense_catalog_path: Path to defense_catalog.csv
        edges_path: Path to attack_defense_edges.csv
        ransomware_only: If True, filter to ransomware-related ATT&CK techniques
        
    Returns:
        DataFrame with the deterministic lookup table
    """
    # Load all three
    attack_df = load_attack_catalog(attack_catalog_path)
    defense_df = load_defense_catalog(defense_catalog_path)
    edges_df = load_attack_defense_edges(edges_path)
    
    # Filter to ransomware-related techniques if requested
    if ransomware_only:
        # Define comprehensive list of ransomware-related techniques
        # Including primary impact techniques and supporting techniques used by ransomware
        ransomware_technique_ids = [
            "T1486",  # Data Encrypted for Impact (primary ransomware technique)
            "T1490",  # Inhibit System Recovery
            "T1485",  # Data Destruction
            "T1487",  # Disk Structure Wipe
            "T1488",  # Disk Content Wipe
            "T1489",  # Service Stop
            "T1055",  # Process Injection (used by ransomware)
            "T1070",  # Indicator Removal (used by ransomware)
            "T1021",  # Remote Services (used by ransomware)
            "T1041",  # Exfiltration Over C2 Channel (used by ransomware)
            "T1496",  # Resource Hijacking (cryptomining, often with ransomware)
        ]
        
        # Filter to techniques that start with these IDs (including sub-techniques)
        attack_df_filtered = attack_df[
            attack_df["attack_id"].str.startswith(tuple(ransomware_technique_ids), na=False)
        ]
        
        # Also include techniques with ransomware-related keywords in name
        ransomware_keywords = [
            "ransom", "encrypt", "encryption", "crypto", "locker", 
            "wiper", "destroy", "recovery", "inhibit", "encrypted"
        ]
        ransomware_pattern = "|".join(ransomware_keywords)
        keyword_filtered = attack_df[
            attack_df["attack_name"].str.contains(
                ransomware_pattern, case=False, na=False
            )
        ]
        
        # Combine both filters
        attack_df_filtered = pd.concat([attack_df_filtered, keyword_filtered]).drop_duplicates()
        
        logging.info(
            f"Filtered to {len(attack_df_filtered)} ransomware-related ATT&CK techniques "
            f"(out of {len(attack_df)} total)"
        )
        
        # Filter edges to only include ransomware techniques
        edges_df = edges_df[edges_df["attack_id"].isin(attack_df_filtered["attack_id"])]
        attack_df = attack_df_filtered
    
    # Merge edges with attack catalog on attack_id
    merged_df = edges_df.merge(
        attack_df,
        on="attack_id",
        how="inner",  # Use inner join to ensure we only get valid attacks
    )
    
    # Merge that result with defense catalog on defense_id
    # Use inner join and ensure defense_name is populated
    merged_df = merged_df.merge(
        defense_df,
        on="defense_id",
        how="inner",  # Use inner join to ensure we only get valid defenses
    )
    
    # Filter out rows where defense_name is missing
    merged_df = merged_df[merged_df["defense_name"].notna() & (merged_df["defense_name"] != "")]
    
    # Create the final DataFrame with required columns
    result_df = pd.DataFrame({
        "attack_id": merged_df["attack_id"],
        "attack_name": merged_df["attack_name"],
        "defense_id": merged_df["defense_id"],
        "defense_name": merged_df["defense_name"],
        "is_correct": 1,  # Always 1 for deterministic lookup
        "source": merged_df.get("source", "mitre_d3fend").fillna("mitre_d3fend"),
    })
    
    # Log statistics
    total_rows = len(result_df)
    missing_attack_name = result_df["attack_name"].isna().sum()
    missing_defense_name = result_df["defense_name"].isna().sum()
    
    logging.info(f"Built deterministic lookup table with {total_rows} rows")
    logging.info(f"Rows missing attack_name: {missing_attack_name}")
    logging.info(f"Rows missing defense_name: {missing_defense_name}")
    
    if ransomware_only:
        logging.info("Filtered to ransomware-related ATT&CK techniques only")
        logging.info(f"Unique ransomware techniques: {result_df['attack_id'].nunique()}")
        logging.info(f"Unique defenses: {result_df['defense_id'].nunique()}")
    
    return result_df


def save_deterministic_lookup(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Save deterministic lookup table to CSV and Parquet formats.
    
    Args:
        df: DataFrame to save
        out_dir: Output directory (will be created if it doesn't exist)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "deterministic_attack_defense_lookup.csv"
    parquet_path = out_dir / "deterministic_attack_defense_lookup.parquet"
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV to {csv_path}")
    
    # Save Parquet (requires pyarrow)
    try:
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        logging.info(f"Saved Parquet to {parquet_path}")
    except ImportError:
        logging.warning(
            "pyarrow not available, skipping Parquet export. "
            "Install with: pip install pyarrow"
        )
    except Exception as e:
        logging.error(f"Failed to save Parquet: {e}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    attack_catalog = Path("data/mitre/attack_catalog.csv")
    defense_catalog = Path("data/mitre/defense_catalog.csv")
    edges = Path("data/mitre/attack_defense_edges.csv")
    out_dir = Path("data/mappings")
    
    # Build ransomware-only deterministic lookup
    df = build_deterministic_lookup(
        attack_catalog, 
        defense_catalog, 
        edges,
        ransomware_only=True,
    )
    save_deterministic_lookup(df, out_dir)

