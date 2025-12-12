"""Build sample-level deterministic EMBER→ATT&CK→D3FEND mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging
import pandas as pd
import os

LOGGER = logging.getLogger(__name__)


def find_ember_2024(base_paths: list[Path] | None = None) -> Optional[Path]:
    """
    Search for an EMBER 2024 dataset file in likely locations,
    especially under a sibling AICRA project directory.
    
    Strategy:
      - If base_paths is None, use [Path(".."), Path("..") / "AICRA"].
      - Recursively walk these bases and look for filenames containing:
        - 'ember' (case-insensitive)
        - '2024'
        - extension in {'.parquet', '.csv'}
      - Prefer .parquet over .csv if multiple found.
    
    Args:
        base_paths: Optional list of base paths to search. If None, uses defaults.
        
    Returns:
        Path to the first best matching file, or None if not found.
    """
    if base_paths is None:
        current_dir = Path(__file__).parent.parent.parent
        base_paths = [
            current_dir.parent,
            current_dir.parent / "AICRA",
        ]
    
    found_files = []
    
    for base_path in base_paths:
        base_path = Path(base_path).resolve()
        if not base_path.exists():
            continue
        
        LOGGER.debug(f"Searching in {base_path}...")
        
        # Walk recursively
        for root, dirs, files in os.walk(base_path):
            for filename in files:
                filename_lower = filename.lower()
                
                # Check if filename contains 'ember' and '2024'
                if "ember" in filename_lower and "2024" in filename_lower:
                    # Check extension
                    ext = Path(filename).suffix.lower()
                    if ext in [".parquet", ".csv"]:
                        file_path = Path(root) / filename
                        found_files.append((file_path, ext))
                        LOGGER.debug(f"Found candidate: {file_path}")
    
    if not found_files:
        LOGGER.warning(
            "Could not find EMBER 2024 dataset file. Searched in: "
            + ", ".join(str(p) for p in base_paths)
        )
        return None
    
    # Prefer .parquet over .csv
    parquet_files = [f for f, ext in found_files if ext == ".parquet"]
    if parquet_files:
        chosen = parquet_files[0]
        LOGGER.info(f"Found EMBER 2024 dataset: {chosen}")
        return chosen
    
    # Fall back to CSV
    csv_files = [f for f, ext in found_files if ext == ".csv"]
    if csv_files:
        chosen = csv_files[0]
        LOGGER.info(f"Found EMBER 2024 dataset: {chosen}")
        return chosen
    
    # Return first found if no preference
    chosen = found_files[0][0]
    LOGGER.info(f"Found EMBER 2024 dataset: {chosen}")
    return chosen


def load_ember_2024(path: Path) -> pd.DataFrame:
    """
    Load the EMBER 2024 dataset from the given path.
    
    Supports .parquet and .csv formats.
    
    Args:
        path: Path to the EMBER 2024 dataset file
        
    Returns:
        DataFrame with EMBER data
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"EMBER 2024 file not found at {path}")
    
    LOGGER.info(f"Loading EMBER 2024 from {path}...")
    
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    LOGGER.info(f"Loaded {len(df)} samples from EMBER 2024")
    return df


def build_sample_level_deterministic_mapping(
    ember_with_attacks: pd.DataFrame,
    deterministic_lookup: pd.DataFrame,
    sample_id_col: str = "sha256",
    family_col: str = "family",
) -> pd.DataFrame:
    """
    Given EMBER enriched with attack_id and the deterministic ATT&CK→D3FEND lookup,
    join them to produce a sample-level mapping.
    
    Args:
        ember_with_attacks: EMBER DataFrame enriched with attack_id column
        deterministic_lookup: Deterministic ATT&CK→D3FEND lookup DataFrame
        sample_id_col: Name of the sample ID column in ember_with_attacks
        family_col: Name of the family column in ember_with_attacks
        
    Returns:
        DataFrame with columns:
        - sample_id_col
        - family_col
        - attack_id
        - attack_name
        - defense_id
        - defense_name
        - is_correct (1 for deterministic)
        - ransomware_weight (int; default 1)
    """
    # Validate required columns
    required_ember_cols = [sample_id_col, family_col, "attack_id"]
    missing_cols = [col for col in required_ember_cols if col not in ember_with_attacks.columns]
    if missing_cols:
        raise ValueError(
            f"EMBER DataFrame missing required columns: {missing_cols}. "
            f"Available columns: {list(ember_with_attacks.columns)}"
        )
    
    required_lookup_cols = ["attack_id", "attack_name", "defense_id", "defense_name"]
    missing_lookup_cols = [col for col in required_lookup_cols if col not in deterministic_lookup.columns]
    if missing_lookup_cols:
        raise ValueError(
            f"Deterministic lookup missing required columns: {missing_lookup_cols}. "
            f"Available columns: {list(deterministic_lookup.columns)}"
        )
    
    # Filter to rows with attack_id
    ember_with_attacks = ember_with_attacks.dropna(subset=["attack_id"])
    
    # Merge on attack_id
    result_df = ember_with_attacks.merge(
        deterministic_lookup,
        on="attack_id",
        how="left",
    )
    
    # Set is_correct = 1 for all rows (deterministic mapping)
    result_df["is_correct"] = 1
    
    # Set ransomware_weight = 1 (default)
    result_df["ransomware_weight"] = 1
    
    # Calculate percentage of rows with missing defense_id
    total_rows = len(result_df)
    missing_defense_id = result_df["defense_id"].isna().sum()
    pct_missing = (missing_defense_id / total_rows * 100) if total_rows > 0 else 0.0
    
    LOGGER.info(
        f"Built sample-level mapping with {total_rows} rows. "
        f"{missing_defense_id} rows ({pct_missing:.2f}%) missing defense_id"
    )
    
    # Select and order final columns
    final_columns = [
        sample_id_col,
        family_col,
        "attack_id",
        "attack_name",
        "defense_id",
        "defense_name",
        "is_correct",
        "ransomware_weight",
    ]
    
    result_df = result_df[final_columns]
    
    return result_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Locate EMBER 2024 dataset
    ember_path = find_ember_2024()
    if ember_path is None:
        raise FileNotFoundError("Could not locate EMBER 2024 dataset")
    
    # Load EMBER data
    ember_df = load_ember_2024(ember_path)
    
    # Load family→ATT&CK mapping
    from aicra.mappings.ember_family_enrichment import (
        load_family_attack_map,
        enrich_ember_with_attacks,
    )
    
    family_map_path = Path("data/mitre/family_attack_map.csv")
    family_attack_map = load_family_attack_map(family_map_path)
    
    # Enrich EMBER with ATT&CK mappings
    ember_with_attacks = enrich_ember_with_attacks(ember_df, family_attack_map)
    
    # Load deterministic lookup
    deterministic_lookup_path = Path("data/mappings/deterministic_attack_defense_lookup.csv")
    if not deterministic_lookup_path.exists():
        # Try parquet
        deterministic_lookup_path = Path("data/mappings/deterministic_attack_defense_lookup.parquet")
    
    if not deterministic_lookup_path.exists():
        raise FileNotFoundError(
            "Deterministic lookup not found. Run: python -m aicra.mappings.deterministic_builder"
        )
    
    deterministic_lookup = pd.read_csv(deterministic_lookup_path) if deterministic_lookup_path.suffix == ".csv" else pd.read_parquet(deterministic_lookup_path)
    
    # Build sample-level mapping
    sample_mapping = build_sample_level_deterministic_mapping(
        ember_with_attacks,
        deterministic_lookup,
    )
    
    # Save results
    out_dir = Path("data/mappings")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "ember_deterministic_sample_mapping.csv"
    parquet_path = out_dir / "ember_deterministic_sample_mapping.parquet"
    
    sample_mapping.to_csv(csv_path, index=False)
    LOGGER.info(f"Saved CSV to {csv_path}")
    
    try:
        sample_mapping.to_parquet(parquet_path, index=False, engine="pyarrow")
        LOGGER.info(f"Saved Parquet to {parquet_path}")
    except ImportError:
        LOGGER.warning("pyarrow not available, skipping Parquet export")
    except Exception as e:
        LOGGER.error(f"Failed to save Parquet: {e}")
    
    # Final summary
    LOGGER.info(f"EMBER file used: {ember_path}")
    LOGGER.info(f"Number of samples: {len(ember_df)}")
    LOGGER.info(f"Number of rows in sample-level mapping: {len(sample_mapping)}")

