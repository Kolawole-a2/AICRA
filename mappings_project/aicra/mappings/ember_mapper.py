"""Map EMBER-2024 dataset to ATT&CK→D3FEND using deterministic lookup."""

from pathlib import Path
import logging
import pandas as pd
import json
import glob
import os
from typing import Optional


def locate_ember_dataset(search_paths: Optional[list[Path]] = None) -> Optional[Path]:
    """
    Automatically locate the EMBER-2024 dataset in the AICRA project folder.
    
    Searches for common EMBER dataset locations:
    - ../AICRA/data/ember2024/
    - ../../AICRA/data/ember2024/
    - Any path containing 'ember2024' with train_features.jsonl
    
    Args:
        search_paths: Optional list of paths to search. If None, uses default search locations.
        
    Returns:
        Path to the EMBER dataset directory if found, None otherwise.
    """
    if search_paths is None:
        # Default search paths relative to current project
        current_dir = Path(__file__).parent.parent.parent
        search_paths = [
            current_dir.parent / "AICRA" / "data" / "ember2024",
            current_dir.parent.parent / "AICRA" / "data" / "ember2024",
            current_dir / ".." / "AICRA" / "data" / "ember2024",
        ]
        # Also search current directory and parent
        search_paths.extend([
            current_dir / "data" / "ember2024",
            current_dir.parent / "data" / "ember2024",
        ])
    
    for search_path in search_paths:
        search_path = Path(search_path).resolve()
        if search_path.exists() and search_path.is_dir():
            # Check for EMBER files
            train_features = search_path / "train_features.jsonl"
            if train_features.exists():
                logging.info(f"Found EMBER dataset at {search_path}")
                return search_path
    
    # Try glob search for ember2024 directories
    current_dir = Path(__file__).parent.parent.parent
    for pattern in [
        str(current_dir.parent / "**" / "ember2024"),
        str(current_dir.parent.parent / "**" / "ember2024"),
        str(current_dir / "**" / "ember2024"),
    ]:
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            match_path = Path(match)
            train_features = match_path / "train_features.jsonl"
            if train_features.exists():
                logging.info(f"Found EMBER dataset at {match_path}")
                return match_path
    
    logging.error("Could not locate EMBER-2024 dataset. Searched in:")
    for sp in search_paths:
        logging.error(f"  - {sp}")
    return None


def load_ember_features(features_path: Path) -> pd.DataFrame:
    """
    Load EMBER features from JSONL file.
    
    Args:
        features_path: Path to features JSONL file (e.g., train_features.jsonl)
        
    Returns:
        DataFrame with EMBER features, including 'family' column
    """
    logging.info(f"Loading EMBER features from {features_path}...")
    
    data = []
    with features_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                # Add sample index
                record["sample_index"] = line_num - 1
                data.append(record)
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue
    
    df = pd.DataFrame(data)
    logging.info(f"Loaded {len(df)} samples from {features_path}")
    
    # Ensure 'family' column exists
    if "family" not in df.columns:
        logging.warning("'family' column not found in EMBER features. Using 'unknown'.")
        df["family"] = "unknown"
    
    return df


def load_family_attack_map(map_path: Path) -> pd.DataFrame:
    """
    Load family→ATT&CK mapping from CSV.
    
    Args:
        map_path: Path to family_attack_map.csv
        
    Returns:
        DataFrame with columns: family, attack_id
    """
    logging.info(f"Loading family→ATT&CK mapping from {map_path}...")
    
    if not map_path.exists():
        raise FileNotFoundError(f"Family attack map not found at {map_path}")
    
    df = pd.read_csv(map_path)
    
    # Validate columns
    required_cols = ["family", "attack_id"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Family attack map missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )
    
    logging.info(f"Loaded {len(df)} family→ATT&CK mappings")
    return df[required_cols]


def load_deterministic_lookup(lookup_path: Path) -> pd.DataFrame:
    """
    Load deterministic ATT&CK→D3FEND lookup table.
    
    Args:
        lookup_path: Path to deterministic_attack_defense_lookup.csv or .parquet
        
    Returns:
        DataFrame with deterministic lookup table
    """
    logging.info(f"Loading deterministic lookup from {lookup_path}...")
    
    if not lookup_path.exists():
        raise FileNotFoundError(f"Deterministic lookup not found at {lookup_path}")
    
    if lookup_path.suffix == ".parquet":
        df = pd.read_parquet(lookup_path)
    else:
        df = pd.read_csv(lookup_path)
    
    logging.info(f"Loaded {len(df)} deterministic mappings")
    return df


def build_ember_deterministic_mapping(
    ember_dir: Path,
    family_map_path: Path,
    deterministic_lookup_path: Path,
    split: str = "train",
) -> pd.DataFrame:
    """
    Build sample-level EMBER→ATT&CK→D3FEND mapping using deterministic lookup.
    
    Args:
        ember_dir: Path to EMBER dataset directory
        family_map_path: Path to family_attack_map.csv
        deterministic_lookup_path: Path to deterministic_attack_defense_lookup.csv/.parquet
        split: Dataset split to process ("train" or "test")
        
    Returns:
        DataFrame with sample-level mapping:
        - sample_index: EMBER sample index
        - family: Malware family
        - attack_id: ATT&CK technique ID
        - attack_name: ATT&CK technique name
        - defense_id: D3FEND defense ID
        - defense_name: D3FEND defense name
        - is_correct: Always 1 (deterministic)
        - source: Source of the mapping
    """
    # Load data
    features_file = ember_dir / f"{split}_features.jsonl"
    ember_df = load_ember_features(features_file)
    
    family_map_df = load_family_attack_map(family_map_path)
    deterministic_df = load_deterministic_lookup(deterministic_lookup_path)
    
    # Map families to ATT&CK
    ember_with_attack = ember_df.merge(
        family_map_df,
        on="family",
        how="left",
    )
    
    # Filter to samples with valid family→ATT&CK mapping
    ember_with_attack = ember_with_attack.dropna(subset=["attack_id"])
    
    logging.info(
        f"Mapped {len(ember_with_attack)} samples to ATT&CK techniques "
        f"(out of {len(ember_df)} total samples)"
    )
    
    # Join with deterministic lookup to get ATT&CK→D3FEND mappings
    result_df = ember_with_attack.merge(
        deterministic_df,
        on="attack_id",
        how="left",
    )
    
    # Filter to samples with valid ATT&CK→D3FEND mapping
    result_df = result_df.dropna(subset=["defense_id"])
    
    # Select and order columns
    final_columns = [
        "sample_index",
        "family",
        "attack_id",
        "attack_name",
        "defense_id",
        "defense_name",
        "is_correct",
        "source",
    ]
    
    result_df = result_df[final_columns]
    
    logging.info(
        f"Built sample-level mapping with {len(result_df)} samples "
        f"mapped to D3FEND defenses"
    )
    
    return result_df


def save_ember_mapping(df: pd.DataFrame, out_dir: Path, split: str = "train") -> None:
    """
    Save EMBER deterministic mapping to CSV and Parquet.
    
    Args:
        df: DataFrame to save
        out_dir: Output directory
        split: Dataset split name (for filename)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / f"ember_deterministic_sample_mapping_{split}.csv"
    parquet_path = out_dir / f"ember_deterministic_sample_mapping_{split}.parquet"
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV to {csv_path}")
    
    # Save Parquet
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


def build_all_ember_mappings(
    ember_dir: Optional[Path] = None,
    family_map_path: Optional[Path] = None,
    deterministic_lookup_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    splits: list[str] = ["train", "test"],
) -> None:
    """
    High-level orchestration: build EMBER→ATT&CK→D3FEND mappings for all splits.
    
    Args:
        ember_dir: Path to EMBER dataset (auto-located if None)
        family_map_path: Path to family_attack_map.csv (default: data/family_attack_map.csv)
        deterministic_lookup_path: Path to deterministic lookup (default: data/mappings/deterministic_attack_defense_lookup.parquet)
        out_dir: Output directory (default: data/mappings)
        splits: List of dataset splits to process
    """
    # Auto-locate EMBER dataset if not provided
    if ember_dir is None:
        ember_dir = locate_ember_dataset()
        if ember_dir is None:
            raise FileNotFoundError("Could not locate EMBER-2024 dataset")
    
    ember_dir = Path(ember_dir)
    
    # Set default paths
    if family_map_path is None:
        family_map_path = Path("data/family_attack_map.csv")
    else:
        family_map_path = Path(family_map_path)
    
    if deterministic_lookup_path is None:
        # Try parquet first, then CSV
        parquet_path = Path("data/mappings/deterministic_attack_defense_lookup.parquet")
        csv_path = Path("data/mappings/deterministic_attack_defense_lookup.csv")
        if parquet_path.exists():
            deterministic_lookup_path = parquet_path
        elif csv_path.exists():
            deterministic_lookup_path = csv_path
        else:
            raise FileNotFoundError(
                f"Deterministic lookup not found at {parquet_path} or {csv_path}"
            )
    else:
        deterministic_lookup_path = Path(deterministic_lookup_path)
    
    if out_dir is None:
        out_dir = Path("data/mappings")
    else:
        out_dir = Path(out_dir)
    
    # Process each split and collect results
    all_dfs = []
    for split in splits:
        logging.info(f"Processing {split} split...")
        
        df = build_ember_deterministic_mapping(
            ember_dir=ember_dir,
            family_map_path=family_map_path,
            deterministic_lookup_path=deterministic_lookup_path,
            split=split,
        )
        
        # Add split column
        df["split"] = split
        
        save_ember_mapping(df, out_dir, split=split)
        all_dfs.append(df)
        
        logging.info(
            f"Completed {split} split: {len(df)} samples mapped to D3FEND defenses"
        )
    
    # Create combined output (all splits)
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Save combined files (without split suffix)
        csv_path = out_dir / "ember_deterministic_sample_mapping.csv"
        parquet_path = out_dir / "ember_deterministic_sample_mapping.parquet"
        
        combined_df.to_csv(csv_path, index=False)
        logging.info(f"Saved combined CSV to {csv_path}")
        
        try:
            combined_df.to_parquet(parquet_path, index=False, engine="pyarrow")
            logging.info(f"Saved combined Parquet to {parquet_path}")
        except ImportError:
            logging.warning("pyarrow not available, skipping combined Parquet export")
        except Exception as e:
            logging.error(f"Failed to save combined Parquet: {e}")
        
        logging.info(
            f"Combined mapping: {len(combined_df)} total samples across all splits"
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    build_all_ember_mappings()

