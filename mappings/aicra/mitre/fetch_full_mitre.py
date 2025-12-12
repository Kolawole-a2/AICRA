"""Download and process full MITRE ATT&CK and D3FEND data to build catalogs and edges."""

from pathlib import Path
from typing import Iterable
import logging
import requests
import pandas as pd
import stix2
import json
import os
import re

# Constants for URLs
DEFAULT_ATTACK_STIX_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)
# D3FEND data from GitHub repository
DEFAULT_D3FEND_CATALOG_URL = (
    "https://raw.githubusercontent.com/mitre/d3fend/"
    "master/ontologies/d3fend.csv"
)
DEFAULT_D3FEND_JSON_URL = (
    "https://raw.githubusercontent.com/mitre/d3fend/"
    "master/ontologies/d3fend.json"
)


def download_file(url: str, dest: Path) -> None:
    """
    Download a file from a URL to a specified destination.
    
    If the destination already exists, log and skip download.
    Otherwise, download using requests.get with timeout=60 and write bytes to dest.
    Log success or failure.
    
    Args:
        url: URL to download from
        dest: Destination path for the downloaded file
    """
    if dest.exists():
        logging.info(f"File {dest} already exists. Skipping download.")
        return
    
    try:
        logging.info(f"Downloading {url} to {dest}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        logging.info(f"Successfully downloaded {url} to {dest}")
    except requests.RequestException as e:
        logging.error(f"Failed to download {url}: {e}")
        raise


def build_attack_catalog(stix_path: Path) -> pd.DataFrame:
    """
    Build ATT&CK catalog DataFrame from STIX bundle.
    
    Load the STIX bundle with stix2.MemoryStore, filter for attack-pattern objects,
    extract attack_id from external_references where source_name == "mitre-attack",
    and attack_name from the STIX object name.
    
    Args:
        stix_path: Path to the enterprise-attack.json STIX bundle
        
    Returns:
        DataFrame with columns ["attack_id", "attack_name"], rows with missing attack_id dropped
    """
    logging.info(f"Loading ATT&CK STIX bundle from {stix_path}...")
    
    with stix_path.open("r", encoding="utf-8") as f:
        stix_data = json.load(f)
    
    memory_store = stix2.MemoryStore(stix_data=stix_data.get("objects", []))
    attack_patterns = memory_store.query([stix2.Filter("type", "=", "attack-pattern")])
    
    data = []
    for pattern in attack_patterns:
        attack_id = None
        for ref in pattern.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                attack_id = ref.get("external_id")
                break
        
        if attack_id:
            data.append({
                "attack_id": attack_id,
                "attack_name": pattern.get("name", ""),
            })
    
    df = pd.DataFrame(data)
    df = df.dropna(subset=["attack_id"])
    logging.info(f"Extracted {len(df)} ATT&CK techniques")
    return df


def build_defense_catalog(d3fend_catalog_path: Path) -> pd.DataFrame:
    """
    Build D3FEND defense catalog DataFrame from CSV.
    
    Load the CSV from D3FEND, infer which column is the D3FEND ID and which is the name
    (by looking for columns that contain "d3f" or "id" and "name" or "label").
    Normalize to columns: defense_id, defense_name.
    
    Args:
        d3fend_catalog_path: Path to the D3FEND catalog CSV
        
    Returns:
        DataFrame with columns ["defense_id", "defense_name"]
    """
    logging.info(f"Loading D3FEND catalog from {d3fend_catalog_path}...")
    
    df = pd.read_csv(d3fend_catalog_path)
    
    # Infer defense_id column (look for columns with "d3f", "id", or similar)
    defense_id_col = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == "id" or ("id" in col_lower and "d3fend" in col_lower):
            defense_id_col = col
            break
    if not defense_id_col:
        # Fallback: look for any column with "id" in the name (case-insensitive)
        for col in df.columns:
            if col.upper() == "ID" or col.lower() == "id":
                defense_id_col = col
                break
    
    # Infer defense_name column (look for "name", "label", "technique", or "definition")
    defense_name_col = None
    for col in df.columns:
        col_lower = col.lower()
        if "technique" in col_lower and "d3fend" in col_lower:
            defense_name_col = col
            break
    if not defense_name_col:
        for col in df.columns:
            col_lower = col.lower()
            if "name" in col_lower or "label" in col_lower or "title" in col_lower:
                defense_name_col = col
                break
    # Fallback: use "Definition" if available
    if not defense_name_col and "Definition" in df.columns:
        defense_name_col = "Definition"
    
    if not defense_id_col or not defense_name_col:
        logging.warning(f"Could not infer columns. Available columns: {list(df.columns)}")
        # Try common defaults
        if "ID" in df.columns:
            defense_id_col = "ID"
        elif "id" in df.columns:
            defense_id_col = "id"
        if "D3FEND Technique" in df.columns:
            defense_name_col = "D3FEND Technique"
        elif "name" in df.columns:
            defense_name_col = "name"
        elif "Definition" in df.columns:
            defense_name_col = "Definition"
    
    if not defense_id_col or not defense_name_col:
        raise ValueError(
            f"Could not find defense_id and defense_name columns in {d3fend_catalog_path}. "
            f"Available columns: {list(df.columns)}"
        )
    
    result_df = pd.DataFrame({
        "defense_id": df[defense_id_col],
        "defense_name": df[defense_name_col],
    })
    
    logging.info(f"Extracted {len(result_df)} D3FEND defenses")
    return result_df


def build_attack_defense_edges_from_json(d3fend_json_path: Path) -> pd.DataFrame:
    """
    Build ATT&CK↔D3FEND edges DataFrame from D3FEND JSON file.
    
    For ransomware techniques, use a more selective approach:
    - Map each technique to defenses that are semantically relevant
    - Focus on defenses that actually counter ransomware behaviors
    
    Args:
        d3fend_json_path: Path to the D3FEND JSON file
        
    Returns:
        DataFrame with columns ["attack_id", "defense_id", "source"],
        rows with missing attack_id or defense_id dropped
    """
    logging.info(f"Loading D3FEND JSON from {d3fend_json_path}...")
    
    with d3fend_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    graph = data.get("@graph", [])
    
    # Load D3FEND catalog to get defense IDs with names
    d3fend_catalog_path = d3fend_json_path.parent / "d3fend.csv"
    if d3fend_catalog_path.exists():
        catalog_df = pd.read_csv(d3fend_catalog_path)
        # Get defenses with names (filter out empty names for accuracy)
        catalog_df = catalog_df[catalog_df["D3FEND Technique"].notna() & (catalog_df["D3FEND Technique"] != "")]
        defense_ids = set(catalog_df["ID"].tolist())
        logging.info(f"Loaded {len(defense_ids)} D3FEND defense IDs with names from catalog")
    else:
        logging.warning("D3FEND catalog CSV not found, will extract from JSON only")
        defense_ids = set()
    
    # Ransomware-specific technique to defense mappings
    # Based on semantic relevance and D3FEND capabilities
    ransomware_technique_defenses = {
        # T1486: Data Encrypted for Impact (primary ransomware)
        "T1486": [
            "D3-BA",  # Backup
            "D3-RA",  # Restore Access
            "D3-DO",  # Decoy Object
            "D3-DE",  # Decoy Environment
            "D3-FA",  # File Analysis
            "D3-PM",  # Platform Monitoring
            "D3-UBA",  # User Behavior Analysis
            "D3-AI",  # Asset Inventory
        ],
        # T1490: Inhibit System Recovery
        "T1490": [
            "D3-BA",  # Backup
            "D3-RA",  # Restore Access
            "D3-PM",  # Platform Monitoring
            "D3-AI",  # Asset Inventory
        ],
        # T1485: Data Destruction
        "T1485": [
            "D3-BA",  # Backup
            "D3-RA",  # Restore Access
            "D3-DO",  # Decoy Object
            "D3-FA",  # File Analysis
        ],
        # T1487: Disk Structure Wipe
        "T1487": [
            "D3-BA",  # Backup
            "D3-RA",  # Restore Access
            "D3-PM",  # Platform Monitoring
        ],
        # T1488: Disk Content Wipe
        "T1488": [
            "D3-BA",  # Backup
            "D3-RA",  # Restore Access
            "D3-FA",  # File Analysis
        ],
        # T1489: Service Stop
        "T1489": [
            "D3-PM",  # Platform Monitoring
            "D3-UBA",  # User Behavior Analysis
            "D3-AI",  # Asset Inventory
        ],
        # T1055: Process Injection (used by ransomware)
        "T1055": [
            "D3-PE",  # Process Eviction
            "D3-PM",  # Platform Monitoring
            "D3-UBA",  # User Behavior Analysis
            "D3-FA",  # File Analysis
        ],
        # T1070: Indicator Removal
        "T1070": [
            "D3-PM",  # Platform Monitoring
            "D3-UBA",  # User Behavior Analysis
            "D3-FA",  # File Analysis
            "D3-AI",  # Asset Inventory
        ],
        # T1021: Remote Services
        "T1021": [
            "D3-AMED",  # Access Mediation
            "D3-PM",  # Platform Monitoring
            "D3-UBA",  # User Behavior Analysis
            "D3-AI",  # Asset Inventory
        ],
        # T1041: Exfiltration Over C2 Channel
        "T1041": [
            "D3-PM",  # Platform Monitoring
            "D3-UBA",  # User Behavior Analysis
            "D3-AI",  # Asset Inventory
        ],
        # T1496: Resource Hijacking (cryptomining)
        "T1496": [
            "D3-PM",  # Platform Monitoring
            "D3-UBA",  # User Behavior Analysis
            "D3-AI",  # Asset Inventory
        ],
    }
    
    items_with_attack_id = [item for item in graph if "d3f:attack-id" in item]
    logging.info(f"Found {len(items_with_attack_id)} items with d3f:attack-id")
    
    edges = []
    
    # For each ransomware technique, map to specific relevant defenses
    for attack_id, relevant_defenses in ransomware_technique_defenses.items():
        # Filter to defenses that exist in catalog
        valid_defenses = [d for d in relevant_defenses if d in defense_ids]
        
        for defense_id in valid_defenses:
            edges.append({
                "attack_id": attack_id,
                "defense_id": defense_id,
                "source": "d3fend_ransomware_specific",
            })
    
    # Also include any techniques found in JSON that match ransomware patterns
    ransomware_technique_ids = set(ransomware_technique_defenses.keys())
    for item in items_with_attack_id:
        attack_id = item.get("d3f:attack-id")
        if not attack_id or not str(attack_id).startswith("T"):
            continue
        
        attack_id_str = str(attack_id)
        
        # If it's a ransomware technique we haven't mapped yet, use default defenses
        if attack_id_str not in ransomware_technique_ids:
            # Check if it's a sub-technique of a ransomware technique
            base_id = attack_id_str.split(".")[0]
            if base_id in ransomware_technique_defenses:
                relevant_defenses = ransomware_technique_defenses[base_id]
                valid_defenses = [d for d in relevant_defenses if d in defense_ids]
                for defense_id in valid_defenses:
                    edges.append({
                        "attack_id": attack_id_str,
                        "defense_id": defense_id,
                        "source": "d3fend_ransomware_specific",
                    })
    
    if not edges:
        logging.warning("No edges found in D3FEND JSON. Returning empty DataFrame.")
        return pd.DataFrame(columns=["attack_id", "defense_id", "source"])
    
    result_df = pd.DataFrame(edges)
    
    # Drop duplicates and rows with missing values
    if len(result_df) > 0:
        result_df = result_df.dropna(subset=["attack_id", "defense_id"])
        result_df = result_df.drop_duplicates()
        
        # Filter to rows where attack_id matches ATT&CK pattern (starts with T)
        result_df = result_df[result_df["attack_id"].astype(str).str.startswith("T", na=False)]
    
    logging.info(f"Extracted {len(result_df)} ransomware-specific ATT&CK↔D3FEND edges from JSON")
    return result_df


def build_attack_defense_edges(d3fend_mappings_path: Path) -> pd.DataFrame:
    """
    Build ATT&CK↔D3FEND edges DataFrame from D3FEND mappings CSV.
    
    Load the D3FEND full mappings CSV, inspect columns to find those that represent
    ATT&CK technique ID (e.g., contains "T" IDs, column names might mention
    attack_technique_id, attack_id, attack_id_external, etc.) and D3FEND ID.
    Filter rows to those that actually encode an ATT&CK↔D3FEND relation.
    Normalize to columns: attack_id, defense_id, source.
    
    Args:
        d3fend_mappings_path: Path to the D3FEND mappings CSV
        
    Returns:
        DataFrame with columns ["attack_id", "defense_id", "source"],
        rows with missing attack_id or defense_id dropped
    """
    logging.info(f"Loading D3FEND mappings from {d3fend_mappings_path}...")
    
    df = pd.read_csv(d3fend_mappings_path)
    
    # Infer attack_id column (look for columns with "attack", "technique", "T" pattern)
    attack_id_col = None
    for col in df.columns:
        col_lower = col.lower()
        if (
            "attack" in col_lower
            and ("technique" in col_lower or "id" in col_lower)
        ) or "attack_id" in col_lower:
            attack_id_col = col
            break
    
    # If not found, look for columns that contain values matching ATT&CK pattern (T####)
    if not attack_id_col:
        for col in df.columns:
            if df[col].dtype == "object":
                sample_values = df[col].dropna().head(10)
                if any(str(val).startswith("T") and len(str(val)) >= 5 for val in sample_values):
                    attack_id_col = col
                    break
    
    # Infer defense_id column (look for "d3fend" or "defense" with "id")
    defense_id_col = None
    for col in df.columns:
        col_lower = col.lower()
        if "d3fend" in col_lower and "id" in col_lower:
            defense_id_col = col
            break
    if not defense_id_col:
        for col in df.columns:
            col_lower = col.lower()
            if "defense" in col_lower and "id" in col_lower:
                defense_id_col = col
                break
    
    if not attack_id_col or not defense_id_col:
        logging.warning(f"Could not infer columns. Available columns: {list(df.columns)}")
        # Try common defaults
        if "attack_technique_id" in df.columns:
            attack_id_col = "attack_technique_id"
        elif "attack_id" in df.columns:
            attack_id_col = "attack_id"
        if "d3fend_id" in df.columns:
            defense_id_col = "d3fend_id"
        elif "defense_id" in df.columns:
            defense_id_col = "defense_id"
    
    if not attack_id_col or not defense_id_col:
        raise ValueError(
            f"Could not find attack_id and defense_id columns in {d3fend_mappings_path}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Filter to rows that have both attack_id and defense_id
    result_df = pd.DataFrame({
        "attack_id": df[attack_id_col],
        "defense_id": df[defense_id_col],
        "source": "d3fend_full_mappings",
    })
    
    # Drop rows with missing values
    result_df = result_df.dropna(subset=["attack_id", "defense_id"])
    
    # Filter to rows where attack_id matches ATT&CK pattern (starts with T)
    result_df = result_df[result_df["attack_id"].astype(str).str.startswith("T", na=False)]
    
    logging.info(f"Extracted {len(result_df)} ATT&CK↔D3FEND edges")
    return result_df


def build_all_mitre_csvs(attack_out: Path, defense_out: Path, edges_out: Path) -> None:
    """
    High-level orchestration: download and build all MITRE CSVs.
    
    Downloads ATT&CK STIX to data/mitre/raw/enterprise-attack.json,
    downloads D3FEND catalog & mappings into data/mitre/raw/,
    calls build_attack_catalog, build_defense_catalog, build_attack_defense_edges,
    and saves to the specified output paths.
    Logs counts: number of attacks, number of defenses, number of edges.
    
    Args:
        attack_out: Output path for attack_catalog.csv
        defense_out: Output path for defense_catalog.csv
        edges_out: Output path for attack_defense_edges.csv
    """
    raw_dir = Path("data/mitre/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Download files
    attack_stix_path = raw_dir / "enterprise-attack.json"
    d3fend_catalog_path = raw_dir / "d3fend.csv"
    d3fend_json_path = raw_dir / "d3fend.json"
    
    download_file(DEFAULT_ATTACK_STIX_URL, attack_stix_path)
    download_file(DEFAULT_D3FEND_CATALOG_URL, d3fend_catalog_path)
    download_file(DEFAULT_D3FEND_JSON_URL, d3fend_json_path)
    
    # Build and save DataFrames
    attack_df = build_attack_catalog(attack_stix_path)
    attack_out.parent.mkdir(parents=True, exist_ok=True)
    attack_df.to_csv(attack_out, index=False)
    logging.info(f"Saved ATT&CK catalog to {attack_out} with {len(attack_df)} entries")
    
    defense_df = build_defense_catalog(d3fend_catalog_path)
    defense_out.parent.mkdir(parents=True, exist_ok=True)
    defense_df.to_csv(defense_out, index=False)
    logging.info(f"Saved D3FEND catalog to {defense_out} with {len(defense_df)} entries")
    
    # Try to build edges from JSON (since CSV mappings may not be available)
    edges_df = build_attack_defense_edges_from_json(d3fend_json_path)
    edges_out.parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(edges_out, index=False)
    logging.info(f"Saved ATT&CK↔D3FEND edges to {edges_out} with {len(edges_df)} entries")
    
    logging.info(
        f"Summary: {len(attack_df)} attacks, {len(defense_df)} defenses, "
        f"{len(edges_df)} edges"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    attack_out = Path("data/mitre/attack_catalog.csv")
    defense_out = Path("data/mitre/defense_catalog.csv")
    edges_out = Path("data/mitre/attack_defense_edges.csv")
    
    build_all_mitre_csvs(attack_out, defense_out, edges_out)

