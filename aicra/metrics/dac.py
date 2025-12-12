"""
Defense-Attack Consistency (DAC) Metric Computation

DAC measures the proportion of correctly aligned ATT&CK→D3FEND pairs
among all mapped relations, serving as a quality metric for ontology-driven
cybersecurity mappings.

The deterministic lookup mapping is expected to achieve the highest DAC,
serving as a performance ceiling for learned mappings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_deterministic_mapping(mapping_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Load deterministic ATT&CK to D3FEND mapping from YAML.
    
    This is the authoritative, curated mapping that serves as the gold standard.
    It is expected to achieve the highest DAC due to being prefilled and validated.
    
    Args:
        mapping_path: Path to attack_to_d3fend.yaml. If None, uses default location.
        
    Returns:
        Dictionary mapping technique_id to list of control_id values.
    """
    if mapping_path is None:
        from ..config import Settings
        settings = Settings()
        mapping_path = settings.data_dir / "lookups" / "attack_to_d3fend.yaml"
    
    logger.info(f"Loading deterministic mapping from {mapping_path}")
    
    with open(mapping_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    mappings = data.get("mappings", {})
    logger.info(f"Loaded {len(mappings)} technique mappings with {sum(len(v) for v in mappings.values())} total pairs")
    
    return mappings


def load_learned_mapping(mapping_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load learned/heuristic ATT&CK to D3FEND mapping from CSV.
    
    This mapping may contain inference errors and is expected to have lower DAC
    than the deterministic mapping due to semantic uncertainty.
    
    Args:
        mapping_path: Path to learned mapping CSV file.
        
    Returns:
        DataFrame with columns: technique_id, control_id, optionally score.
    """
    mapping_path = Path(mapping_path)
    logger.info(f"Loading learned mapping from {mapping_path}")
    
    df = pd.read_csv(mapping_path)
    
    required_cols = {"technique_id", "control_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Learned mapping missing required columns: {missing}")
    
    logger.info(f"Loaded {len(df)} learned mapping pairs covering {df['technique_id'].nunique()} techniques")
    
    return df


def load_reference_pairs(ref_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load canonical MITRE D3FEND reference pairs.
    
    These are the ground-truth ATT&CK→D3FEND mappings from MITRE,
    used to validate mapping correctness.
    
    Args:
        ref_path: Path to reference pairs CSV. If None, uses deterministic mapping as reference.
        
    Returns:
        DataFrame with columns: technique_id, control_id.
    """
    if ref_path is None:
        # Use deterministic mapping as reference (it is authoritative)
        det_mapping = load_deterministic_mapping()
        rows = []
        for technique_id, controls in det_mapping.items():
            for control_id in controls:
                rows.append({
                    "technique_id": technique_id,
                    "control_id": control_id
                })
        return pd.DataFrame(rows)
    
    ref_path = Path(ref_path)
    logger.info(f"Loading reference pairs from {ref_path}")
    
    df = pd.read_csv(ref_path)
    
    required_cols = {"technique_id", "control_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Reference pairs missing required columns: {missing}")
    
    logger.info(f"Loaded {len(df)} reference pairs")
    
    return df


def compute_dac(
    mapping_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    mapping_type: str = "unknown"
) -> float:
    """
    Compute Defense-Attack Consistency (DAC) metric.
    
    DAC = (number of correctly aligned pairs) / (total mapped pairs)
    
    The deterministic lookup mapping is expected to achieve DAC ≈ 1.0
    (or very close) because it is curated and authoritative.
    
    Learned mappings may have lower DAC due to inference uncertainty.
    
    Args:
        mapping_df: DataFrame with technique_id and control_id columns.
        reference_df: DataFrame with canonical reference pairs.
        mapping_type: Type of mapping ("deterministic" or "learned") for logging.
        
    Returns:
        DAC value in [0, 1], where 1.0 = perfect alignment with reference.
    """
    logger.info(f"Computing DAC for {mapping_type} mapping")
    
    # Convert to sets of tuples for efficient comparison
    mapping_pairs = set(
        tuple(row) for row in mapping_df[["technique_id", "control_id"]].dropna().values.tolist()
    )
    reference_pairs = set(
        tuple(row) for row in reference_df[["technique_id", "control_id"]].dropna().values.tolist()
    )
    
    # Count correctly aligned pairs (intersection)
    correct_pairs = mapping_pairs & reference_pairs
    total_pairs = len(mapping_pairs)
    
    if total_pairs == 0:
        logger.warning(f"{mapping_type} mapping has no pairs, DAC = 0.0")
        return 0.0
    
    dac = len(correct_pairs) / total_pairs
    
    logger.info(
        f"{mapping_type} DAC: {dac:.4f} ({len(correct_pairs)}/{total_pairs} pairs correct)"
    )
    
    return float(dac)


def compute_coverage(
    mapping_df: pd.DataFrame,
    ransomware_techniques: Optional[List[str]] = None
) -> float:
    """
    Compute coverage as fraction of ransomware-relevant ATT&CK techniques covered.
    
    Coverage = (techniques with at least one mapped control) / (total techniques)
    
    Args:
        mapping_df: DataFrame with technique_id and control_id columns.
        ransomware_techniques: List of ransomware-relevant technique IDs.
                              If None, uses all techniques in mapping.
        
    Returns:
        Coverage value in [0, 1].
    """
    if ransomware_techniques is None:
        # Use all techniques that appear in the mapping
        techniques_total = mapping_df["technique_id"].nunique()
        techniques_with_controls = mapping_df.dropna(subset=["control_id"])["technique_id"].nunique()
    else:
        techniques_total = len(set(ransomware_techniques))
        mapped_techniques = set(mapping_df.dropna(subset=["control_id"])["technique_id"].unique())
        techniques_with_controls = len(set(ransomware_techniques) & mapped_techniques)
    
    if techniques_total == 0:
        return 0.0
    
    coverage = techniques_with_controls / techniques_total
    
    logger.info(
        f"Coverage: {coverage:.4f} ({techniques_with_controls}/{techniques_total} techniques covered)"
    )
    
    return float(coverage)


def compute_dac_metrics(
    deterministic_mapping: Union[Dict[str, List[str]], pd.DataFrame],
    learned_mapping: pd.DataFrame,
    reference_pairs: pd.DataFrame,
    ransomware_techniques: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive DAC metrics for both mapping types.
    
    This function compares deterministic (authoritative) vs learned (inferred) mappings,
    with the expectation that deterministic mapping achieves higher DAC.
    
    Args:
        deterministic_mapping: Either dict (from YAML) or DataFrame of deterministic mapping.
        learned_mapping: DataFrame of learned mapping.
        reference_pairs: DataFrame of canonical reference pairs.
        ransomware_techniques: Optional list of ransomware-relevant techniques.
        
    Returns:
        Dictionary with metrics:
        - dac_deterministic: DAC for deterministic mapping (expected to be highest)
        - dac_learned: DAC for learned mapping (expected to be lower)
        - coverage_deterministic: Coverage for deterministic mapping
        - coverage_learned: Coverage for learned mapping
        - dac_delta: Difference (deterministic - learned), expected to be positive
    """
    # Convert deterministic mapping to DataFrame if needed
    if isinstance(deterministic_mapping, dict):
        det_rows = []
        for technique_id, controls in deterministic_mapping.items():
            for control_id in controls:
                det_rows.append({
                    "technique_id": technique_id,
                    "control_id": control_id
                })
        det_df = pd.DataFrame(det_rows)
    else:
        det_df = deterministic_mapping
    
    # Compute DAC for both mappings
    dac_det = compute_dac(det_df, reference_df=reference_pairs, mapping_type="deterministic")
    dac_learned = compute_dac(learned_mapping, reference_df=reference_pairs, mapping_type="learned")
    
    # Compute coverage
    coverage_det = compute_coverage(det_df, ransomware_techniques)
    coverage_learned = compute_coverage(learned_mapping, ransomware_techniques)
    
    # Compute delta
    dac_delta = dac_det - dac_learned
    
    metrics = {
        "dac_deterministic": dac_det,
        "dac_learned": dac_learned,
        "coverage_deterministic": coverage_det,
        "coverage_learned": coverage_learned,
        "dac_delta": dac_delta
    }
    
    logger.info(
        f"DAC Metrics Summary:\n"
        f"  Deterministic DAC: {dac_det:.4f} (expected to be highest)\n"
        f"  Learned DAC: {dac_learned:.4f} (expected to be lower due to inference uncertainty)\n"
        f"  DAC Delta: {dac_delta:.4f} (positive = deterministic outperforms learned)"
    )
    
    return metrics


def compute_dac_between_mappings(
    df_deterministic: pd.DataFrame,
    df_learned: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute Defense–Attack Consistency (DAC) between a deterministic (gold) mapping
    and a learned mapping.

    Both inputs must contain:
      - attack_id
      - defense_id

    DAC is defined as:
      (# of learned (attack, defense) pairs that exist in the deterministic mapping)
      divided by
      (total # of deterministic (attack, defense) pairs)
    
    Args:
        df_deterministic: DataFrame with attack_id and defense_id columns (gold standard)
        df_learned: DataFrame with attack_id and defense_id columns (learned mapping)
        
    Returns:
        Dictionary with DAC metrics:
        - dac: DAC score (overlap / deterministic pairs)
        - n_det_pairs: Number of deterministic pairs
        - n_learned_pairs: Number of learned pairs
        - n_overlap_pairs: Number of overlapping pairs
        - precision_learned_wrt_deterministic: Precision of learned pairs (overlap / learned pairs)
    """
    logger.info("Computing DAC between deterministic and learned mappings")
    
    # Validate required columns
    required_det = {"attack_id", "defense_id"}
    required_learned = {"attack_id", "defense_id"}
    
    missing_det = required_det - set(df_deterministic.columns)
    missing_learned = required_learned - set(df_learned.columns)
    
    if missing_det:
        raise ValueError(f"Deterministic mapping missing required columns: {missing_det}")
    if missing_learned:
        raise ValueError(f"Learned mapping missing required columns: {missing_learned}")
    
    # Normalize to common key: create pair column
    df_det_clean = df_deterministic[["attack_id", "defense_id"]].dropna()
    df_learned_clean = df_learned[["attack_id", "defense_id"]].dropna()
    
    # Create pair strings
    df_det_clean["pair"] = df_det_clean["attack_id"].astype(str) + "||" + df_det_clean["defense_id"].astype(str)
    df_learned_clean["pair"] = df_learned_clean["attack_id"].astype(str) + "||" + df_learned_clean["defense_id"].astype(str)
    
    # Convert to sets
    pairs_det = set(df_det_clean["pair"].unique())
    pairs_learned = set(df_learned_clean["pair"].unique())
    
    # Compute metrics
    n_det = len(pairs_det)
    n_learned = len(pairs_learned)
    n_overlap = len(pairs_det & pairs_learned)
    
    # Compute DAC (handle division by zero)
    if n_det == 0:
        logger.warning("Deterministic mapping has no pairs, DAC = 0.0")
        dac = 0.0
    else:
        dac = n_overlap / n_det
    
    # Compute precision of learned mapping (how many learned pairs are valid)
    if n_learned == 0:
        logger.warning("Learned mapping has no pairs, precision = 0.0")
        precision_learned_wrt_det = 0.0
    else:
        precision_learned_wrt_det = n_overlap / n_learned
    
    logger.info(
        f"DAC Metrics:\n"
        f"  Deterministic pairs: {n_det}\n"
        f"  Learned pairs: {n_learned}\n"
        f"  Overlapping pairs: {n_overlap}\n"
        f"  DAC: {dac:.4f}\n"
        f"  Precision (learned wrt deterministic): {precision_learned_wrt_det:.4f}"
    )
    
    return {
        "dac": dac,
        "n_det_pairs": n_det,
        "n_learned_pairs": n_learned,
        "n_overlap_pairs": n_overlap,
        "precision_learned_wrt_deterministic": precision_learned_wrt_det,
    }


def compute_dac_per_attack(
    df_deterministic: pd.DataFrame,
    df_learned: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Defense–Attack Consistency (DAC) per attack_id.
    
    For each attack_id, computes:
    - n_det_pairs: Number of deterministic (attack_id, defense_id) pairs
    - n_learned_pairs: Number of learned (attack_id, defense_id) pairs
    - n_overlap_pairs: Number of overlapping pairs
    - dac_attack: DAC score for this attack (overlap / det_pairs)
    - precision_learned_wrt_det_attack: Precision of learned pairs (overlap / learned_pairs)
    - coverage_det: 1 if det_pairs > 0 else 0
    - coverage_learned: 1 if learned_pairs > 0 else 0
    
    Args:
        df_deterministic: DataFrame with attack_id and defense_id columns (gold standard)
        df_learned: DataFrame with attack_id and defense_id columns (learned mapping)
        
    Returns:
        DataFrame with one row per attack_id containing all metrics.
    """
    logger.info("Computing DAC per attack")
    
    # Validate required columns
    required_det = {"attack_id", "defense_id"}
    required_learned = {"attack_id", "defense_id"}
    
    missing_det = required_det - set(df_deterministic.columns)
    missing_learned = required_learned - set(df_learned.columns)
    
    if missing_det:
        raise ValueError(f"Deterministic mapping missing required columns: {missing_det}")
    if missing_learned:
        raise ValueError(f"Learned mapping missing required columns: {missing_learned}")
    
    # Clean and create pair columns
    df_det_clean = df_deterministic[["attack_id", "defense_id"]].dropna()
    df_learned_clean = df_learned[["attack_id", "defense_id"]].dropna()
    
    df_det_clean["pair"] = (
        df_det_clean["attack_id"].astype(str) + "||" + df_det_clean["defense_id"].astype(str)
    )
    df_learned_clean["pair"] = (
        df_learned_clean["attack_id"].astype(str) + "||" + df_learned_clean["defense_id"].astype(str)
    )
    
    # Get all unique attack_ids from both mappings
    all_attack_ids = set(df_det_clean["attack_id"].unique()) | set(df_learned_clean["attack_id"].unique())
    
    results = []
    
    for attack_id in all_attack_ids:
        # Get pairs for this attack_id
        det_pairs = set(df_det_clean[df_det_clean["attack_id"] == attack_id]["pair"].unique())
        learned_pairs = set(df_learned_clean[df_learned_clean["attack_id"] == attack_id]["pair"].unique())
        
        n_det = len(det_pairs)
        n_learned = len(learned_pairs)
        n_overlap = len(det_pairs & learned_pairs)
        
        # Compute DAC for this attack
        if n_det == 0:
            dac_attack = 0.0
        else:
            dac_attack = n_overlap / n_det
        
        # Compute precision of learned mapping for this attack
        if n_learned == 0:
            precision_learned_wrt_det_attack = 0.0
        else:
            precision_learned_wrt_det_attack = n_overlap / n_learned
        
        # Coverage flags
        coverage_det = 1 if n_det > 0 else 0
        coverage_learned = 1 if n_learned > 0 else 0
        
        results.append({
            "attack_id": attack_id,
            "n_det_pairs": n_det,
            "n_learned_pairs": n_learned,
            "n_overlap_pairs": n_overlap,
            "dac_attack": dac_attack,
            "precision_learned_wrt_det_attack": precision_learned_wrt_det_attack,
            "coverage_det": coverage_det,
            "coverage_learned": coverage_learned,
        })
    
    df_result = pd.DataFrame(results)
    logger.info(f"Computed DAC per attack for {len(df_result)} attacks")
    
    return df_result


if __name__ == "__main__":
    import logging as logging_module
    
    logging_module.basicConfig(level=logging_module.INFO)
    det_path = Path("data/mappings/deterministic_attack_defense_lookup.csv")
    learned_path = Path("data/mappings/learned_attack_defense_mapping.csv")
    
    df_det = pd.read_csv(det_path)
    df_learned = pd.read_csv(learned_path)
    
    # Test global DAC
    metrics = compute_dac_between_mappings(df_det, df_learned)
    logger.info("Global DAC metrics: %s", metrics)
    
    # Test per-attack DAC
    df_dac_per_attack = compute_dac_per_attack(df_det, df_learned)
    logger.info(f"Per-attack DAC computed for {len(df_dac_per_attack)} attacks")
    logger.info(f"Average DAC per attack: {df_dac_per_attack['dac_attack'].mean():.4f}")

