"""
DAC Evaluation for Embedding-based Learned Mapping.

This module computes Defense-Attack Consistency (DAC) metrics comparing
the deterministic mapping with the learned embedding mapping.

For each attack_id:
    D_det = deterministic defenses
    D_learn = learned defenses (rank ≤ k)
    DAC = |D_det ∩ D_learn| / |D_det|
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_deterministic_mapping(mapping_path: Path) -> pd.DataFrame:
    """
    Load deterministic ATT&CK→D3FEND lookup table.

    Args:
        mapping_path: Path to deterministic_attack_defense_lookup.csv

    Returns:
        DataFrame with columns: attack_id, defense_id
    """
    LOGGER.info(f"Loading deterministic mapping from {mapping_path}")

    if not mapping_path.exists():
        raise FileNotFoundError(f"Deterministic lookup not found at {mapping_path}")

    df = pd.read_csv(mapping_path)

    # Validate required columns
    required_cols = ["attack_id", "defense_id"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter to correct mappings only if is_correct column exists
    if "is_correct" in df.columns:
        df = df[df["is_correct"] == 1]
        LOGGER.info(f"Filtered to {len(df)} correct mappings")

    # Select only required columns
    df = df[["attack_id", "defense_id"]].drop_duplicates()

    LOGGER.info(f"Loaded {len(df)} deterministic mappings")
    LOGGER.info(
        f"Unique attacks: {df['attack_id'].nunique()}, Unique defenses: {df['defense_id'].nunique()}"
    )

    return df


def load_learned_mapping(mapping_path: Path) -> pd.DataFrame:
    """
    Load learned embedding mapping.

    Args:
        mapping_path: Path to learned_embedding_attack_defense_mapping.csv

    Returns:
        DataFrame with columns: attack_id, defense_id, rank
    """
    LOGGER.info(f"Loading learned mapping from {mapping_path}")

    if not mapping_path.exists():
        raise FileNotFoundError(f"Learned mapping not found at {mapping_path}")

    df = pd.read_csv(mapping_path)

    # Validate required columns
    required_cols = ["attack_id", "defense_id"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Select only required columns (rank is optional but useful for filtering)
    df = df[["attack_id", "defense_id"]].drop_duplicates()

    LOGGER.info(f"Loaded {len(df)} learned mappings")
    LOGGER.info(
        f"Unique attacks: {df['attack_id'].nunique()}, Unique defenses: {df['defense_id'].nunique()}"
    )

    return df


def build_attack_to_defenses_dict(df: pd.DataFrame) -> dict[str, set[str]]:
    """
    Build dictionary mapping attack_id to set of defense_ids.

    Args:
        df: DataFrame with attack_id and defense_id columns

    Returns:
        Dictionary mapping attack_id to set of defense_ids
    """
    attack_to_defenses: dict[str, set[str]] = {}

    for _, row in df.iterrows():
        attack_id = str(row["attack_id"])
        defense_id = str(row["defense_id"])

        if attack_id not in attack_to_defenses:
            attack_to_defenses[attack_id] = set()

        attack_to_defenses[attack_id].add(defense_id)

    return attack_to_defenses


def compute_dac_per_attack(
    deterministic_mapping: pd.DataFrame,
    learned_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute DAC per attack_id.

    For each attack_id:
        D_det = deterministic defenses
        D_learn = learned defenses (all ranks)
        DAC = |D_det ∩ D_learn| / |D_det|

    Args:
        deterministic_mapping: DataFrame with attack_id and defense_id columns
        learned_mapping: DataFrame with attack_id and defense_id columns

    Returns:
        DataFrame with columns: attack_id, n_det_defenses, n_learn_defenses, n_overlap, dac
    """
    LOGGER.info("Computing DAC per attack")

    # Build dictionaries mapping attack_id to sets of defense_ids
    det_dict = build_attack_to_defenses_dict(deterministic_mapping)
    learn_dict = build_attack_to_defenses_dict(learned_mapping)

    # Get all unique attack_ids from both mappings
    all_attack_ids = set(det_dict.keys()) | set(learn_dict.keys())

    results = []

    for attack_id in all_attack_ids:
        # Get defense sets for this attack
        det_defenses = det_dict.get(attack_id, set())
        learn_defenses = learn_dict.get(attack_id, set())

        # Compute intersection
        overlap = det_defenses & learn_defenses

        # Compute DAC
        n_det = len(det_defenses)
        n_learn = len(learn_defenses)
        n_overlap = len(overlap)

        if n_det == 0:
            dac = 0.0
        else:
            dac = n_overlap / n_det

        results.append(
            {
                "attack_id": attack_id,
                "n_det_defenses": n_det,
                "n_learn_defenses": n_learn,
                "n_overlap": n_overlap,
                "dac": dac,
            }
        )

    df_result = pd.DataFrame(results)

    LOGGER.info(f"Computed DAC per attack for {len(df_result)} attacks")
    LOGGER.info(f"Average DAC: {df_result['dac'].mean():.4f}")
    LOGGER.info(f"Attacks with DAC > 0: {(df_result['dac'] > 0).sum()}")
    LOGGER.info(f"Attacks with DAC = 1.0: {(df_result['dac'] == 1.0).sum()}")

    return df_result


def save_dac_results(
    dac_results_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save DAC results to CSV.

    Args:
        dac_results_df: DataFrame with DAC results
        output_path: Path to save results CSV
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dac_results_df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved DAC results to {output_path}")

    # Log summary statistics
    avg_dac = dac_results_df["dac"].mean()
    median_dac = dac_results_df["dac"].median()
    max_dac = dac_results_df["dac"].max()
    min_dac = dac_results_df["dac"].min()

    LOGGER.info("DAC Summary Statistics:")
    LOGGER.info(f"  - Mean DAC: {avg_dac:.4f}")
    LOGGER.info(f"  - Median DAC: {median_dac:.4f}")
    LOGGER.info(f"  - Max DAC: {max_dac:.4f}")
    LOGGER.info(f"  - Min DAC: {min_dac:.4f}")
    LOGGER.info(f"  - Attacks with DAC = 1.0: {(dac_results_df['dac'] == 1.0).sum()}")
    LOGGER.info(f"  - Attacks with DAC = 0.0: {(dac_results_df['dac'] == 0.0).sum()}")


def evaluate_dac_embedding(
    deterministic_path: Path,
    learned_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Evaluate DAC for embedding-based learned mapping.

    This function:
    1. Loads deterministic and learned mappings
    2. Computes DAC per attack
    3. Saves results to CSV

    Args:
        deterministic_path: Path to deterministic_attack_defense_lookup.csv
        learned_path: Path to learned_embedding_attack_defense_mapping.csv
        output_path: Path to save DAC results CSV

    Returns:
        DataFrame with DAC results per attack
    """
    LOGGER.info("=" * 80)
    LOGGER.info("Evaluating DAC for Embedding-based Learned Mapping")
    LOGGER.info("=" * 80)

    # Step 1: Load mappings
    deterministic_mapping = load_deterministic_mapping(deterministic_path)
    learned_mapping = load_learned_mapping(learned_path)

    # Step 2: Compute DAC per attack
    dac_results_df = compute_dac_per_attack(deterministic_mapping, learned_mapping)

    # Step 3: Save results
    save_dac_results(dac_results_df, output_path)

    LOGGER.info("=" * 80)
    LOGGER.info("DAC evaluation completed successfully")
    LOGGER.info("=" * 80)

    return dac_results_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Default paths
    deterministic_path = Path("data/mappings/deterministic_attack_defense_lookup.csv")
    learned_path = Path("data/mappings/learned_embedding_attack_defense_mapping.csv")
    output_path = Path("results/dac_embedding_comparison.csv")

    # Evaluate DAC
    dac_results_df = evaluate_dac_embedding(
        deterministic_path=deterministic_path,
        learned_path=learned_path,
        output_path=output_path,
    )

    LOGGER.info(f"Successfully computed DAC for {len(dac_results_df)} attacks")
