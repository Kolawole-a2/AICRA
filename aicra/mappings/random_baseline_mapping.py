"""
Random Baseline Mapping Generator

Generates a random baseline mapping for ATT&CK→D3FEND pairs.
This provides a lower-bound comparison for DAC evaluation.

The random baseline:
- Uses the same set of attacks as the deterministic mapping
- Randomly assigns defenses to each attack
- Provides a baseline DAC (expected to be low) for comparison
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def build_random_baseline_mapping(
    deterministic_mapping: pd.DataFrame,
    seed: int = 17,
    controls_per_attack: int = 1,
) -> pd.DataFrame:
    """
    Build a random baseline mapping for contrast.

    Uses same attacks as deterministic but randomly assigns defenses.

    Args:
        deterministic_mapping: DataFrame with technique_id/control_id or attack_id/defense_id columns
        seed: Random seed for reproducibility
        controls_per_attack: Number of random controls to assign per attack (default: 1)

    Returns:
        DataFrame with random baseline mapping (technique_id, control_id columns)
    """
    LOGGER.info("Building random baseline mapping...")

    # Normalize column names
    df = deterministic_mapping.copy()
    if "attack_id" in df.columns and "technique_id" not in df.columns:
        df = df.rename(columns={"attack_id": "technique_id"})
    if "defense_id" in df.columns and "control_id" not in df.columns:
        df = df.rename(columns={"defense_id": "control_id"})

    if "technique_id" not in df.columns or "control_id" not in df.columns:
        raise ValueError(
            f"Deterministic mapping must have technique_id/control_id or attack_id/defense_id columns. "
            f"Found: {list(df.columns)}"
        )

    rng = np.random.default_rng(seed)
    unique_attacks = df["technique_id"].unique()
    unique_defenses = df["control_id"].unique()

    if len(unique_defenses) == 0:
        raise ValueError("No defenses found in deterministic mapping")

    LOGGER.info(f"  Unique attacks: {len(unique_attacks)}")
    LOGGER.info(f"  Unique defenses: {len(unique_defenses)}")
    LOGGER.info(f"  Controls per attack: {controls_per_attack}")

    rows = []
    for attack_id in unique_attacks:
        # Choose random defenses for this attack
        n_choices = min(controls_per_attack, len(unique_defenses))
        chosen_defenses = rng.choice(unique_defenses, size=n_choices, replace=False)
        for defense_id in chosen_defenses:
            rows.append({"technique_id": attack_id, "control_id": defense_id})

    random_mapping = pd.DataFrame(rows).drop_duplicates()

    LOGGER.info(f"  Generated {len(random_mapping)} random pairs")
    LOGGER.info(
        f"  Average controls per attack: {len(random_mapping) / len(unique_attacks):.2f}"
    )

    return random_mapping


def generate_and_save_random_baseline(
    deterministic_path: Path,
    out_csv: Path,
    out_parquet: Path | None = None,
    seed: int = 17,
    controls_per_attack: int = 1,
) -> pd.DataFrame:
    """
    Generate random baseline mapping from deterministic lookup and save to files.

    Args:
        deterministic_path: Path to deterministic_attack_defense_lookup.csv
        out_csv: Path to save CSV output
        out_parquet: Optional path to save Parquet output
        seed: Random seed for reproducibility
        controls_per_attack: Number of random controls per attack

    Returns:
        DataFrame with random baseline mapping
    """
    LOGGER.info("=" * 80)
    LOGGER.info("GENERATING RANDOM BASELINE MAPPING")
    LOGGER.info("=" * 80)

    # Load deterministic mapping
    if not deterministic_path.exists():
        raise FileNotFoundError(
            f"Deterministic mapping not found: {deterministic_path}"
        )

    det_df = pd.read_csv(deterministic_path)
    LOGGER.info(f"Loaded deterministic mapping from: {deterministic_path}")
    LOGGER.info(f"  Shape: {det_df.shape}")
    LOGGER.info(f"  Columns: {list(det_df.columns)}")

    # Filter by is_correct if present
    if "is_correct" in det_df.columns:
        det_df = det_df[det_df["is_correct"] == 1]
        LOGGER.info(f"  After filtering is_correct=1: {det_df.shape}")

    # Generate random baseline
    random_mapping = build_random_baseline_mapping(
        deterministic_mapping=det_df,
        seed=seed,
        controls_per_attack=controls_per_attack,
    )

    # Ensure output directory exists
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    random_mapping.to_csv(out_csv, index=False)
    LOGGER.info(f"Saved random baseline mapping to: {out_csv}")

    # Save to Parquet if requested
    if out_parquet is not None:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        random_mapping.to_parquet(out_parquet, index=False)
        LOGGER.info(f"Saved random baseline mapping to: {out_parquet}")

    LOGGER.info("=" * 80)
    LOGGER.info("Random baseline mapping generation completed")
    LOGGER.info("=" * 80)

    return random_mapping


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    det_path = Path("data/mappings/deterministic_attack_defense_lookup.csv")
    out_csv = Path("data/mappings/random_baseline_attack_defense_mapping.csv")
    out_parquet = Path("data/mappings/random_baseline_attack_defense_mapping.parquet")

    generate_and_save_random_baseline(
        deterministic_path=det_path,
        out_csv=out_csv,
        out_parquet=out_parquet,
        seed=17,
        controls_per_attack=1,
    )














