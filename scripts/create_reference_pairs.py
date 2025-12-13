#!/usr/bin/env python3
"""
Create canonical ATT&CK-D3FEND reference pairs from YAML mapping.

This script converts the authoritative YAML mapping (data/lookups/attack_to_d3fend.yaml)
into a CSV file that serves as the canonical reference for H3 evaluation.

The reference pairs are the ground-truth mappings that should be used to validate
both deterministic and learned mappings.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_yaml_mapping(yaml_path: Path) -> dict[str, list[str]]:
    """Load ATT&CK to D3FEND mapping from YAML file."""
    logger.info(f"Loading YAML mapping from {yaml_path}")

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML mapping not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    mappings = data.get("mappings", {})
    logger.info(f"Loaded {len(mappings)} technique mappings")

    return mappings


def create_reference_pairs_csv(yaml_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Create canonical reference pairs CSV from YAML mapping.

    Args:
        yaml_path: Path to attack_to_d3fend.yaml
        output_path: Path to output CSV file

    Returns:
        DataFrame with columns: technique_id, control_id
    """
    mappings = load_yaml_mapping(yaml_path)

    rows = []
    for technique_id, controls in mappings.items():
        for control_id in controls:
            rows.append({"technique_id": technique_id, "control_id": control_id})

    df = pd.DataFrame(rows)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Created reference pairs CSV with {len(df)} pairs: {output_path}")

    return df


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent

    yaml_path = repo_root / "data" / "lookups" / "attack_to_d3fend.yaml"
    output_path = repo_root / "data" / "ontology" / "d3fend_reference_pairs.csv"

    # Also create in root for backward compatibility
    output_path_root = repo_root / "d3fend_reference_pairs.csv"

    try:
        df = create_reference_pairs_csv(yaml_path, output_path)

        # Also save to root for backward compatibility
        df.to_csv(output_path_root, index=False)
        logger.info(f"Also saved to root: {output_path_root}")

        logger.info("=" * 80)
        logger.info("Reference pairs created successfully!")
        logger.info(f"  Total pairs: {len(df)}")
        logger.info(f"  Techniques: {df['technique_id'].nunique()}")
        logger.info(f"  Controls: {df['control_id'].nunique()}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Failed to create reference pairs: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
