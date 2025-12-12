#!/usr/bin/env python3
"""
Regenerate Learned Mapping for H3 Evaluation

This script regenerates the learned/heuristic mapping with generic, broad parameters
that are NOT ransomware-specific. The goal is to create a mapping that:
- Uses ALL (or almost all) D3FEND controls
- Is generic and noisy
- Performs worse than the deterministic ransomware-focused mapping

This supports the H3 hypothesis that deterministic mappings are better for
ransomware defense than generic learned mappings.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aicra.mappings.heuristic_mapping import (
    HeuristicMappingConfig,
    build_heuristic_mapping,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

LOGGER = logging.getLogger(__name__)


def main():
    """Regenerate learned mapping with generic, broad parameters."""
    LOGGER.info("=" * 80)
    LOGGER.info("Regenerating Learned Mapping for H3 Evaluation")
    LOGGER.info("=" * 80)
    LOGGER.info("")
    LOGGER.info("This will create a GENERIC, BROAD mapping that:")
    LOGGER.info("  - Uses top_k=10 controls per technique (broad coverage)")
    LOGGER.info("  - Uses min_similarity=0.25 (low threshold, noisy)")
    LOGGER.info("  - Is NOT ransomware-specific")
    LOGGER.info("  - Expected to perform WORSE than deterministic mapping")
    LOGGER.info("")
    
    # Find ontology files
    repo_root = Path(__file__).parent.parent
    
    # Try common locations for ATT&CK techniques
    attack_candidates = [
        repo_root / "data" / "ontology" / "attack_techniques.csv",
        repo_root / "data" / "mitre" / "raw" / "enterprise-attack.json",
    ]
    
    attack_path = None
    for candidate in attack_candidates:
        if candidate.exists():
            attack_path = candidate
            LOGGER.info(f"Found ATT&CK data at: {attack_path}")
            break
    
    if attack_path is None:
        LOGGER.error("Could not find ATT&CK techniques file.")
        LOGGER.error("Expected locations:")
        for candidate in attack_candidates:
            LOGGER.error(f"  - {candidate}")
        sys.exit(1)
    
    # Try common locations for D3FEND controls
    d3fend_candidates = [
        repo_root / "data" / "ontology" / "d3fend_controls.csv",
        repo_root / "data" / "mitre" / "raw" / "d3fend.csv",
    ]
    
    d3fend_path = None
    for candidate in d3fend_candidates:
        if candidate.exists():
            d3fend_path = candidate
            LOGGER.info(f"Found D3FEND data at: {d3fend_path}")
            break
    
    if d3fend_path is None:
        LOGGER.error("Could not find D3FEND controls file.")
        LOGGER.error("Expected locations:")
        for candidate in d3fend_candidates:
            LOGGER.error(f"  - {candidate}")
        sys.exit(1)
    
    # Create config for generic, broad mapping
    config = HeuristicMappingConfig(
        top_k=10,  # Broad: map to many controls
        min_similarity=0.25,  # Low threshold: include more controls (noisier)
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        seed=42,
    )
    
    LOGGER.info("")
    LOGGER.info("Building generic heuristic mapping...")
    LOGGER.info(f"  top_k: {config.top_k}")
    LOGGER.info(f"  min_similarity: {config.min_similarity}")
    LOGGER.info("")
    
    # Build mapping (validation happens inside build_heuristic_mapping)
    mapping_df = build_heuristic_mapping(
        attack_path=attack_path,
        d3fend_path=d3fend_path,
        config=config,
    )
    
    # Save to learned_mapping.csv
    output_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(output_path, index=False)
    
    LOGGER.info("")
    LOGGER.info("=" * 80)
    LOGGER.info("Learned Mapping Regeneration Complete")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Saved to: {output_path}")
    LOGGER.info(f"Total mappings: {len(mapping_df)}")
    LOGGER.info(f"Unique techniques: {mapping_df['technique_id'].nunique()}")
    LOGGER.info(f"Unique controls: {mapping_df['control_id'].nunique()}")
    LOGGER.info(f"Average mappings per technique: {len(mapping_df) / mapping_df['technique_id'].nunique():.2f}")
    LOGGER.info("")
    LOGGER.info("Next step: Run H3 evaluation:")
    LOGGER.info("  python -m aicra.experiments.h3_evaluation")


if __name__ == "__main__":
    main()
