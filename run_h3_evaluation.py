#!/usr/bin/env python3
"""
Main entry point for H3 evaluation experiment.

Usage:
    python run_h3_evaluation.py
    python -m aicra.experiments.h3_evaluation --config config/h3_splits.yaml
"""

import logging
import sys
from pathlib import Path

from aicra.experiments.h3_evaluation import run_h3_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    repo_root = Path(__file__).parent
    splits_config = repo_root / "config" / "h3_splits.yaml"
    det_candidates = [
        repo_root / "data" / "mappings" / "deterministic_attack_defense_lookup.csv",
        repo_root / "data" / "mappings" / "deterministic_lookup.csv",
    ]
    det_mapping = next((p for p in det_candidates if p.exists()), det_candidates[0])
    learned_mapping = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_candidates = [
        repo_root / "d3fend_reference_pairs.csv",
        repo_root / "data" / "ontology" / "d3fend_reference_pairs.csv",
    ]
    ref_pairs = next((p for p in ref_candidates if p.exists()), ref_candidates[0])
    output_dir = repo_root / "results" / "H3_full_evaluation"

    if not splits_config.exists():
        logger.error("Splits configuration not found: %s", splits_config)
        sys.exit(1)
    if not det_mapping.exists():
        logger.error("Deterministic mapping not found: %s", det_mapping)
        sys.exit(1)
    if not learned_mapping.exists():
        logger.error("Learned mapping not found: %s", learned_mapping)
        sys.exit(1)
    if not ref_pairs.exists():
        logger.warning(
            "External reference pairs not found at %s (DAC_external sections will be skipped)",
            ref_pairs,
        )
        ref_pairs = None

    try:
        results = run_h3_evaluation(
            splits_config_path=splits_config,
            det_mapping_path=det_mapping,
            learned_mapping_path=learned_mapping,
            ref_pairs_path=ref_pairs,
            output_dir=output_dir,
            repo_root=repo_root,
        )
        logger.info(
            "H3 evaluation complete: %s splits", len(results["splits_evaluated"])
        )
        logger.info("Results: %s", output_dir)
    except Exception as exc:
        logger.error("H3 evaluation failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
