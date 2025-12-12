#!/usr/bin/env python3
"""
Main entry point for H3 evaluation experiment.

This script runs the canonical H3 evaluation pipeline that compares
deterministic vs learned ATT&CK–D3FEND mappings across all evaluation splits.

Usage:
    python run_h3_evaluation.py [--splits-config config/h3_splits.yaml] [--output results/H3_full_evaluation]
"""

import logging
import sys
from pathlib import Path

from aicra.experiments.h3_evaluation import run_h3_evaluation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run H3 evaluation experiment."""
    logger.info("=" * 80)
    logger.info("H3 Evaluation: Deterministic vs Learned Mapping Comparison")
    logger.info("=" * 80)
    
    # Default paths
    repo_root = Path(__file__).parent
    splits_config = repo_root / "config" / "h3_splits.yaml"
    det_mapping = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    learned_mapping = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_pairs = repo_root / "d3fend_reference_pairs.csv"
    output_dir = repo_root / "results" / "H3_full_evaluation"
    
    # Validate inputs
    if not splits_config.exists():
        logger.error(f"Splits configuration not found: {splits_config}")
        logger.error("Please create config/h3_splits.yaml with your evaluation splits")
        sys.exit(1)
    
    if not det_mapping.exists():
        logger.error(f"Deterministic mapping not found: {det_mapping}")
        sys.exit(1)
    
    if not learned_mapping.exists():
        logger.error(f"Learned mapping not found: {learned_mapping}")
        sys.exit(1)
    
    if not ref_pairs.exists():
        logger.error(f"Reference pairs not found: {ref_pairs}")
        sys.exit(1)
    
    # Run evaluation
    try:
        results = run_h3_evaluation(
            splits_config_path=splits_config,
            det_mapping_path=det_mapping,
            learned_mapping_path=learned_mapping,
            ref_pairs_path=ref_pairs,
            output_dir=output_dir,
            repo_root=repo_root,
        )
        
        logger.info("=" * 80)
        logger.info("H3 Evaluation completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Evaluated {len(results['splits_evaluated'])} splits")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"H3 evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
