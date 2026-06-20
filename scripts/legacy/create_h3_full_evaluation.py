#!/usr/bin/env python3
"""
Script to create H3_full_evaluation folder with all required outputs.
"""

import sys
import logging
from pathlib import Path

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('h3_evaluation_run.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

try:
    logger.info("=" * 80)
    logger.info("Creating H3_full_evaluation folder")
    logger.info("=" * 80)
    
    repo_root = Path(__file__).parent
    logger.info(f"Repository root: {repo_root}")
    
    # Import the evaluation function
    from aicra.experiments.h3_evaluation import run_h3_evaluation
    import yaml
    
    # Set up paths
    splits_config_path = repo_root / "config" / "h3_splits.yaml"
    det_mapping_path = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    learned_mapping_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_pairs_path = repo_root / "d3fend_reference_pairs.csv"
    output_dir = repo_root / "results" / "H3_full_evaluation"
    
    logger.info(f"Splits config: {splits_config_path} (exists: {splits_config_path.exists()})")
    logger.info(f"Deterministic mapping: {det_mapping_path} (exists: {det_mapping_path.exists()})")
    logger.info(f"Learned mapping: {learned_mapping_path} (exists: {learned_mapping_path.exists()})")
    logger.info(f"Reference pairs: {ref_pairs_path} (exists: {ref_pairs_path.exists()})")
    logger.info(f"Output directory: {output_dir}")
    
    # Validate inputs
    if not splits_config_path.exists():
        logger.error(f"Splits configuration not found: {splits_config_path}")
        sys.exit(1)
    
    if not det_mapping_path.exists():
        logger.error(f"Deterministic mapping not found: {det_mapping_path}")
        sys.exit(1)
    
    if not learned_mapping_path.exists():
        logger.error(f"Learned mapping not found: {learned_mapping_path}")
        sys.exit(1)
    
    if not ref_pairs_path.exists():
        logger.error(f"Reference pairs not found: {ref_pairs_path}")
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("Running H3 evaluation...")
    logger.info("=" * 80)
    
    # Run evaluation
    results = run_h3_evaluation(
        splits_config_path=splits_config_path,
        det_mapping_path=det_mapping_path,
        learned_mapping_path=learned_mapping_path,
        ref_pairs_path=ref_pairs_path,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    
    logger.info("=" * 80)
    logger.info("H3 Evaluation completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    
    # Verify outputs
    json_path = output_dir / "H3_full_results.json"
    md_path = output_dir / "H3_full_summary.md"
    plots_dir = output_dir / "plots"
    
    logger.info("\nVerifying outputs:")
    logger.info(f"  - H3_full_results.json: {json_path.exists()}")
    logger.info(f"  - H3_full_summary.md: {md_path.exists()}")
    logger.info(f"  - plots/ directory: {plots_dir.exists()}")
    
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        logger.info(f"  - Plot files found: {len(plot_files)}")
        for plot_file in plot_files:
            logger.info(f"    - {plot_file.name}")
    
    logger.info("\n" + "=" * 80)
    logger.info("SUCCESS! H3_full_evaluation folder created with all outputs.")
    logger.info("=" * 80)
    
except Exception as e:
    logger.error(f"Error: {e}", exc_info=True)
    sys.exit(1)
