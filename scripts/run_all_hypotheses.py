#!/usr/bin/env python3
"""
Orchestration script to run all hypothesis experiments (H1, H2, H3).

This script runs the canonical experiments for all three hypotheses in order:
1. H1: Static PE Classification Reliability
2. H2: Calibration and Cost-Aware Thresholding
3. H3: Deterministic vs Learned Mapping Comparison

Usage:
    python scripts/run_all_hypotheses.py [--skip-h1] [--skip-h2] [--skip-h3]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_h1_experiment(repo_root: Path, output_dir: Optional[Path] = None) -> Path:
    """Run H1 classification experiment."""
    logger.info("=" * 80)
    logger.info("Running H1: Static PE Classification Reliability")
    logger.info("=" * 80)
    
    from aicra.experiments.h1_classification import run_h1_classification_experiment
    
    if output_dir is None:
        output_dir = repo_root / "results" / "H1_classification"
    
    metrics = run_h1_classification_experiment(
        output_dir=output_dir,
        model_type="lgbm",
        operational_threshold=0.5,
        use_pe_features=True,
        repo_root=repo_root,
    )
    
    logger.info(f"H1 experiment completed. Results: {output_dir}")
    return output_dir


def run_h2_experiment(repo_root: Path, output_dir: Optional[Path] = None) -> Path:
    """Run H2 calibration and thresholding experiment."""
    logger.info("=" * 80)
    logger.info("Running H2: Calibration and Cost-Aware Thresholding")
    logger.info("=" * 80)
    
    from aicra.experiments.h2_calibration_thresholds import run_h2_calibration_thresholds_experiment
    
    if output_dir is None:
        output_dir = repo_root / "results" / "H2_calibration_thresholds"
    
    metrics = run_h2_calibration_thresholds_experiment(
        output_dir=output_dir,
        cost_fn=10.0,
        cost_fp=1.0,
        calibration_method="auto",
        repo_root=repo_root,
    )
    
    logger.info(f"H2 experiment completed. Results: {output_dir}")
    return output_dir


def run_h3_experiment(repo_root: Path, output_dir: Optional[Path] = None) -> Path:
    """Run H3 mapping comparison experiment."""
    logger.info("=" * 80)
    logger.info("Running H3: Deterministic vs Learned Mapping Comparison")
    logger.info("=" * 80)
    
    from aicra.experiments.h3_evaluation import run_h3_evaluation
    
    splits_config = repo_root / "config" / "h3_splits.yaml"
    det_mapping = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    learned_mapping = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_pairs = repo_root / "d3fend_reference_pairs.csv"
    
    if output_dir is None:
        output_dir = repo_root / "results" / "H3_full_evaluation"
    
    # Validate inputs
    if not splits_config.exists():
        logger.error(f"Splits configuration not found: {splits_config}")
        raise FileNotFoundError(f"Please create {splits_config}")
    
    if not det_mapping.exists():
        logger.error(f"Deterministic mapping not found: {det_mapping}")
        raise FileNotFoundError(f"Deterministic mapping not found: {det_mapping}")
    
    if not learned_mapping.exists():
        logger.warning(f"Learned mapping not found: {learned_mapping}")
        logger.warning("Attempting to generate learned mapping...")
        # Try to generate learned mapping
        try:
            from generate_learned_mapping import build_learned_embedding_mapping
            logger.info("Generating learned mapping...")
            # This would need to be implemented based on your actual generation script
            raise NotImplementedError("Please generate learned_mapping.csv first using generate_learned_mapping.py")
        except Exception as e:
            logger.error(f"Failed to generate learned mapping: {e}")
            raise
    
    if not ref_pairs.exists():
        logger.warning(f"Reference pairs not found: {ref_pairs}")
        logger.warning("Creating reference pairs from YAML...")
        # Create reference pairs
        from scripts.create_reference_pairs import create_reference_pairs_csv
        yaml_path = repo_root / "data" / "lookups" / "attack_to_d3fend.yaml"
        create_reference_pairs_csv(yaml_path, ref_pairs)
    
    results = run_h3_evaluation(
        splits_config_path=splits_config,
        det_mapping_path=det_mapping,
        learned_mapping_path=learned_mapping,
        ref_pairs_path=ref_pairs,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    
    logger.info(f"H3 experiment completed. Results: {output_dir}")
    return output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all hypothesis experiments (H1, H2, H3)"
    )
    parser.add_argument(
        "--skip-h1",
        action="store_true",
        help="Skip H1 experiment",
    )
    parser.add_argument(
        "--skip-h2",
        action="store_true",
        help="Skip H2 experiment",
    )
    parser.add_argument(
        "--skip-h3",
        action="store_true",
        help="Skip H3 experiment",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (default: current directory)",
    )
    
    args = parser.parse_args()
    
    repo_root = args.repo_root if args.repo_root else Path.cwd()
    
    logger.info("=" * 80)
    logger.info("AICRA Hypothesis Experiments Orchestration")
    logger.info("=" * 80)
    logger.info(f"Repository root: {repo_root}")
    logger.info("")
    
    results = {}
    
    try:
        # Run H1
        if not args.skip_h1:
            h1_output = run_h1_experiment(repo_root)
            results["H1"] = {
                "output_dir": str(h1_output),
                "metrics": str(h1_output / "metrics.json"),
                "summary": str(h1_output / "summary.md"),
            }
        else:
            logger.info("Skipping H1 experiment")
        
        # Run H2
        if not args.skip_h2:
            h2_output = run_h2_experiment(repo_root)
            results["H2"] = {
                "output_dir": str(h2_output),
                "metrics": str(h2_output / "metrics.json"),
                "summary": str(h2_output / "summary.md"),
            }
        else:
            logger.info("Skipping H2 experiment")
        
        # Run H3
        if not args.skip_h3:
            h3_output = run_h3_experiment(repo_root)
            results["H3"] = {
                "output_dir": str(h3_output),
                "results_json": str(h3_output / "H3_full_results.json"),
                "summary": str(h3_output / "H3_full_summary.md"),
            }
        else:
            logger.info("Skipping H3 experiment")
        
        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("All Experiments Complete")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Results Summary:")
        logger.info("")
        
        for hypothesis, paths in results.items():
            logger.info(f"{hypothesis}:")
            for key, path in paths.items():
                logger.info(f"  {key}: {path}")
            logger.info("")
        
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
