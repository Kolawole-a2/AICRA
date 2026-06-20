#!/usr/bin/env python3
"""Run H1 and H2 experiments with proper error handling and output."""

import logging
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

def run_h1():
    """Run H1 experiment."""
    logger.info("=" * 80)
    logger.info("Running H1: Static PE Classification Reliability")
    logger.info("=" * 80)
    
    try:
        from aicra.experiments.h1_classification import run_h1_classification_experiment
        
        output_dir = Path("results/H1_classification")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {output_dir}")
        
        metrics = run_h1_classification_experiment(
            output_dir=output_dir,
            model_type="lgbm",
            operational_threshold=0.5,
            use_pe_features=True,
            repo_root=Path.cwd(),
        )
        
        logger.info("=" * 80)
        logger.info("H1 Experiment Complete!")
        logger.info("=" * 80)
        logger.info(f"AUROC: {metrics.get('auroc', 'N/A'):.4f}")
        logger.info(f"PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}")
        logger.info(f"Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"H1 experiment failed: {e}", exc_info=True)
        return False


def run_h2():
    """Run H2 experiment."""
    logger.info("=" * 80)
    logger.info("Running H2: Calibration and Cost-Aware Thresholding")
    logger.info("=" * 80)
    
    try:
        from aicra.experiments.h2_calibration_thresholds import run_h2_calibration_thresholds_experiment
        
        output_dir = Path("results/H2_calibration_thresholds")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {output_dir}")
        
        metrics = run_h2_calibration_thresholds_experiment(
            output_dir=output_dir,
            cost_fn=10.0,
            cost_fp=1.0,
            calibration_method="auto",
            repo_root=Path.cwd(),
        )
        
        logger.info("=" * 80)
        logger.info("H2 Experiment Complete!")
        logger.info("=" * 80)
        cal = metrics.get('calibration', {})
        logger.info(f"Brier (uncalibrated): {cal.get('brier_uncalibrated', 'N/A'):.4f}")
        logger.info(f"Brier (calibrated): {cal.get('brier_calibrated', 'N/A'):.4f}")
        logger.info(f"Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"H2 experiment failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    logger.info("Starting H1 and H2 experiments...")
    logger.info("")
    
    # Run H1
    h1_success = run_h1()
    
    if not h1_success:
        logger.error("H1 failed. Cannot run H2 (depends on H1).")
        return 1
    
    logger.info("")
    
    # Run H2
    h2_success = run_h2()
    
    if not h2_success:
        logger.error("H2 failed.")
        return 1
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("All experiments complete!")
    logger.info("=" * 80)
    logger.info("Next step: Run validation report generator:")
    logger.info("  python scripts/generate_praxis_validation_report.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
