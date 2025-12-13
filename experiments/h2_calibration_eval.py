#!/usr/bin/env python3
"""
Standardized H2 Experiment Entrypoint

Canonical script to run H2: Calibration and Cost-Aware Thresholding

Usage:
    python experiments/h2_calibration_eval.py [--output-dir artifacts/H2_calibration_thresholds]

This script:
- Uses AICRA_EMBER2024_DIR environment variable via get_ember2024_dir()
- Runs temporal calibration check (calibrate on earlier window, test on later)
- Writes outputs to artifacts/
- Logs seeds, configs, and timestamps
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aicra.experiments.h2_calibration_thresholds import run_h2_calibration_thresholds_experiment
from aicra.utils.data_paths import get_ember2024_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for H2 experiment."""
    parser = argparse.ArgumentParser(description="H2: Calibration and Cost-Aware Thresholding")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/H2_calibration_thresholds"),
        help="Output directory for results (default: artifacts/H2_calibration_thresholds)",
    )
    parser.add_argument(
        "--cost-fn",
        type=float,
        default=10.0,
        help="Cost of false negative (default: 10.0 for banking)",
    )
    parser.add_argument(
        "--cost-fp",
        type=float,
        default=1.0,
        help="Cost of false positive (default: 1.0)",
    )
    parser.add_argument(
        "--calibration-method",
        type=str,
        default="auto",
        choices=["platt", "isotonic", "auto"],
        help="Calibration method (default: auto)",
    )
    
    args = parser.parse_args()
    
    # Verify data directory
    try:
        ember_dir = get_ember2024_dir()
        logger.info(f"Using EMBER-2024 directory: {ember_dir}")
    except FileNotFoundError as e:
        logger.error(f"Data directory error: {e}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log experiment metadata
    repo_root = Path(__file__).parent.parent
    metadata = {
        "experiment": "H2",
        "hypothesis": "Isotonic calibration improves susceptibility score transferability",
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(ember_dir),
        "output_dir": str(args.output_dir),
        "cost_fn": args.cost_fn,
        "cost_fp": args.cost_fp,
        "calibration_method": args.calibration_method,
        "random_seed": 42,  # Standard seed
    }
    
    metadata_path = args.output_dir / "experiment_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved experiment metadata to {metadata_path}")
    
    # Run experiment
    logger.info("=" * 80)
    logger.info("Starting H2 Experiment")
    logger.info("=" * 80)
    
    try:
        metrics = run_h2_calibration_thresholds_experiment(
            output_dir=args.output_dir,
            cost_fn=args.cost_fn,
            cost_fp=args.cost_fp,
            calibration_method=args.calibration_method,
            repo_root=repo_root,
        )
        
        logger.info("=" * 80)
        logger.info("H2 Experiment Complete")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {args.output_dir}")
        
        if 'calibration' in metrics:
            cal = metrics['calibration']
            logger.info(f"Brier (uncalibrated): {cal.get('brier_uncalibrated', 'N/A'):.4f}")
            logger.info(f"Brier (calibrated): {cal.get('brier_calibrated', 'N/A'):.4f}")
            logger.info(f"ECE (uncalibrated): {cal.get('ece_uncalibrated', 'N/A'):.4f}")
            logger.info(f"ECE (calibrated): {cal.get('ece_calibrated', 'N/A'):.4f}")
        
        if 'improvement' in metrics:
            logger.info(f"Brier improvement: {metrics['improvement'].get('brier_improvement_pct', 'N/A'):.2f}%")
            logger.info(f"ECE improvement: {metrics['improvement'].get('ece_improvement_pct', 'N/A'):.2f}%")
        
        # Generate benchmark improvements report
        try:
            from aicra.utils.benchmark_reporter import generate_benchmark_improvements_table
            artifacts_dir = repo_root / "artifacts"
            generate_benchmark_improvements_table(
                h2_results_dir=args.output_dir,
                output_dir=artifacts_dir,
            )
            logger.info(f"Benchmark improvements report generated in {artifacts_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate benchmark report: {e}")
        
        return 0
        
    except Exception as e:
        logger.error(f"H2 experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

