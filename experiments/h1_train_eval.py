#!/usr/bin/env python3
"""
Standardized H1 Experiment Entrypoint

Canonical script to run H1: Static PE Classification Reliability

Usage:
    python experiments/h1_train_eval.py [--output-dir artifacts/H1_classification]

This script:
- Uses AICRA_EMBER2024_DIR environment variable via get_ember2024_dir()
- Runs time-ordered split and out-of-family test
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

from aicra.experiments.h1_classification import run_h1_classification_experiment
from aicra.utils.data_paths import get_ember2024_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for H1 experiment."""
    parser = argparse.ArgumentParser(
        description="H1: Static PE Classification Reliability"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/H1_classification"),
        help="Output directory for results (default: artifacts/H1_classification)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="lgbm",
        choices=["lgbm", "ffnn"],
        help="Model type (default: lgbm)",
    )
    parser.add_argument(
        "--operational-threshold",
        type=float,
        default=0.5,
        help="Operational threshold (default: 0.5, will be optimized for banking)",
    )
    parser.add_argument(
        "--use-pe-features",
        action="store_true",
        default=True,
        help="Use PE static features (default: True)",
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
        "experiment": "H1",
        "hypothesis": "Static PE features enable reliable ransomware classification",
        "timestamp": datetime.now().isoformat(),
        "data_dir": str(ember_dir),
        "output_dir": str(args.output_dir),
        "model_type": args.model_type,
        "operational_threshold": args.operational_threshold,
        "use_pe_features": args.use_pe_features,
        "random_seed": 42,  # Standard seed
    }

    metadata_path = args.output_dir / "experiment_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved experiment metadata to {metadata_path}")

    # Run experiment
    logger.info("=" * 80)
    logger.info("Starting H1 Experiment")
    logger.info("=" * 80)

    try:
        metrics = run_h1_classification_experiment(
            output_dir=args.output_dir,
            model_type=args.model_type,
            operational_threshold=args.operational_threshold,
            use_pe_features=args.use_pe_features,
            repo_root=repo_root,
        )

        logger.info("=" * 80)
        logger.info("H1 Experiment Complete")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"AUROC: {metrics.get('auroc', 'N/A'):.4f}")
        logger.info(f"Precision: {metrics.get('precision', 'N/A'):.4f}")
        logger.info(f"Recall: {metrics.get('recall', 'N/A'):.4f}")
        logger.info(f"F1: {metrics.get('f1', 'N/A'):.4f}")

        if "improvement" in metrics:
            logger.info(
                f"AUROC improvement: {metrics['improvement'].get('auroc_pct', 'N/A'):.2f}%"
            )

        # Generate benchmark improvements report
        try:
            from aicra.utils.benchmark_reporter import (
                generate_benchmark_improvements_table,
            )

            artifacts_dir = repo_root / "artifacts"
            generate_benchmark_improvements_table(
                h1_results_dir=args.output_dir,
                output_dir=artifacts_dir,
            )
            logger.info(f"Benchmark improvements report generated in {artifacts_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate benchmark report: {e}")

        return 0

    except Exception as e:
        logger.error(f"H1 experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
