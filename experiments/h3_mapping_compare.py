#!/usr/bin/env python3
"""
Standardized H3 Experiment Entrypoint

Canonical script to run H3: Deterministic vs Learned ATT&CK–D3FEND Mapping Comparison

Usage:
    python experiments/h3_mapping_compare.py [--output-dir artifacts/H3_full_evaluation]

This script:
- Compares deterministic vs learned mappings
- Includes learned == deterministic bug check
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

from aicra.experiments.h3_evaluation import run_h3_evaluation
from aicra.utils.data_paths import get_ember2024_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for H3 experiment."""
    parser = argparse.ArgumentParser(
        description="H3: Deterministic vs Learned Mapping Comparison"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/H3_full_evaluation"),
        help="Output directory for results (default: artifacts/H3_full_evaluation)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/h3_splits.yaml"),
        help="H3 splits configuration file (default: config/h3_splits.yaml)",
    )

    args = parser.parse_args()

    # Verify data directory (for reference, H3 uses pre-computed risk scores)
    try:
        ember_dir = get_ember2024_dir()
        logger.info(f"EMBER-2024 directory available: {ember_dir}")
    except FileNotFoundError as e:
        logger.warning(
            f"EMBER-2024 directory not found (H3 uses pre-computed risk scores): {e}"
        )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Log experiment metadata
    repo_root = Path(__file__).parent.parent
    metadata = {
        "experiment": "H3",
        "hypothesis": "Deterministic ATT&CK–D3FEND lookup beats learned mapping",
        "timestamp": datetime.now().isoformat(),
        "output_dir": str(args.output_dir),
        "config_file": str(args.config),
        "random_seed": 42,  # Standard seed
    }

    metadata_path = args.output_dir / "experiment_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved experiment metadata to {metadata_path}")

    # Run experiment
    logger.info("=" * 80)
    logger.info("Starting H3 Experiment")
    logger.info("=" * 80)

    try:
        results = run_h3_evaluation(
            output_dir=args.output_dir,
            config_path=args.config,
            repo_root=repo_root,
        )

        logger.info("=" * 80)
        logger.info("H3 Experiment Complete")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {args.output_dir}")

        if "aggregated_metrics" in results:
            agg = results["aggregated_metrics"]
            if "improvements" in agg:
                imp = agg["improvements"]
                logger.info(
                    f"Coverage improvement: {imp.get('coverage_improvement_pct', 'N/A'):.2f}%"
                )
                logger.info(
                    f"Variance reduction: {imp.get('variance_reduction_pct', 'N/A'):.2f}%"
                )
                logger.info(
                    f"Alert fatigue reduction: {imp.get('estimated_fatigue_reduction_pct', 'N/A'):.2f}%"
                )

        # Generate benchmark improvements report
        try:
            from aicra.utils.benchmark_reporter import (
                generate_benchmark_improvements_table,
            )

            artifacts_dir = repo_root / "artifacts"
            generate_benchmark_improvements_table(
                h3_results_dir=args.output_dir,
                output_dir=artifacts_dir,
            )
            logger.info(f"Benchmark improvements report generated in {artifacts_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate benchmark report: {e}")

        return 0

    except Exception as e:
        logger.error(f"H3 experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
