#!/usr/bin/env python3
"""
Fix H3 plots by ensuring learned mapping has sufficient diversity.

This script:
1. Regenerates learned mapping with top_k=5 for more diversity
2. Verifies reference pairs are correct
3. Runs diagnostic to check overlap
4. Re-runs H3 evaluation to generate new plots
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 80)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stdout:
            logger.error(f"stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"stderr: {e.stderr}")
        return False


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent

    logger.info("=" * 80)
    logger.info("FIXING H3 PLOTS - Ensuring Mapping Diversity")
    logger.info("=" * 80)

    # Step 1: Regenerate learned mapping with higher diversity
    logger.info("\nStep 1: Regenerating learned mapping with top_k=5...")
    success = run_command(
        [
            sys.executable,
            "-m",
            "aicra.mapping.heuristic_mapping",
            "--top-k",
            "5",
            "--min-similarity",
            "0.35",
            "--out",
            "data/mappings/learned_mapping.csv",
        ],
        "Regenerate learned mapping",
    )
    if not success:
        logger.error("Failed to regenerate learned mapping")
        return 1

    # Step 2: Verify/Create reference pairs
    logger.info("\nStep 2: Verifying reference pairs...")
    ref_pairs_path = repo_root / "d3fend_reference_pairs.csv"
    if not ref_pairs_path.exists():
        logger.info("Reference pairs not found, creating them...")
        success = run_command(
            [sys.executable, "scripts/create_reference_pairs.py"],
            "Create reference pairs",
        )
        if not success:
            logger.warning("Failed to create reference pairs, continuing anyway...")
    else:
        logger.info(f"Reference pairs found at {ref_pairs_path}")

    # Step 3: Run diagnostic
    logger.info("\nStep 3: Running mapping overlap diagnostic...")
    success = run_command(
        [sys.executable, "scripts/diagnose_mapping_overlap.py"],
        "Diagnose mapping overlap",
    )
    if not success:
        logger.warning("Diagnostic failed, continuing anyway...")

    # Step 4: Re-run H3 evaluation
    logger.info("\nStep 4: Re-running H3 evaluation to generate new plots...")
    h3_config = repo_root / "config" / "h3_splits.yaml"
    if h3_config.exists():
        success = run_command(
            [
                sys.executable,
                "-m",
                "aicra.experiments.h3_evaluation",
                "--config",
                str(h3_config),
            ],
            "Run H3 evaluation",
        )
    else:
        logger.warning(f"H3 config not found at {h3_config}, trying without config...")
        success = run_command(
            [sys.executable, "-m", "aicra.experiments.h3_evaluation"],
            "Run H3 evaluation (no config)",
        )

    if not success:
        logger.error("Failed to run H3 evaluation")
        return 1

    logger.info("\n" + "=" * 80)
    logger.info("H3 PLOTS FIX COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nCheck the following for updated plots:")
    logger.info("  - results/H3_full_evaluation/plots/")
    logger.info("\nKey files:")
    logger.info("  - results/H3_full_evaluation/H3_full_results.json")
    logger.info("  - results/H3_full_evaluation/H3_full_summary.md")
    logger.info("  - results/H3_diagnostics/mapping_overlap.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
