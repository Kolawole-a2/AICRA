#!/usr/bin/env python3
"""
Reprocess All H3 Deliverables

This script:
1. Regenerates the learned/heuristic mapping with generic, broad parameters
2. Re-runs H3 evaluation
3. Updates all H3 deliverables (results, summaries, plots)

The learned mapping is designed to be:
- Generic and broad (not ransomware-specific)
- Uses ALL (or almost all) D3FEND controls
- Noisy and less aligned with ransomware defense
- Expected to perform worse than deterministic mapping
"""

import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

LOGGER = logging.getLogger(__name__)


def regenerate_learned_mapping():
    """Regenerate learned mapping with generic, broad parameters."""
    LOGGER.info("=" * 80)
    LOGGER.info("Step 1: Regenerating Learned Mapping")
    LOGGER.info("=" * 80)

    repo_root = Path(__file__).parent.parent

    # Try to use the heuristic mapping CLI
    cmd = [
        sys.executable,
        "-m",
        "aicra.mappings.heuristic_mapping",
        "--top-k",
        "10",
        "--min-similarity",
        "0.25",
        "--out",
        str(repo_root / "data" / "mappings" / "learned_mapping.csv"),
    ]

    # Try to auto-discover paths
    attack_candidates = [
        repo_root / "data" / "ontology" / "attack_techniques.csv",
        repo_root / "data" / "mitre" / "raw" / "enterprise-attack.json",
        repo_root / "mappings" / "data" / "mitre" / "raw" / "enterprise-attack.json",
    ]

    for candidate in attack_candidates:
        if candidate.exists():
            cmd.extend(["--attack", str(candidate)])
            LOGGER.info(f"Using ATT&CK data: {candidate}")
            break

    d3fend_candidates = [
        repo_root / "data" / "ontology" / "d3fend_controls.csv",
        repo_root / "data" / "mitre" / "raw" / "d3fend.csv",
        repo_root / "mappings" / "data" / "mitre" / "raw" / "d3fend.csv",
    ]

    for candidate in d3fend_candidates:
        if candidate.exists():
            cmd.extend(["--d3fend", str(candidate)])
            LOGGER.info(f"Using D3FEND data: {candidate}")
            break

    if "--attack" not in " ".join(cmd) or "--d3fend" not in " ".join(cmd):
        LOGGER.error("Could not find ATT&CK or D3FEND data files.")
        LOGGER.error("Please ensure one of the following exists:")
        LOGGER.error("  - data/ontology/attack_techniques.csv")
        LOGGER.error("  - data/ontology/d3fend_controls.csv")
        LOGGER.error("  - data/mitre/raw/enterprise-attack.json")
        LOGGER.error("  - data/mitre/raw/d3fend.csv")
        LOGGER.error("")
        LOGGER.error("Alternatively, run manually:")
        LOGGER.error(
            "  python -m aicra.mappings.heuristic_mapping --attack <path> --d3fend <path> --top-k 10 --min-similarity 0.25 --out data/mappings/learned_mapping.csv"
        )
        return False

    LOGGER.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

    if result.returncode != 0:
        LOGGER.error("Failed to regenerate learned mapping:")
        LOGGER.error(result.stderr)
        return False

    LOGGER.info("✓ Learned mapping regenerated successfully")
    LOGGER.info("")
    return True


def run_h3_evaluation():
    """Run H3 evaluation with regenerated learned mapping."""
    LOGGER.info("=" * 80)
    LOGGER.info("Step 2: Running H3 Evaluation")
    LOGGER.info("=" * 80)

    repo_root = Path(__file__).parent.parent

    cmd = [
        sys.executable,
        "-m",
        "aicra.experiments.h3_evaluation",
        "--config",
        str(repo_root / "config" / "h3_splits.yaml"),
        "--deterministic",
        str(
            repo_root / "data" / "mappings" / "deterministic_attack_defense_lookup.csv"
        ),
        "--learned",
        str(repo_root / "data" / "mappings" / "learned_mapping.csv"),
        "--output",
        str(repo_root / "results" / "H3_full_evaluation"),
    ]

    LOGGER.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)

    if result.returncode != 0:
        LOGGER.error("H3 evaluation failed:")
        LOGGER.error(result.stderr)
        return False

    LOGGER.info("✓ H3 evaluation completed successfully")
    LOGGER.info("")
    return True


def main():
    """Main entry point."""
    LOGGER.info("=" * 80)
    LOGGER.info("Reprocessing All H3 Deliverables")
    LOGGER.info("=" * 80)
    LOGGER.info("")
    LOGGER.info("This will:")
    LOGGER.info("  1. Regenerate learned_mapping.csv with generic, broad parameters")
    LOGGER.info("  2. Re-run H3 evaluation")
    LOGGER.info("  3. Update all H3 deliverables (results, summaries, plots)")
    LOGGER.info("")

    # Step 1: Regenerate learned mapping
    if not regenerate_learned_mapping():
        LOGGER.error("Failed to regenerate learned mapping. Exiting.")
        sys.exit(1)

    # Step 2: Run H3 evaluation
    if not run_h3_evaluation():
        LOGGER.error("Failed to run H3 evaluation. Exiting.")
        sys.exit(1)

    LOGGER.info("=" * 80)
    LOGGER.info("All H3 Deliverables Reprocessed Successfully")
    LOGGER.info("=" * 80)
    LOGGER.info("")
    LOGGER.info("Results are available at:")
    LOGGER.info("  - results/H3_full_evaluation/H3_full_results.json")
    LOGGER.info("  - results/H3_full_evaluation/H3_full_summary.md")
    LOGGER.info("  - results/H3_full_evaluation/plots/")


if __name__ == "__main__":
    main()
