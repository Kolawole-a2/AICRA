#!/usr/bin/env python3
"""
H3 Praxis Experiment Runner - Enhanced CLI for Doctor of Engineering Praxis Project

This script provides an enhanced interface for running the H3 experiment that validates
the Defense–Attack Consistency (DAC) metric and compares deterministic vs learned
ATT&CK–D3FEND mappings.

Before running, you will be prompted to ensure your deterministic mapping CSV is uploaded.
"""

import sys
import logging
from pathlib import Path

from aicra.experiments.h3_evaluation import run_h3_evaluation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_and_prompt_for_deterministic_mapping(
    default_path: Path,
    repo_root: Path
) -> Path:
    """
    Check if deterministic mapping exists, and prompt user if not found.
    
    Args:
        default_path: Default path where deterministic mapping should be
        repo_root: Repository root directory
        
    Returns:
        Path to deterministic mapping CSV
    """
    if default_path.exists():
        logger.info(f"✓ Found deterministic mapping at: {default_path}")
        return default_path
    
    logger.info("=" * 80)
    logger.info("DETERMINISTIC MAPPING NOT FOUND")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Please upload your deterministic ATT&CK–D3FEND mapping CSV file.")
    logger.info("")
    logger.info("Expected location:")
    logger.info(f"  {default_path}")
    logger.info("")
    logger.info("The file should contain columns:")
    logger.info("  - technique_id (or attack_id)")
    logger.info("  - control_id (or defense_id)")
    logger.info("")
    logger.info("Once you have uploaded the file to the above location, re-run this script.")
    logger.info("")
    logger.info("The learned mapping will be automatically generated and saved to:")
    logger.info(f"  {repo_root / 'data' / 'mappings' / 'learned_mapping.csv'}")
    logger.info("")
    logger.info("=" * 80)
    
    raise FileNotFoundError(
        f"Deterministic mapping not found at {default_path}. "
        f"Please upload your deterministic ATT&CK–D3FEND mapping CSV file to this location."
    )


def discover_splits(repo_root: Path, config_path: Path) -> dict:
    """
    Discover all available evaluation splits.
    
    Checks the config file and also searches for risk_scores*.csv files.
    
    Args:
        repo_root: Repository root directory
        config_path: Path to splits configuration YAML
        
    Returns:
        Dictionary mapping split names to relative paths
    """
    splits = {}
    
    # Load from config file
    if config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        splits = config.get("splits", {})
        logger.info(f"Loaded {len(splits)} splits from config file")
    
    # Also search for risk_scores*.csv files in common locations
    additional_splits = {}
    search_paths = [
        repo_root,
        repo_root / "results",
        repo_root / "data",
    ]
    
    for search_path in search_paths:
        if search_path.exists():
            for csv_file in search_path.rglob("risk_scores*.csv"):
                # Create a split name from the file path
                rel_path = csv_file.relative_to(repo_root)
                split_name = rel_path.stem.replace("risk_scores", "").strip("_")
                if not split_name:
                    split_name = "main"
                
                # Avoid duplicates
                if split_name not in splits and split_name not in additional_splits:
                    additional_splits[split_name] = str(rel_path)
    
    if additional_splits:
        logger.info(f"Found {len(additional_splits)} additional splits: {list(additional_splits.keys())}")
        splits.update(additional_splits)
    
    return splits


def main():
    """Main entry point for H3 Praxis experiment."""
    logger.info("=" * 80)
    logger.info("H3 Praxis Experiment: DAC Validation & Mapping Comparison")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This experiment validates the Defense–Attack Consistency (DAC) metric")
    logger.info("and compares deterministic vs learned ATT&CK–D3FEND mappings.")
    logger.info("")
    
    # Determine repository root
    repo_root = Path(__file__).parent.resolve()
    logger.info(f"Repository root: {repo_root}")
    logger.info("")
    
    # Default paths
    det_mapping_default = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    learned_mapping_default = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_pairs_default = repo_root / "d3fend_reference_pairs.csv"
    splits_config_default = repo_root / "config" / "h3_splits.yaml"
    output_dir_default = repo_root / "results" / "H3_full_evaluation"
    
    # Check and prompt for deterministic mapping
    logger.info("Step 1: Checking for deterministic mapping...")
    try:
        det_mapping_path = check_and_prompt_for_deterministic_mapping(
            det_mapping_default,
            repo_root
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Check for reference pairs
    logger.info("")
    logger.info("Step 2: Checking for reference pairs...")
    if not ref_pairs_default.exists():
        logger.warning(f"Reference pairs not found at {ref_pairs_default}")
        logger.warning("Attempting to create from deterministic mapping...")
        try:
            from scripts.create_reference_pairs import create_reference_pairs_csv
            yaml_path = repo_root / "data" / "lookups" / "attack_to_d3fend.yaml"
            if yaml_path.exists():
                create_reference_pairs_csv(yaml_path, ref_pairs_default)
                logger.info(f"✓ Created reference pairs at {ref_pairs_default}")
            else:
                logger.warning("Could not create reference pairs - will use deterministic as reference")
        except Exception as e:
            logger.warning(f"Could not create reference pairs: {e}")
            logger.warning("Will proceed using deterministic mapping as reference")
    
    ref_pairs_path = ref_pairs_default if ref_pairs_default.exists() else det_mapping_path
    
    # Discover splits
    logger.info("")
    logger.info("Step 3: Discovering evaluation splits...")
    splits_config_path = splits_config_default
    if not splits_config_path.exists():
        logger.warning(f"Splits config not found at {splits_config_path}")
        logger.warning("Creating default config with discovered splits...")
        
        # Discover splits
        splits = discover_splits(repo_root, splits_config_path)
        
        if not splits:
            # Fallback: use risk_scores.csv in root if it exists
            if (repo_root / "risk_scores.csv").exists():
                splits = {"main": "risk_scores.csv"}
                logger.info("Using default: risk_scores.csv in root")
            else:
                logger.error("No evaluation splits found!")
                logger.error("Please create config/h3_splits.yaml or ensure risk_scores.csv exists")
                sys.exit(1)
        
        # Create config file
        import yaml
        splits_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(splits_config_path, "w") as f:
            yaml.dump({"splits": splits}, f, default_flow_style=False)
        logger.info(f"✓ Created splits config at {splits_config_path}")
    else:
        splits = discover_splits(repo_root, splits_config_path)
    
    logger.info(f"Found {len(splits)} evaluation splits: {list(splits.keys())}")
    
    # Set output directory
    output_dir = output_dir_default
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Configuration Summary")
    logger.info("=" * 80)
    logger.info(f"  Deterministic mapping: {det_mapping_path}")
    logger.info(f"  Learned mapping: {learned_mapping_default} (will be auto-generated if missing)")
    logger.info(f"  Reference pairs: {ref_pairs_path}")
    logger.info(f"  Splits config: {splits_config_path}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Number of splits: {len(splits)}")
    logger.info("=" * 80)
    logger.info("")
    
    # Run evaluation
    logger.info("Starting H3 evaluation...")
    logger.info("")
    
    try:
        results = run_h3_evaluation(
            splits_config_path=splits_config_path,
            det_mapping_path=det_mapping_path,
            learned_mapping_path=learned_mapping_default,
            ref_pairs_path=ref_pairs_path,
            output_dir=output_dir,
            repo_root=repo_root,
            auto_generate_learned=True,
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("H3 Evaluation Completed Successfully!")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"  - H3_full_results.json (complete metrics and statistical tests)")
        logger.info(f"  - H3_full_summary.md (human-readable report)")
        logger.info(f"  - plots/ (visualization files)")
        logger.info("")
        logger.info(f"Evaluated {len(results['splits_evaluated'])} splits")
        logger.info("")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("H3 Evaluation Failed")
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
