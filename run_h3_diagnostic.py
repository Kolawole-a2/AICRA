#!/usr/bin/env python3
"""Diagnostic script to run H3 evaluation with full error reporting."""

import sys
import traceback
from pathlib import Path
import logging

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('h3_diagnostic.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

try:
    logger.info("=" * 80)
    logger.info("H3 Evaluation Diagnostic Run")
    logger.info("=" * 80)
    
    repo_root = Path(__file__).parent.resolve()
    logger.info(f"Repository root: {repo_root}")
    
    # Check all required files
    logger.info("\nChecking required files:")
    config_path = repo_root / "config" / "h3_splits.yaml"
    det_path = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    lrn_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_path = repo_root / "d3fend_reference_pairs.csv"
    risk_scores = repo_root / "risk_scores.csv"
    output_dir = repo_root / "results" / "H3_full_evaluation"
    
    files_to_check = {
        "Config": config_path,
        "Deterministic mapping": det_path,
        "Learned mapping": lrn_path,
        "Reference pairs": ref_path,
        "Risk scores": risk_scores,
    }
    
    all_exist = True
    for name, path in files_to_check.items():
        exists = path.exists()
        logger.info(f"  {name}: {'✓' if exists else '✗'} - {path}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        logger.error("Missing required files!")
        sys.exit(1)
    
    logger.info("\nImporting modules...")
    from aicra.experiments.h3_evaluation import run_h3_evaluation
    import yaml
    logger.info("✓ Imports successful")
    
    logger.info(f"\nOutput directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Created/verified: {output_dir.exists()}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Running H3 evaluation...")
    logger.info("=" * 80)
    
    try:
        results = run_h3_evaluation(
            splits_config_path=config_path,
            det_mapping_path=det_path,
            learned_mapping_path=lrn_path,
            ref_pairs_path=ref_path,
            output_dir=output_dir,
            repo_root=repo_root,
        )
        logger.info("\n✓ Evaluation function returned successfully")
        logger.info(f"  Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
    except Exception as e:
        logger.error(f"\n✗ Evaluation failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Check outputs
    logger.info("\n" + "=" * 80)
    logger.info("Checking outputs...")
    logger.info("=" * 80)
    
    json_file = output_dir / "H3_full_results.json"
    md_file = output_dir / "H3_full_summary.md"
    plots_dir = output_dir / "plots"
    
    logger.info(f"\nH3_full_results.json: {'✓ EXISTS' if json_file.exists() else '✗ MISSING'} - {json_file}")
    logger.info(f"H3_full_summary.md: {'✓ EXISTS' if md_file.exists() else '✗ MISSING'} - {md_file}")
    logger.info(f"plots/ directory: {'✓ EXISTS' if plots_dir.exists() else '✗ MISSING'} - {plots_dir}")
    
    if plots_dir.exists():
        plots = sorted(plots_dir.glob("*.png"))
        logger.info(f"\nPlot files found: {len(plots)}")
        for p in plots:
            logger.info(f"  - {p.name}")
    
    logger.info("\n" + "=" * 80)
    if json_file.exists() and md_file.exists() and plots_dir.exists():
        logger.info("SUCCESS! All outputs created.")
    else:
        logger.warning("WARNING: Some outputs are missing.")
    logger.info("=" * 80)
    logger.info(f"\nDiagnostic log saved to: h3_diagnostic.log")
    
except Exception as e:
    logger.error("=" * 80)
    logger.error("FATAL ERROR")
    logger.error("=" * 80)
    logger.error(f"Error: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)
