#!/usr/bin/env python3
"""
Run H3 evaluation with comprehensive validation and diagnostics.

This script:
1. Audits all splits for technique ID and risk score quality
2. Regenerates risk scores if needed
3. Runs H3 evaluation with validated data
4. Generates diagnostics report
"""

import logging
import sys
from pathlib import Path

from audit_and_fix_h3_splits import main as audit_main
from aicra.experiments.h3_evaluation import run_h3_evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main workflow: audit, fix if needed, then evaluate."""
    logger.info("=" * 80)
    logger.info("H3 EVALUATION WITH VALIDATION")
    logger.info("=" * 80)
    
    repo_root = Path.cwd()
    config_path = repo_root / "config" / "h3_splits.yaml"
    output_dir = repo_root / "results" / "H3_full_evaluation"
    
    # Step 1: Audit all splits
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: AUDITING ALL SPLITS")
    logger.info("=" * 80)
    
    audits = audit_main()
    
    # Check if any splits need regeneration
    needing_regeneration = [name for name, audit in audits.items() if audit.get('needs_regeneration')]
    
    if needing_regeneration:
        logger.warning(f"\n⚠️  {len(needing_regeneration)} splits need regeneration:")
        for name in needing_regeneration:
            logger.warning(f"  - {name}: {audits[name].get('regeneration_reason', 'Unknown')}")
        
        logger.info("\nAttempting to regenerate splits...")
        from create_ember_splits import main as regenerate_splits
        try:
            regenerate_splits()
            logger.info("✓ Regenerated splits")
        except Exception as e:
            logger.error(f"Failed to regenerate splits: {e}")
            logger.error("Please manually fix the splits before running H3 evaluation")
            sys.exit(1)
    
    # Step 2: Run H3 evaluation with validated data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: RUNNING H3 EVALUATION WITH VALIDATED DATA")
    logger.info("=" * 80)
    
    det_mapping_path = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    learned_mapping_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_pairs_path = repo_root / "d3fend_reference_pairs.csv"
    
    try:
        results = run_h3_evaluation(
            splits_config_path=config_path,
            det_mapping_path=det_mapping_path,
            learned_mapping_path=learned_mapping_path,
            ref_pairs_path=ref_pairs_path,
            output_dir=output_dir,
            repo_root=repo_root
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("H3 EVALUATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"  - H3_full_results.json")
        logger.info(f"  - H3_full_summary.md")
        logger.info(f"  - H3_diagnostics.md")
        logger.info(f"  - plots/")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error running H3 evaluation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
