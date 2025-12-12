#!/usr/bin/env python3
"""
Comprehensive H3 Split Audit and Fix Script

This script:
1. Validates all technique IDs in all splits
2. Audits risk scores for alignment and consistency
3. Regenerates risk scores if needed
4. Generates diagnostics report
5. Re-runs H3 evaluation with validated data
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
from typing import Dict, List, Tuple, Set
from datetime import datetime

from aicra.utils.technique_validator import (
    validate_risk_scores_file,
    extract_valid_techniques_from_mapping,
    normalize_technique_id,
)
from aicra.experiments.h3_evaluation import (
    load_mapping_csv,
    compute_file_hash,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def audit_risk_scores(
    file_path: Path,
    valid_techniques: Set[str],
    split_name: str
) -> Dict:
    """
    Audit a risk scores file for data quality issues.
    
    Returns:
        Dictionary with audit results
    """
    logger.info(f"Auditing {split_name}: {file_path}")
    
    audit = {
        "split_name": split_name,
        "file_path": str(file_path),
        "file_exists": file_path.exists(),
        "total_rows": 0,
        "valid_technique_rows": 0,
        "invalid_technique_rows": 0,
        "unique_valid_techniques": 0,
        "risk_score_stats": {},
        "risk_score_issues": [],
        "technique_coverage": {},
        "needs_regeneration": False,
        "regeneration_reason": None,
    }
    
    if not file_path.exists():
        audit["risk_score_issues"].append("File not found")
        return audit
    
    # Load and validate
    try:
        df, diagnostics = validate_risk_scores_file(
            str(file_path),
            valid_techniques=valid_techniques,
            drop_invalid=False
        )
        
        audit.update(diagnostics)
        
        # Check risk score quality
        if "risk_score" in df.columns:
            risk_scores = df["risk_score"].values
            audit["risk_score_stats"] = {
                "mean": float(np.mean(risk_scores)),
                "std": float(np.std(risk_scores)),
                "min": float(np.min(risk_scores)),
                "max": float(np.max(risk_scores)),
                "unique_values": int(len(np.unique(risk_scores))),
                "all_same": len(np.unique(risk_scores)) == 1,
                "all_zero": np.all(risk_scores == 0),
                "all_one": np.all(risk_scores == 1),
            }
            
            # Check for degenerate risk scores
            if audit["risk_score_stats"]["all_same"]:
                audit["risk_score_issues"].append("All risk scores are identical")
                audit["needs_regeneration"] = True
                audit["regeneration_reason"] = "All risk scores are identical - likely data corruption"
            
            if audit["risk_score_stats"]["all_zero"]:
                audit["risk_score_issues"].append("All risk scores are zero")
                audit["needs_regeneration"] = True
                audit["regeneration_reason"] = "All risk scores are zero - likely data corruption"
            
            if audit["risk_score_stats"]["all_one"]:
                audit["risk_score_issues"].append("All risk scores are one")
                audit["needs_regeneration"] = True
                audit["regeneration_reason"] = "All risk scores are one - likely data corruption"
            
            # Check if risk scores are in valid range [0, 1]
            if np.any(risk_scores < 0) or np.any(risk_scores > 1):
                audit["risk_score_issues"].append("Risk scores outside [0, 1] range")
                audit["needs_regeneration"] = True
                audit["regeneration_reason"] = "Risk scores outside valid range [0, 1]"
        
        # Check technique coverage
        if "technique_id" in df.columns:
            valid_tech_mask = df["technique_id"].notna() & (df["technique_id"] != "")
            valid_techniques_in_split = set(df[valid_tech_mask]["technique_id"].unique())
            
            audit["technique_coverage"] = {
                "techniques_in_split": len(valid_techniques_in_split),
                "techniques_in_deterministic": len(valid_techniques_in_split & valid_techniques),
                "techniques_not_in_mappings": len(valid_techniques_in_split - valid_techniques),
            }
            
            if audit["technique_coverage"]["techniques_in_split"] == 0:
                audit["risk_score_issues"].append("No valid techniques found")
                audit["needs_regeneration"] = True
                audit["regeneration_reason"] = "No valid techniques - cannot compute metrics"
        
        # Check label alignment
        if "true_label" in df.columns and "predicted_label" in df.columns:
            true_labels = df["true_label"].values
            pred_labels = df["predicted_label"].values
            
            # Check if predicted labels match risk scores
            if "risk_score" in df.columns:
                risk_scores = df["risk_score"].values
                expected_pred = (risk_scores >= 0.5).astype(int)
                mismatch = np.sum(pred_labels != expected_pred)
                
                if mismatch > 0:
                    mismatch_pct = (mismatch / len(df)) * 100
                    audit["risk_score_issues"].append(
                        f"Predicted labels don't match risk scores: {mismatch}/{len(df)} ({mismatch_pct:.1f}%)"
                    )
                    if mismatch_pct > 10:
                        audit["needs_regeneration"] = True
                        audit["regeneration_reason"] = f"High mismatch ({mismatch_pct:.1f}%) between predicted labels and risk scores"
        
    except Exception as e:
        logger.error(f"Error auditing {split_name}: {e}", exc_info=True)
        audit["risk_score_issues"].append(f"Error during audit: {str(e)}")
        audit["needs_regeneration"] = True
        audit["regeneration_reason"] = f"Error during audit: {str(e)}"
    
    return audit


def regenerate_risk_scores_from_register(
    register_path: Path,
    output_path: Path,
    split_name: str
) -> bool:
    """
    Regenerate risk scores from register file.
    
    This function attempts to regenerate risk_scores.csv from the source register file.
    If the register file doesn't exist or doesn't have the right structure, returns False.
    """
    logger.info(f"Attempting to regenerate risk scores for {split_name} from register...")
    
    # Map split names to register files
    register_map = {
        "main": "register/risk_register_main.csv",
        "small_ember": "register/risk_register_small_ember.csv",
        "full_ember": "register/risk_register_full.csv",
        "smoke_test": "register/smoke_test_register.csv",
    }
    
    if split_name not in register_map:
        logger.warning(f"No register file mapping for split '{split_name}'")
        return False
    
    register_file = Path(register_map[split_name])
    if not register_file.exists():
        logger.warning(f"Register file not found: {register_file}")
        return False
    
    try:
        # Use create_ember_splits.py logic
        import subprocess
        import sys
        
        logger.info(f"Running create_ember_splits.py to regenerate {split_name}...")
        result = subprocess.run(
            [sys.executable, "create_ember_splits.py"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Successfully regenerated {split_name}")
            return True
        else:
            logger.error(f"Failed to regenerate {split_name}: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error regenerating {split_name}: {e}", exc_info=True)
        return False


def generate_diagnostics_report(
    audits: Dict[str, Dict],
    valid_techniques: Set[str],
    output_dir: Path
) -> Path:
    """
    Generate comprehensive diagnostics report.
    """
    report_path = output_dir / "H3_diagnostics.md"
    
    with open(report_path, "w") as f:
        f.write("# H3 Evaluation Diagnostics Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write("## Overview\n\n")
        f.write("This report provides detailed diagnostics for each evaluation split, ")
        f.write("including technique ID validation, risk score quality checks, and ")
        f.write("recommendations for data fixes.\n\n")
        
        f.write("## Technique Validation\n\n")
        f.write(f"Total valid techniques across all mappings: {len(valid_techniques)}\n\n")
        f.write("Valid techniques are those that:\n")
        f.write("- Match the pattern T#### or T####.###\n")
        f.write("- Are present in at least one mapping (deterministic, learned, or reference)\n\n")
        
        f.write("## Per-Split Diagnostics\n\n")
        
        for split_name, audit in audits.items():
            f.write(f"### {split_name}\n\n")
            f.write(f"- **File Path:** `{audit['file_path']}`\n")
            f.write(f"- **File Exists:** {'✓' if audit['file_exists'] else '✗'}\n")
            
            if not audit['file_exists']:
                f.write(f"- **Status:** File not found - split will be skipped\n\n")
                continue
            
            f.write(f"- **Total Rows:** {audit['total_rows']}\n")
            f.write(f"- **Valid Technique Rows:** {audit['valid_technique_rows']} ({audit['valid_technique_rows']/audit['total_rows']*100:.1f}%)\n")
            f.write(f"- **Invalid Technique Rows:** {audit['invalid_technique_rows']}\n")
            f.write(f"- **Unique Valid Techniques:** {audit['unique_valid_techniques']}\n")
            
            if audit['invalid_technique_rows'] > 0:
                f.write(f"- **Invalid Technique IDs:** {', '.join(audit.get('invalid_ids', [])[:10])}\n")
                if len(audit.get('invalid_ids', [])) > 10:
                    f.write(f"  (showing first 10 of {len(audit.get('invalid_ids', []))})\n")
            
            # Risk score stats
            if audit.get('risk_score_stats'):
                stats = audit['risk_score_stats']
                f.write(f"\n**Risk Score Statistics:**\n")
                f.write(f"- Mean: {stats['mean']:.6f}\n")
                f.write(f"- Std: {stats['std']:.6f}\n")
                f.write(f"- Range: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                f.write(f"- Unique values: {stats['unique_values']}\n")
                
                if stats['all_same']:
                    f.write(f"- ⚠️ **WARNING:** All risk scores are identical\n")
                if stats['all_zero']:
                    f.write(f"- ⚠️ **WARNING:** All risk scores are zero\n")
                if stats['all_one']:
                    f.write(f"- ⚠️ **WARNING:** All risk scores are one\n")
            
            # Technique coverage
            if audit.get('technique_coverage'):
                coverage = audit['technique_coverage']
                f.write(f"\n**Technique Coverage:**\n")
                f.write(f"- Techniques in split: {coverage['techniques_in_split']}\n")
                f.write(f"- Techniques in mappings: {coverage['techniques_in_deterministic']}\n")
                f.write(f"- Techniques not in mappings: {coverage['techniques_not_in_mappings']}\n")
            
            # Issues
            if audit.get('risk_score_issues'):
                f.write(f"\n**Issues Found:**\n")
                for issue in audit['risk_score_issues']:
                    f.write(f"- ⚠️ {issue}\n")
            
            # Regeneration status
            if audit.get('needs_regeneration'):
                f.write(f"\n**⚠️ REGENERATION RECOMMENDED**\n")
                f.write(f"- Reason: {audit.get('regeneration_reason', 'Unknown')}\n")
            else:
                f.write(f"\n**✓ No regeneration needed**\n")
            
            f.write("\n")
        
        f.write("## Summary\n\n")
        
        total_splits = len(audits)
        splits_with_issues = sum(1 for a in audits.values() if a.get('risk_score_issues'))
        splits_needing_regeneration = sum(1 for a in audits.values() if a.get('needs_regeneration'))
        
        f.write(f"- **Total Splits Audited:** {total_splits}\n")
        f.write(f"- **Splits with Issues:** {splits_with_issues}\n")
        f.write(f"- **Splits Needing Regeneration:** {splits_needing_regeneration}\n\n")
        
        if splits_needing_regeneration > 0:
            f.write("### Recommended Actions\n\n")
            f.write("1. Review the issues listed above for each split\n")
            f.write("2. Regenerate risk scores using `create_ember_splits.py`\n")
            f.write("3. Re-run H3 evaluation after fixes\n\n")
    
    logger.info(f"Diagnostics report saved to {report_path}")
    return report_path


def main():
    """Main audit and fix workflow."""
    logger.info("=" * 80)
    logger.info("H3 SPLIT AUDIT AND FIX")
    logger.info("=" * 80)
    
    repo_root = Path.cwd()
    config_path = repo_root / "config" / "h3_splits.yaml"
    output_dir = repo_root / "results" / "H3_full_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    splits = config.get("splits", {})
    
    logger.info(f"Found {len(splits)} splits in config")
    
    # Load mappings to extract valid techniques
    det_mapping_path = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    learned_mapping_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_pairs_path = repo_root / "d3fend_reference_pairs.csv"
    
    logger.info("Loading mappings to extract valid techniques...")
    det_mapping = load_mapping_csv(det_mapping_path)
    learned_mapping = load_mapping_csv(learned_mapping_path)
    ref_pairs = load_mapping_csv(ref_pairs_path)
    
    # Filter deterministic by is_correct if present
    if "is_correct" in det_mapping.columns:
        det_mapping = det_mapping[det_mapping["is_correct"] == 1].copy()
    
    # Normalize column names
    for df in [det_mapping, learned_mapping, ref_pairs]:
        if "attack_id" in df.columns and "technique_id" not in df.columns:
            df.rename(columns={"attack_id": "technique_id"}, inplace=True)
        if "defense_id" in df.columns and "control_id" not in df.columns:
            df.rename(columns={"defense_id": "control_id"}, inplace=True)
    
    # Extract valid techniques
    det_valid = extract_valid_techniques_from_mapping(det_mapping, "technique_id")
    learned_valid = extract_valid_techniques_from_mapping(learned_mapping, "technique_id")
    ref_valid = extract_valid_techniques_from_mapping(ref_pairs, "technique_id")
    all_valid_techniques = det_valid | learned_valid | ref_valid
    
    logger.info(f"Valid techniques: {len(all_valid_techniques)}")
    
    # Audit all splits
    logger.info("\n" + "=" * 80)
    logger.info("AUDITING ALL SPLITS")
    logger.info("=" * 80)
    
    audits = {}
    for split_name, rel_path in splits.items():
        split_path = repo_root / rel_path
        audit = audit_risk_scores(split_path, all_valid_techniques, split_name)
        audits[split_name] = audit
        
        logger.info(f"\n{split_name}:")
        logger.info(f"  Total rows: {audit['total_rows']}")
        logger.info(f"  Valid technique rows: {audit['valid_technique_rows']}")
        logger.info(f"  Unique valid techniques: {audit['unique_valid_techniques']}")
        if audit.get('risk_score_issues'):
            logger.warning(f"  Issues: {len(audit['risk_score_issues'])}")
            for issue in audit['risk_score_issues']:
                logger.warning(f"    - {issue}")
        if audit.get('needs_regeneration'):
            logger.warning(f"  ⚠️  Needs regeneration: {audit.get('regeneration_reason')}")
    
    # Generate diagnostics report
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING DIAGNOSTICS REPORT")
    logger.info("=" * 80)
    
    diagnostics_path = generate_diagnostics_report(audits, all_valid_techniques, output_dir)
    
    # Save audit results as JSON
    audit_json_path = output_dir / "H3_audit_results.json"
    with open(audit_json_path, "w") as f:
        json.dump(audits, f, indent=2, default=str)
    logger.info(f"Audit results saved to {audit_json_path}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("AUDIT SUMMARY")
    logger.info("=" * 80)
    
    total = len(audits)
    with_issues = sum(1 for a in audits.values() if a.get('risk_score_issues'))
    needing_regeneration = sum(1 for a in audits.values() if a.get('needs_regeneration'))
    
    logger.info(f"Total splits: {total}")
    logger.info(f"Splits with issues: {with_issues}")
    logger.info(f"Splits needing regeneration: {needing_regeneration}")
    
    if needing_regeneration > 0:
        logger.warning("\n⚠️  Some splits need regeneration. Review diagnostics report.")
        logger.warning(f"   Diagnostics: {diagnostics_path}")
    else:
        logger.info("\n✓ All splits passed audit - ready for H3 evaluation")
    
    logger.info("=" * 80)
    
    return audits


if __name__ == "__main__":
    main()
