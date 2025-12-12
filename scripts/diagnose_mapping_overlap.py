#!/usr/bin/env python3
"""
Diagnose overlap between deterministic and learned ATT&CK–D3FEND mappings.

This script:
1. Loads deterministic and learned mappings
2. Normalizes them to (technique_id, set(control_id))
3. Computes global Jaccard similarity
4. Computes per-technique overlap (EXACT_MATCH, PARTIAL_OVERLAP, DISJOINT)
5. Restricts analysis to techniques present in risk scores
6. Generates human-readable report and JSON summary
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_mapping(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Normalize mapping DataFrame to dict: technique_id -> set(control_id).
    
    Handles column name variations: attack_id/technique_id, defense_id/control_id.
    """
    # Normalize column names
    tech_col = None
    ctrl_col = None
    
    for col in df.columns:
        if col in ["attack_id", "technique_id"]:
            tech_col = col
        elif col in ["defense_id", "control_id"]:
            ctrl_col = col
    
    if tech_col is None or ctrl_col is None:
        raise ValueError(f"Cannot find technique and control columns in mapping. Found columns: {list(df.columns)}")
    
    # Build mapping
    mapping = defaultdict(set)
    for _, row in df.iterrows():
        tech = row[tech_col]
        ctrl = row[ctrl_col]
        if pd.notna(tech) and pd.notna(ctrl):
            mapping[str(tech)].add(str(ctrl))
    
    return dict(mapping)


def compute_jaccard(set1: Set, set2: Set) -> float:
    """Compute Jaccard similarity: |A ∩ B| / |A ∪ B|."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def classify_technique_overlap(
    det_controls: Set[str],
    learned_controls: Set[str]
) -> str:
    """Classify overlap type for a technique."""
    if det_controls == learned_controls:
        return "EXACT_MATCH"
    elif len(det_controls & learned_controls) > 0:
        return "PARTIAL_OVERLAP"
    else:
        return "DISJOINT"


def load_risk_scores_techniques(splits_config_path: Path, repo_root: Path) -> Set[str]:
    """
    Load all technique_ids from risk score files defined in splits config.
    
    Returns:
        Set of technique_id strings present in any risk score file
    """
    if not splits_config_path.exists():
        logger.warning(f"Splits config not found: {splits_config_path}")
        return set()
    
    with open(splits_config_path, "r") as f:
        config = yaml.safe_load(f)
    
    splits = config.get("splits", {})
    all_techniques = set()
    
    for split_name, risk_scores_rel_path in splits.items():
        risk_scores_path = repo_root / risk_scores_rel_path
        
        if not risk_scores_path.exists():
            logger.warning(f"Risk scores not found for split '{split_name}': {risk_scores_path}")
            continue
        
        try:
            df = pd.read_csv(risk_scores_path)
            
            # Normalize technique_id column name
            tech_col = None
            for col in df.columns:
                if col in ["attack_id", "technique_id"]:
                    tech_col = col
                    break
            
            if tech_col is None:
                logger.warning(f"No technique_id column found in {risk_scores_path}. Columns: {list(df.columns)}")
                continue
            
            techniques = set(df[tech_col].dropna().astype(str).unique())
            all_techniques.update(techniques)
            logger.info(f"Split '{split_name}': {len(techniques)} unique techniques")
            
        except Exception as e:
            logger.error(f"Error loading risk scores from {risk_scores_path}: {e}")
            continue
    
    logger.info(f"Total unique techniques across all risk scores: {len(all_techniques)}")
    return all_techniques


def diagnose_mapping_overlap(
    det_path: Path,
    learned_path: Path,
    splits_config_path: Path,
    repo_root: Path,
    output_dir: Path,
    risk_scores_path: Optional[Path] = None,
) -> Dict:
    """
    Diagnose overlap between deterministic and learned mappings.
    
    Returns:
        Dictionary with diagnostic results
    """
    logger.info("=" * 80)
    logger.info("Mapping Overlap Diagnostic")
    logger.info("=" * 80)
    
    # Load mappings
    logger.info("\n1. Loading mappings...")
    det_df = pd.read_csv(det_path)
    learned_df = pd.read_csv(learned_path)
    
    det_mapping = normalize_mapping(det_df)
    learned_mapping = normalize_mapping(learned_df)
    
    logger.info(f"   Deterministic: {len(det_mapping)} techniques, {sum(len(v) for v in det_mapping.values())} pairs")
    logger.info(f"   Learned: {len(learned_mapping)} techniques, {sum(len(v) for v in learned_mapping.values())} pairs")
    
    # Compute global Jaccard on pair sets
    logger.info("\n2. Computing global pair-set Jaccard similarity...")
    det_pairs = set()
    for tech, controls in det_mapping.items():
        for ctrl in controls:
            det_pairs.add((tech, ctrl))
    
    learned_pairs = set()
    for tech, controls in learned_mapping.items():
        for ctrl in controls:
            learned_pairs.add((tech, ctrl))
    
    global_jaccard = compute_jaccard(det_pairs, learned_pairs)
    logger.info(f"   Global Jaccard: {global_jaccard:.4f} ({global_jaccard*100:.2f}%)")
    
    # Load techniques from risk scores
    logger.info("\n3. Loading techniques from risk scores...")
    if risk_scores_path and risk_scores_path.exists():
        # Use provided risk scores path if given
        risk_df = pd.read_csv(risk_scores_path)
        tech_col = None
        for col in risk_df.columns:
            if col in ["attack_id", "technique_id"]:
                tech_col = col
                break
        if tech_col:
            risk_score_techniques = set(risk_df[tech_col].dropna().astype(str).unique())
            logger.info(f"   Loaded {len(risk_score_techniques)} techniques from provided risk scores")
        else:
            risk_score_techniques = load_risk_scores_techniques(splits_config_path, repo_root)
    else:
        risk_score_techniques = load_risk_scores_techniques(splits_config_path, repo_root)
    
    # Per-technique analysis (restricted to risk score techniques if available)
    logger.info("\n4. Analyzing per-technique overlap...")
    
    if risk_score_techniques:
        logger.info(f"   Restricting to {len(risk_score_techniques)} techniques present in risk scores")
        analysis_techniques = risk_score_techniques
    else:
        logger.info("   No risk scores found, analyzing all techniques in mappings")
        analysis_techniques = set(det_mapping.keys()) | set(learned_mapping.keys())
    
    technique_classifications = {}
    exact_matches = []
    partial_overlaps = []
    disjoints = []
    
    for tech in sorted(analysis_techniques):
        det_controls = det_mapping.get(tech, set())
        learned_controls = learned_mapping.get(tech, set())
        
        if not det_controls and not learned_controls:
            continue  # Skip techniques not in either mapping
        
        overlap_type = classify_technique_overlap(det_controls, learned_controls)
        technique_classifications[tech] = {
            "overlap_type": overlap_type,
            "det_controls": sorted(list(det_controls)),
            "learned_controls": sorted(list(learned_controls)),
            "intersection": sorted(list(det_controls & learned_controls)),
            "jaccard": compute_jaccard(det_controls, learned_controls),
        }
        
        if overlap_type == "EXACT_MATCH":
            exact_matches.append(tech)
        elif overlap_type == "PARTIAL_OVERLAP":
            partial_overlaps.append(tech)
        else:
            disjoints.append(tech)
    
    # Generate report
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC REPORT")
    logger.info("=" * 80)
    
    logger.info(f"\nGlobal Jaccard between Det and Learned: {global_jaccard:.4f} ({global_jaccard*100:.2f}%)")
    logger.info(f"\nTechniques analyzed: {len(technique_classifications)}")
    if risk_score_techniques:
        logger.info(f"  (Restricted to techniques in risk scores)")
    logger.info(f"\nOverlap Classification:")
    logger.info(f"  EXACT_MATCH: {len(exact_matches)}")
    logger.info(f"  PARTIAL_OVERLAP: {len(partial_overlaps)}")
    logger.info(f"  DISJOINT: {len(disjoints)}")
    
    # Show sample EXACT_MATCH techniques
    if exact_matches:
        logger.info(f"\nSample EXACT_MATCH techniques (showing first 10):")
        for tech in exact_matches[:10]:
            info = technique_classifications[tech]
            logger.info(f"  {tech}:")
            logger.info(f"    Det controls: {info['det_controls']}")
            logger.info(f"    Learned controls: {info['learned_controls']}")
            logger.info(f"    Jaccard: {info['jaccard']:.4f}")
    
    # Show sample PARTIAL_OVERLAP techniques
    if partial_overlaps:
        logger.info(f"\nSample PARTIAL_OVERLAP techniques (showing first 5):")
        for tech in partial_overlaps[:5]:
            info = technique_classifications[tech]
            logger.info(f"  {tech}:")
            logger.info(f"    Det controls: {info['det_controls']}")
            logger.info(f"    Learned controls: {info['learned_controls']}")
            logger.info(f"    Intersection: {info['intersection']}")
            logger.info(f"    Jaccard: {info['jaccard']:.4f}")
    
    # Risk score coverage analysis
    risk_score_coverage = None
    if risk_score_techniques:
        logger.info("\n5. Analyzing risk score coverage...")
        
        # Build technique-to-controls mappings
        det_tech_to_controls = {}
        for tech, ctrl in det_pairs:
            if tech not in det_tech_to_controls:
                det_tech_to_controls[tech] = set()
            det_tech_to_controls[tech].add(ctrl)
        
        learned_tech_to_controls = {}
        for tech, ctrl in learned_pairs:
            if tech not in learned_tech_to_controls:
                learned_tech_to_controls[tech] = set()
            learned_tech_to_controls[tech].add(ctrl)
        
        risk_exact_matches = []
        risk_partial_overlaps = []
        risk_disjoints = []
        
        for tech in risk_score_techniques:
            det_ctrls = det_tech_to_controls.get(tech, set())
            learned_ctrls = learned_tech_to_controls.get(tech, set())
            
            if not det_ctrls and not learned_ctrls:
                continue  # Technique not in either mapping
            
            if det_ctrls == learned_ctrls:
                risk_exact_matches.append(tech)
            elif len(det_ctrls & learned_ctrls) > 0:
                risk_partial_overlaps.append(tech)
            else:
                risk_disjoints.append(tech)
        
        risk_total = len(risk_exact_matches) + len(risk_partial_overlaps) + len(risk_disjoints)
        if risk_total > 0:
            risk_exact_match_pct = (len(risk_exact_matches) / risk_total * 100.0)
            logger.info(f"   Techniques in risk scores with mappings: {risk_total}")
            logger.info(f"   EXACT_MATCH: {len(risk_exact_matches)} ({risk_exact_match_pct:.1f}%)")
            logger.info(f"   PARTIAL_OVERLAP: {len(risk_partial_overlaps)}")
            logger.info(f"   DISJOINT: {len(risk_disjoints)}")
            
            if risk_exact_match_pct >= 100.0:
                logger.error("\n" + "=" * 80)
                logger.error("⚠️  CRITICAL: ALL techniques in risk scores have EXACT_MATCH mappings!")
                logger.error("=" * 80)
                logger.error("H3 cannot demonstrate any difference without additional or different techniques.")
                logger.error("SOLUTION:")
                logger.error("  1. Regenerate learned mapping with higher diversity")
                logger.error("  2. Or use risk scores that include techniques with different mappings")
                logger.error("=" * 80)
            elif risk_exact_match_pct > 80.0:
                logger.warning("\n" + "=" * 80)
                logger.warning(f"⚠️  WARNING: {risk_exact_match_pct:.1f}% of techniques in risk scores have EXACT_MATCH mappings!")
                logger.warning("=" * 80)
                logger.warning("H3 results may show very similar metrics.")
                logger.warning("Consider regenerating learned mapping or using risk scores with more diverse techniques.")
                logger.warning("=" * 80)
            
            risk_score_coverage = {
                "total_techniques_in_risk_scores": len(risk_score_techniques),
                "techniques_with_mappings": risk_total,
                "exact_match_count": len(risk_exact_matches),
                "exact_match_fraction": float(len(risk_exact_matches) / risk_total) if risk_total > 0 else 0.0,
                "partial_overlap_count": len(risk_partial_overlaps),
                "disjoint_count": len(risk_disjoints),
                "exact_match_techniques": sorted(risk_exact_matches),
            }
    
    # Compile results
    results = {
        "global_jaccard": float(global_jaccard),
        "global_jaccard_percent": float(global_jaccard * 100),
        "total_det_pairs": len(det_pairs),
        "total_learned_pairs": len(learned_pairs),
        "intersection_pairs": len(det_pairs & learned_pairs),
        "union_pairs": len(det_pairs | learned_pairs),
        "techniques_analyzed": len(technique_classifications),
        "techniques_in_risk_scores": len(risk_score_techniques) if risk_score_techniques else None,
        "overlap_classification": {
            "exact_match_count": len(exact_matches),
            "partial_overlap_count": len(partial_overlaps),
            "disjoint_count": len(disjoints),
            "exact_match_techniques": sorted(exact_matches),
            "partial_overlap_techniques": sorted(partial_overlaps),
            "disjoint_techniques": sorted(disjoints),
        },
        "per_technique": technique_classifications,
        "risk_score_coverage": risk_score_coverage,
    }
    
    # Save JSON summary
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mapping_overlap.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Saved diagnostic results to {json_path}")
    
    # Warning if too similar
    if global_jaccard > 0.90:
        logger.warning("\n" + "=" * 80)
        logger.warning("⚠️  WARNING: Mappings are VERY SIMILAR (Jaccard > 90%)!")
        logger.warning("=" * 80)
        logger.warning("H3 results will likely show almost identical metrics.")
        logger.warning("Consider regenerating learned mapping with increased top_k or different parameters.")
        logger.warning("=" * 80)
    
    if len(exact_matches) > len(technique_classifications) * 0.8:
        logger.warning("\n" + "=" * 80)
        logger.warning(f"⚠️  WARNING: {len(exact_matches)}/{len(technique_classifications)} techniques have EXACT_MATCH mappings!")
        logger.warning("=" * 80)
        logger.warning("Most techniques have identical control mappings.")
        logger.warning("H3 will show identical results for these techniques.")
        logger.warning("Consider regenerating learned mapping with increased diversity.")
        logger.warning("=" * 80)
    
    logger.info("\n" + "=" * 80)
    logger.info("Diagnostic Complete")
    logger.info("=" * 80)
    
    return results


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent
    
    det_path = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    learned_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
    splits_config_path = repo_root / "config" / "h3_splits.yaml"
    output_dir = repo_root / "results" / "H3_diagnostics"
    
    if not det_path.exists():
        logger.error(f"Deterministic mapping not found: {det_path}")
        sys.exit(1)
    
    if not learned_path.exists():
        logger.error(f"Learned mapping not found: {learned_path}")
        logger.error("Please generate it first:")
        logger.error("  python -m aicra.mapping.heuristic_mapping --top-k 5 --min-similarity 0.40")
        sys.exit(1)
    
    # Optional: check specific risk scores file if provided
    risk_scores_path = None
    if len(sys.argv) > 1:
        risk_scores_path = Path(sys.argv[1])
        if not risk_scores_path.exists():
            logger.warning(f"Provided risk scores path does not exist: {risk_scores_path}")
            risk_scores_path = None
    
    try:
        results = diagnose_mapping_overlap(
            det_path=det_path,
            learned_path=learned_path,
            splits_config_path=splits_config_path,
            repo_root=repo_root,
            output_dir=output_dir,
            risk_scores_path=risk_scores_path,
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Global Jaccard: {results['global_jaccard_percent']:.2f}%")
        print(f"EXACT_MATCH techniques: {results['overlap_classification']['exact_match_count']}")
        print(f"PARTIAL_OVERLAP techniques: {results['overlap_classification']['partial_overlap_count']}")
        print(f"DISJOINT techniques: {results['overlap_classification']['disjoint_count']}")
        
        if results.get('risk_score_coverage'):
            rsc = results['risk_score_coverage']
            print(f"\nRisk Score Coverage:")
            print(f"  Techniques in risk scores: {rsc['total_techniques_in_risk_scores']}")
            print(f"  EXACT_MATCH: {rsc['exact_match_count']} ({rsc['exact_match_fraction']*100:.1f}%)")
            print(f"  PARTIAL_OVERLAP: {rsc['partial_overlap_count']}")
            print(f"  DISJOINT: {rsc['disjoint_count']}")
            
            if rsc['exact_match_fraction'] >= 1.0:
                print("\n⚠️  CRITICAL: ALL techniques in risk scores have EXACT_MATCH mappings!")
                print("   H3 cannot demonstrate any difference.")
            elif rsc['exact_match_fraction'] > 0.80:
                print(f"\n⚠️  WARNING: {rsc['exact_match_fraction']*100:.1f}% of risk score techniques have EXACT_MATCH")
        
        if results['global_jaccard_percent'] > 90:
            print("\n⚠️  WARNING: Jaccard > 90% - mappings are too similar!")
            print("   Run: python -m aicra.mapping.heuristic_mapping --top-k 5")
        elif results['global_jaccard_percent'] > 80:
            print("\n⚠️  WARNING: Jaccard > 80% - mappings are very similar")
            print("   Consider: python -m aicra.mapping.heuristic_mapping --top-k 5")
        else:
            print("\n✓ Mappings have reasonable diversity")
        
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
