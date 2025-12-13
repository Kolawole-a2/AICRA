#!/usr/bin/env python3
"""
Diagnose why H3 is producing identical results for deterministic and learned mappings.

This script helps identify the root cause when H3 results show identical metrics.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def diagnose_identical_results(
    det_path: Path,
    learned_path: Path,
    ref_path: Path,
    risk_scores_path: Path = None,
) -> None:
    """Diagnose why H3 results are identical."""
    logger.info("=" * 80)
    logger.info("H3 Identical Results Diagnostic")
    logger.info("=" * 80)

    # Load mappings
    logger.info("\n1. Loading mappings...")
    det_df = pd.read_csv(det_path)
    learned_df = pd.read_csv(learned_path)
    ref_df = pd.read_csv(ref_path)

    # Normalize column names
    det_tech_col = "attack_id" if "attack_id" in det_df.columns else "technique_id"
    det_ctrl_col = "defense_id" if "defense_id" in det_df.columns else "control_id"

    det_pairs = set(
        zip(det_df[det_tech_col].dropna(), det_df[det_ctrl_col].dropna(), strict=False)
    )
    learned_pairs = set(
        zip(
            learned_df["technique_id"].dropna(),
            learned_df["control_id"].dropna(),
            strict=False,
        )
    )
    ref_pairs = set(
        zip(
            ref_df["technique_id"].dropna(), ref_df["control_id"].dropna(), strict=False
        )
    )

    det_techniques = set(det_df[det_tech_col].dropna().unique())
    learned_techniques = set(learned_df["technique_id"].dropna().unique())

    logger.info(
        f"   Deterministic: {len(det_pairs)} pairs, {len(det_techniques)} techniques"
    )
    logger.info(
        f"   Learned: {len(learned_pairs)} pairs, {len(learned_techniques)} techniques"
    )
    logger.info(f"   Reference: {len(ref_pairs)} pairs")

    # Check if mappings are identical
    logger.info("\n2. Checking if mappings are identical...")
    intersection = det_pairs & learned_pairs
    only_in_det = det_pairs - learned_pairs
    only_in_learned = learned_pairs - det_pairs

    if det_pairs == learned_pairs:
        logger.error("   ✗ PROBLEM: Mappings are IDENTICAL!")
        logger.error("   This is why H3 produces identical results.")
        logger.error("   Solution: Run 'python scripts/fix_learned_mapping_for_h3.py'")
        return

    logger.info("   ✓ Mappings are different")
    logger.info(f"   Intersection: {len(intersection)} pairs")
    logger.info(f"   Only in deterministic: {len(only_in_det)} pairs")
    logger.info(f"   Only in learned: {len(only_in_learned)} pairs")

    # Check technique coverage
    logger.info("\n3. Checking technique coverage...")
    missing_in_learned = det_techniques - learned_techniques
    extra_in_learned = learned_techniques - det_techniques

    if missing_in_learned:
        logger.warning(
            f"   ⚠ Learned mapping missing {len(missing_in_learned)} techniques from deterministic:"
        )
        logger.warning(f"      {sorted(missing_in_learned)}")
        logger.warning(
            "   This may cause issues if risk scores contain these techniques."
        )

    if extra_in_learned:
        logger.info(
            f"   ✓ Learned mapping has {len(extra_in_learned)} extra techniques (OK)"
        )

    # Check overlap on techniques present in both
    logger.info("\n4. Checking overlap on common techniques...")
    common_techniques = det_techniques & learned_techniques
    logger.info(f"   Common techniques: {len(common_techniques)}")

    if common_techniques:
        # For each common technique, check if mappings are identical
        identical_techniques = []
        different_techniques = []

        for tech in common_techniques:
            det_controls = set(
                det_df[det_df[det_tech_col] == tech][det_ctrl_col].dropna()
            )
            learned_controls = set(
                learned_df[learned_df["technique_id"] == tech]["control_id"].dropna()
            )

            if det_controls == learned_controls:
                identical_techniques.append(tech)
            else:
                different_techniques.append(tech)

        logger.info(
            f"   Techniques with IDENTICAL controls: {len(identical_techniques)}"
        )
        if identical_techniques:
            logger.warning(f"      {sorted(identical_techniques)[:10]}")
            logger.warning(
                "   ⚠ If risk scores only contain these techniques, H3 will show identical results!"
            )

        logger.info(
            f"   Techniques with DIFFERENT controls: {len(different_techniques)}"
        )
        if different_techniques:
            logger.info(f"      {sorted(different_techniques)[:10]}")

    # Check risk scores if provided
    if risk_scores_path and risk_scores_path.exists():
        logger.info("\n5. Checking risk scores...")
        risk_df = pd.read_csv(risk_scores_path)
        risk_techniques = set(risk_df["technique_id"].dropna().unique())
        logger.info(f"   Risk scores contain {len(risk_techniques)} techniques")

        # Check which techniques are in risk scores
        in_both_mappings = risk_techniques & det_techniques & learned_techniques
        only_in_det_mapping = risk_techniques & det_techniques - learned_techniques
        only_in_learned_mapping = risk_techniques & learned_techniques - det_techniques

        logger.info(f"   Techniques in BOTH mappings: {len(in_both_mappings)}")
        logger.info(f"   Techniques only in deterministic: {len(only_in_det_mapping)}")
        logger.info(f"   Techniques only in learned: {len(only_in_learned_mapping)}")

        if in_both_mappings:
            # Check if these techniques have identical mappings
            identical_in_risk = []
            for tech in in_both_mappings:
                det_controls = set(
                    det_df[det_df[det_tech_col] == tech][det_ctrl_col].dropna()
                )
                learned_controls = set(
                    learned_df[learned_df["technique_id"] == tech][
                        "control_id"
                    ].dropna()
                )
                if det_controls == learned_controls:
                    identical_in_risk.append(tech)

            if len(identical_in_risk) == len(in_both_mappings):
                logger.error(
                    "   ✗ PROBLEM: All techniques in risk scores have IDENTICAL mappings!"
                )
                logger.error("   This is why H3 produces identical results.")
                logger.error(
                    "   Solution: Regenerate learned mapping with different top_k or parameters"
                )
            elif identical_in_risk:
                logger.warning(
                    f"   ⚠ {len(identical_in_risk)}/{len(in_both_mappings)} techniques have identical mappings"
                )
                logger.warning("   This may cause similar (but not identical) results")

    # Summary and recommendations
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSIS SUMMARY")
    logger.info("=" * 80)

    if det_pairs == learned_pairs:
        logger.error("ROOT CAUSE: Mappings are completely identical")
        logger.error("SOLUTION: Run 'python scripts/fix_learned_mapping_for_h3.py'")
    elif (
        len(identical_techniques) == len(common_techniques)
        if common_techniques
        else False
    ):
        logger.error(
            "ROOT CAUSE: All common techniques have identical control mappings"
        )
        logger.error(
            "SOLUTION: Regenerate learned mapping with top_k=4 or top_k=5 to increase diversity"
        )
    elif (
        risk_scores_path
        and risk_scores_path.exists()
        and len(identical_in_risk) == len(in_both_mappings)
    ):
        logger.error(
            "ROOT CAUSE: All techniques in risk scores have identical mappings"
        )
        logger.error(
            "SOLUTION: Regenerate learned mapping or use risk scores with different techniques"
        )
    else:
        logger.info("Mappings appear different. If H3 still shows identical results:")
        logger.info(
            "1. Check that risk scores contain techniques with different mappings"
        )
        logger.info("2. Verify H3 is loading the correct mapping files")
        logger.info(
            "3. Check H3 evaluation code for any filtering/normalization issues"
        )

    logger.info("=" * 80)


def main():
    """Main entry point."""
    repo_root = Path(__file__).parent.parent

    det_path = repo_root / "data" / "mappings" / "deterministic_lookup.csv"
    learned_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
    ref_path = repo_root / "d3fend_reference_pairs.csv"

    # Optional: check a risk scores file if provided
    risk_scores_path = None
    if len(sys.argv) > 1:
        risk_scores_path = Path(sys.argv[1])

    # Try to find a risk scores file
    if risk_scores_path is None:
        common_paths = [
            repo_root / "results" / "time_test" / "risk_scores.csv",
            repo_root / "results" / "oof_test" / "risk_scores.csv",
        ]
        for path in common_paths:
            if path.exists():
                risk_scores_path = path
                break

    try:
        diagnose_identical_results(
            det_path=det_path,
            learned_path=learned_path,
            ref_path=ref_path,
            risk_scores_path=risk_scores_path,
        )
        return 0
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
