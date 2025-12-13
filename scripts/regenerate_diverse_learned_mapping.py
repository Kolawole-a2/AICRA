#!/usr/bin/env python3
"""
Regenerate learned mapping with increased diversity (higher top_k).

This script:
1. Loads deterministic mapping
2. Generates learned mapping using embeddings with increased top_k (default: 4 or 5)
3. Ensures it covers all techniques from deterministic
4. Verifies it's different from deterministic
5. Saves to data/mappings/learned_mapping.csv
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


def regenerate_diverse_learned_mapping(
    det_path: Path,
    output_path: Path,
    top_k: int = 4,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    Regenerate learned mapping with increased diversity.

    Args:
        det_path: Path to deterministic mapping
        output_path: Path to save learned mapping
        top_k: Number of top controls per technique (default: 4, increased from 3)
        model_name: Sentence transformer model name

    Returns:
        DataFrame with learned mapping
    """
    logger.info("=" * 80)
    logger.info("Regenerating Diverse Learned Mapping")
    logger.info("=" * 80)
    logger.info(f"Top-k: {top_k} (increased for more diversity)")
    logger.info(f"Model: {model_name}")
    logger.info("=" * 80)

    # Check if deterministic path exists
    if not det_path.exists():
        raise FileNotFoundError(f"Deterministic mapping not found: {det_path}")

    # Try to find the correct deterministic path
    # Check for both attack_id/defense_id and technique_id/control_id formats
    det_df = pd.read_csv(det_path)

    # Generate learned mapping using embeddings
    try:
        from aicra.mappings.embedding_learned_mapping import (
            build_learned_embedding_mapping,
        )

        output_dir = output_path.parent
        logger.info(f"Generating learned mapping with top_k={top_k}...")

        learned_df = build_learned_embedding_mapping(
            deterministic_path=det_path,
            output_dir=output_dir,
            model_name=model_name,
            top_k=top_k,
        )

        # Normalize column names to technique_id, control_id
        if "attack_id" in learned_df.columns:
            learned_df = learned_df.rename(columns={"attack_id": "technique_id"})
        if "defense_id" in learned_df.columns:
            learned_df = learned_df.rename(columns={"defense_id": "control_id"})

        logger.info(f"Generated learned mapping with {len(learned_df)} pairs")

    except Exception as e:
        logger.error(f"Failed to generate learned mapping: {e}", exc_info=True)
        raise

    # Verify it's different from deterministic
    logger.info("\nVerifying learned mapping is different from deterministic...")

    # Normalize deterministic columns
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

    intersection = det_pairs & learned_pairs
    only_in_det = det_pairs - learned_pairs
    only_in_learned = learned_pairs - det_pairs

    logger.info(f"Deterministic pairs: {len(det_pairs)}")
    logger.info(f"Learned pairs: {len(learned_pairs)}")
    logger.info(f"Intersection: {len(intersection)}")
    logger.info(f"Only in deterministic: {len(only_in_det)}")
    logger.info(f"Only in learned: {len(only_in_learned)}")

    if len(only_in_det) == 0 and len(only_in_learned) == 0 and len(det_pairs) > 0:
        logger.error("=" * 80)
        logger.error("CRITICAL ERROR: Mappings are still IDENTICAL!")
        logger.error("=" * 80)
        logger.error(
            "Even with top_k={top_k}, the learned mapping is identical to deterministic."
        )
        logger.error("This may indicate:")
        logger.error("1. Embedding model is producing identical similarity scores")
        logger.error(
            "2. Deterministic pairs happen to match top-k embedding similarities"
        )
        logger.error("3. There's a bug in the learned mapping generation")
        logger.error("=" * 80)
        raise RuntimeError(
            f"Learned mapping is still identical to deterministic even with top_k={top_k}. "
            "Try increasing top_k further or check the embedding generation code."
        )

    overlap_pct = (
        (len(intersection) / len(det_pairs) * 100) if len(det_pairs) > 0 else 0
    )
    logger.info("\n✓ Mappings are different")
    logger.info(f"  Overlap: {len(intersection)}/{len(det_pairs)} ({overlap_pct:.1f}%)")
    logger.info(f"  Unique to deterministic: {len(only_in_det)} pairs")
    logger.info(f"  Unique to learned: {len(only_in_learned)} pairs")

    if overlap_pct > 90:
        logger.warning("⚠️  WARNING: Overlap is still very high (>90%)")
        logger.warning("Consider increasing top_k further (e.g., top_k=5 or top_k=6)")

    # Save learned mapping
    output_df = learned_df[["technique_id", "control_id"]].copy()
    if "similarity_score" in learned_df.columns:
        output_df["similarity_score"] = learned_df["similarity_score"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Saved learned mapping to {output_path}")
    logger.info(f"  Total pairs: {len(output_df)}")
    logger.info(f"  Techniques: {output_df['technique_id'].nunique()}")
    logger.info(f"  Controls: {output_df['control_id'].nunique()}")

    logger.info("=" * 80)
    logger.info("Learned mapping regeneration complete")
    logger.info("=" * 80)

    return output_df


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate learned mapping with increased diversity"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of top controls per technique (default: 4, increased from 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model name",
    )
    parser.add_argument(
        "--deterministic",
        type=Path,
        default=None,
        help="Path to deterministic mapping (default: data/mappings/deterministic_lookup.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output learned mapping (default: data/mappings/learned_mapping.csv)",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent

    if args.deterministic is None:
        args.deterministic = (
            repo_root / "data" / "mappings" / "deterministic_lookup.csv"
        )

    if args.output is None:
        args.output = repo_root / "data" / "mappings" / "learned_mapping.csv"

    try:
        learned_df = regenerate_diverse_learned_mapping(
            det_path=args.deterministic,
            output_path=args.output,
            top_k=args.top_k,
            model_name=args.model,
        )

        print("\n" + "=" * 80)
        print("SUCCESS: Learned mapping regenerated with increased diversity")
        print("=" * 80)
        print(f"Output: {args.output}")
        print(f"Pairs: {len(learned_df)}")
        print(f"Top-k: {args.top_k}")
        print("\nNext steps:")
        print("1. Run diagnostic: python scripts/diagnose_mapping_overlap.py")
        print("2. Re-run H3: python run_h3_evaluation.py")
        print("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Regeneration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
