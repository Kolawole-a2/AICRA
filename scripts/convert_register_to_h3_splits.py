#!/usr/bin/env python3
"""
Convert EMBER register files to H3 evaluation split format.

This script converts register CSV files (from small_ember, full_ember, etc.)
into risk score CSV files that can be used as H3 evaluation splits.

Each register file will be converted to a risk_scores CSV with:
- asset_id (from index)
- risk_score (from probability column)
- predicted_label (derived from probability > threshold, default 0.5)
- true_label (from label column)
- technique_id (extracted from attack_techniques list)
"""

import sys
import ast
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_technique_id(attack_techniques_str: str) -> Optional[str]:
    """
    Extract first technique_id from attack_techniques string/list.
    
    Args:
        attack_techniques_str: String representation of list or actual list
        
    Returns:
        First technique_id or None if empty
    """
    if pd.isna(attack_techniques_str) or attack_techniques_str == '':
        return None
    
    try:
        # Try to parse as Python list
        if isinstance(attack_techniques_str, str):
            techniques = ast.literal_eval(attack_techniques_str)
        else:
            techniques = attack_techniques_str
        
        if isinstance(techniques, list) and len(techniques) > 0:
            # Return first technique
            return str(techniques[0])
        elif isinstance(techniques, str) and techniques:
            return techniques
        else:
            return None
    except (ValueError, SyntaxError):
        # If parsing fails, try treating as string
        if isinstance(attack_techniques_str, str) and attack_techniques_str.strip():
            # Try to extract first technique pattern (e.g., "T1486")
            import re
            match = re.search(r'T\d{4}(?:\.\d{3})?', attack_techniques_str)
            if match:
                return match.group(0)
        return None


def convert_register_to_risk_scores(
    register_path: Path,
    output_path: Path,
    threshold: float = 0.5,
    split_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert register CSV to H3 risk scores CSV format.
    
    Args:
        register_path: Path to register CSV file
        output_path: Path to save risk scores CSV
        threshold: Probability threshold for predicted_label (default 0.5)
        split_name: Optional name for the split (for logging)
        
    Returns:
        DataFrame with risk scores in H3 format
    """
    logger.info(f"Converting register file: {register_path}")
    
    if not register_path.exists():
        raise FileNotFoundError(f"Register file not found: {register_path}")
    
    # Load register
    df = pd.read_csv(register_path)
    logger.info(f"Loaded {len(df)} records from register")
    
    # Check required columns
    required_cols = {"probability", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Register file missing required columns: {missing}. Found: {list(df.columns)}")
    
    # Create asset_id from index
    df["asset_id"] = df.index.map(lambda i: f"asset_{i:04d}")
    
    # Map probability to risk_score
    df["risk_score"] = df["probability"].clip(0.0, 1.0)
    
    # Create predicted_label from probability threshold
    df["predicted_label"] = (df["risk_score"] >= threshold).astype(int)
    
    # Map label to true_label
    df["true_label"] = df["label"].astype(int)
    
    # Extract technique_id from attack_techniques
    if "attack_techniques" in df.columns:
        df["technique_id"] = df["attack_techniques"].apply(extract_technique_id)
    elif "technique_id" in df.columns:
        # Already has technique_id
        df["technique_id"] = df["technique_id"].astype(str)
    else:
        logger.warning("No attack_techniques or technique_id column found. Setting technique_id to None.")
        df["technique_id"] = None
    
    # Select and reorder columns for H3 format
    h3_columns = ["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]
    h3_df = df[h3_columns].copy()
    
    # Filter out rows with missing technique_id (if needed)
    # For H3, we need technique_id, but we'll keep them and let the evaluation handle it
    n_with_technique = h3_df["technique_id"].notna().sum()
    logger.info(f"Records with technique_id: {n_with_technique}/{len(h3_df)}")
    
    # VALIDATION: Ensure risk scores are not constant before writing
    from ..utils.validation import assert_non_constant_scores, validate_risk_scores_file
    split_name = split_name or output_path.stem
    assert_non_constant_scores(h3_df["risk_score"], split_name=split_name, min_unique=5, min_std=1e-6)
    
    # Save to output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h3_df.to_csv(output_path, index=False)
    
    # FINAL VALIDATION: Ensure risk scores are not constant after writing
    validate_risk_scores_file(output_path, split_name)
    
    logger.info(f"Saved H3 risk scores to: {output_path}")
    logger.info(f"  Total records: {len(h3_df)}")
    logger.info(f"  Records with technique_id: {n_with_technique}")
    logger.info(f"  Unique techniques: {h3_df['technique_id'].nunique()}")
    
    return h3_df


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert EMBER register files to H3 evaluation split format"
    )
    parser.add_argument(
        "--register",
        type=Path,
        required=True,
        help="Path to register CSV file (e.g., register/risk_register_small_ember.csv)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for risk scores CSV (default: results/<split_name>/risk_scores.csv)"
    )
    parser.add_argument(
        "--split-name",
        type=str,
        help="Name for this split (e.g., 'small_ember', 'full_ember')"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for predicted_label (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Determine split name from register filename if not provided
    if args.split_name is None:
        # Extract from filename: risk_register_<name>.csv -> <name>
        stem = args.register.stem
        if stem.startswith("risk_register_"):
            args.split_name = stem.replace("risk_register_", "")
        else:
            args.split_name = stem
    
    # Determine output path if not provided
    if args.output is None:
        repo_root = Path(__file__).parent.parent
        args.output = repo_root / "results" / args.split_name / "risk_scores.csv"
    
    # Convert register to risk scores
    try:
        h3_df = convert_register_to_risk_scores(
            register_path=args.register,
            output_path=args.output,
            threshold=args.threshold,
            split_name=args.split_name
        )
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("Conversion Complete!")
        logger.info("=" * 80)
        logger.info(f"Split name: {args.split_name}")
        logger.info(f"Output file: {args.output}")
        logger.info(f"Total records: {len(h3_df)}")
        logger.info(f"Records with technique_id: {h3_df['technique_id'].notna().sum()}")
        logger.info(f"Unique techniques: {h3_df['technique_id'].nunique()}")
        logger.info("")
        logger.info("You can now add this split to config/h3_splits.yaml:")
        logger.info(f"  {args.split_name}: \"{args.output.relative_to(Path.cwd())}\"")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
