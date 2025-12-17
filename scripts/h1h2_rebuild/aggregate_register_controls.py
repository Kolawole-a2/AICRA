#!/usr/bin/env python3
"""
H1/H2 Rebuild – Optional: Aggregate ransomware-only registers to one row per fingerprint.

For each split, this script reads:
    register/h1h2_rebuild/<split>/ransomware_only_risk_register.csv

and writes an aggregated view:
    register/h1h2_rebuild/<split>/ransomware_only_risk_register_AGGREGATED.csv

Row grain:
    One row per sample fingerprint (`sample_fingerprint_sha256`).

Columns:
    - sample_id
    - sample_fingerprint_sha256
    - family_fingerprint_sha256
    - family
    - p_ransomware
    - susceptibility_bucket
    - impact
    - expected_loss
    - attack_techniques       (semicolon-joined unique ATT&CK technique IDs, e.g. "T1486;T1055")
    - d3fend_control_ids      (semicolon-joined unique non-empty D3FEND control IDs)
    - d3fend_control_names    (semicolon-joined unique non-empty D3FEND control names)

Notes:
    - This is a pure post-processing view; it does NOT change the underlying
      per-(sample, technique, control) register.
    - Techniques such as T1055 or T1027 that have no mapped D3FEND controls in
      the H3 lookup tables will still appear in `attack_techniques`, but their
      controls lists will be empty; we do not invent new mappings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def aggregate_split(split_name: str, repo_root: Path) -> Path | None:
    """Aggregate one split's ransomware-only register to one row per fingerprint."""
    in_path = (
        repo_root
        / "register"
        / "h1h2_rebuild"
        / split_name
        / "ransomware_only_risk_register.csv"
    )

    if not in_path.exists():
        logger.warning("Skipping %s: missing %s", split_name, in_path)
        return None

    logger.info("=" * 80)
    logger.info("Aggregating ransomware-only register for split: %s", split_name)
    logger.info("Input: %s", in_path)

    df = pd.read_csv(in_path)
    if df.empty:
        logger.warning("  Input register is empty; writing empty aggregated file.")
        out_path = in_path.with_name(in_path.stem + "_AGGREGATED.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path

    required_cols = {
        "sample_id",
        "sample_fingerprint_sha256",
        "family_fingerprint_sha256",
        "family",
        "p_ransomware",
        "susceptibility_bucket",
        "impact",
        "expected_loss",
        "attack_technique_id",
        "d3fend_control_id",
        "d3fend_control_name",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{split_name}: missing required columns in register: {missing}")

    # Group by fingerprint (one row per fingerprint).
    def _agg_group(group: pd.DataFrame) -> pd.Series:
        # Take stable scalar fields from the first row (they should be constant per sample).
        row0 = group.iloc[0]

        # Unique sorted technique IDs (including those with no mapped controls).
        techs = sorted(
            set(
                str(t)
                for t in group["attack_technique_id"]
                .dropna()
                .astype(str)
                .tolist()
                if t != ""
            )
        )

        # Unique sorted non-empty control IDs / names.
        ctrl_ids = sorted(
            set(
                str(c)
                for c in group["d3fend_control_id"]
                .dropna()
                .astype(str)
                .tolist()
                if c not in ("", "nan", "None", "none", "null")
            )
        )
        ctrl_names = sorted(
            set(
                str(n)
                for n in group["d3fend_control_name"]
                .dropna()
                .astype(str)
                .tolist()
                if n not in ("", "nan", "None", "none", "null")
            )
        )

        return pd.Series(
            {
                "sample_id": row0["sample_id"],
                "sample_fingerprint_sha256": row0["sample_fingerprint_sha256"],
                "family_fingerprint_sha256": row0["family_fingerprint_sha256"],
                "family": row0["family"],
                "p_ransomware": float(row0["p_ransomware"]),
                "susceptibility_bucket": row0["susceptibility_bucket"],
                "impact": float(row0["impact"]),
                "expected_loss": float(row0["expected_loss"]),
                "attack_techniques": ";".join(techs),
                "d3fend_control_ids": ";".join(ctrl_ids),
                "d3fend_control_names": ";".join(ctrl_names),
            }
        )

    grouped = df.groupby("sample_fingerprint_sha256", sort=False)
    out_df = grouped.apply(_agg_group).reset_index(drop=True)

    out_path = in_path.with_name(in_path.stem + "_AGGREGATED.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    logger.info(
        "  ✓ Wrote aggregated register for %s: %s (%d rows)",
        split_name,
        out_path,
        len(out_df),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate ransomware-only registers to one row per fingerprint "
            "with semicolon-joined techniques and controls."
        )
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["smoke_test", "small_ember", "main", "full_ember"],
        help="Splits to aggregate (default: smoke_test small_ember main full_ember)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent

    for split in args.splits:
        try:
            aggregate_split(split, repo_root)
        except Exception as e:
            logger.error("Error aggregating split %s: %s", split, e)


if __name__ == "__main__":
    main()


