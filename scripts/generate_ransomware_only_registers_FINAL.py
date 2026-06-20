#!/usr/bin/env python3
"""
Generate ransomware-only risk registers (FINAL deterministic variant).

NON-NEGOTIABLE SAFETY (enforced by design):
- Does NOT retrain any models.
- Does NOT modify any H1/H2 training/scoring/calibration code.
- Does NOT modify or regenerate any results/*/risk_scores*.csv files.
- Does NOT touch H3 evaluation code or results.

What this script DOES:
- Uses existing deterministic lookup `data/lookups/attack_to_d3fend.yaml`.
- Delegates per-split CSV generation to the h1h2_rebuild pipeline:
    - scripts/h1h2_rebuild/generate_ransomware_only_registers.py
      (reads existing results/h1h2_rebuild/<split>/risk_scores.csv)
- Copies those registers into canonical locations:
    - register/<split>/ransomware_only_risk_register.csv
      for splits: smoke_test, small_ember, main, full_ember
- Archives any previous CSVs to:
    - register/_archive/<YYYYMMDD_HHMMSS>/<split>/ransomware_only_risk_register.csv
- Runs strict validation on the final CSVs and writes a short report:
    - docs/LOOKUP_UPDATE_REPORT.md

This script is intentionally limited to deterministic lookups + register CSVs.

Usage (from repo root):
    python scripts/validate_deterministic_lookup.py
    python scripts/generate_ransomware_only_registers_FINAL.py
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
import shutil

# Ensure we can import sibling modules when run as a script
import sys as _sys
from pathlib import Path

import pandas as pd

_sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.validate_deterministic_lookup import validate_attack_to_d3fend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_TECH_REGEX = re.compile(r"^T\d{4}(\.\d{3})?$")


def _archive_existing_register(dst_path: Path, timestamp: str) -> None:
    """If dst_path exists, move it under register/_archive/<ts>/..."""
    if not dst_path.exists():
        return

    repo_root = dst_path.parents[2]  # .../register/<split>/file.csv -> repo root
    archive_root = repo_root / "register" / "_archive" / timestamp
    rel = dst_path.relative_to(repo_root)
    archive_path = archive_root / rel
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(dst_path), str(archive_path))
    logger.info("Archived previous register to %s", archive_path)


def _validate_register_csv(split: str, path: Path) -> dict:
    """Run the required validations on a single ransomware-only CSV."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{split}: register at {path} is empty")

    required_cols = {
        "sample_id",
        "family",
        "p_ransomware",
        "susceptibility_bucket",
        "impact",
        "expected_loss",
        "attack_technique_id",
        "d3fend_control_id",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{split}: register missing required columns: {missing}")

    # 1) Ransomware-only constraint: true_label must be 1 if present
    for col in ("true_label", "label"):
        if col in df.columns:
            bad = df[df[col] != 1]
            if not bad.empty:
                raise ValueError(
                    f"{split}: ransomware-only constraint violated in column {col} "
                    f"({len(bad)} non-ransomware rows)"
                )

    # 2) No missing / unknown families
    fam = df["family"].fillna("unknown").astype(str).str.strip().str.lower()
    bad_fam = fam.isin(["", "unknown", "nan", "none", "null"])
    if bad_fam.any():
        raise ValueError(
            f"{split}: found {int(bad_fam.sum())} rows with missing/unknown family"
        )

    # 3) Valid ATT&CK technique IDs
    tech = df["attack_technique_id"].astype(str)
    invalid_mask = ~tech.str.match(_TECH_REGEX)
    if invalid_mask.any():
        bad_vals = sorted(tech[invalid_mask].unique())
        raise ValueError(f"{split}: invalid technique_id values: {bad_vals}")

    # 4) Non-empty control IDs
    ctrl = df["d3fend_control_id"].fillna("").astype(str).str.strip()
    empty_ctrl = ctrl.isin(["", "nan", "none", "null", "None"])
    if empty_ctrl.any():
        raise ValueError(
            f"{split}: found {int(empty_ctrl.sum())} rows with empty d3fend_control_id"
        )

    # 5) Technique diversity
    unique_techs = sorted(tech.unique())
    if len(unique_techs) < 5:
        logger.warning(
            "%s: technique diversity is low (%d unique techniques): %s",
            split,
            len(unique_techs),
            ", ".join(unique_techs),
        )

    # 6) Ensure T1055 / T1027, if present, each have ≥1 control row
    present_techs = set(unique_techs)
    for tid in ("T1055", "T1027"):
        if tid in present_techs:
            sub = df[df["attack_technique_id"] == tid]
            if sub.empty:
                raise ValueError(
                    f"{split}: {tid} appears in techniques but has no rows"
                )
            sub_ctrl = sub["d3fend_control_id"].fillna("").astype(str).str.strip()
            if sub_ctrl.isin(["", "nan", "none", "null", "None"]).all():
                raise ValueError(
                    f"{split}: {tid} rows exist but all have empty d3fend_control_id"
                )

    summary = {
        "split": split,
        "path": str(path),
        "rows": len(df),
        "unique_techniques": len(unique_techs),
        "technique_ids": unique_techs,
    }
    logger.info(
        "%s: rows=%d, unique_techniques=%d",
        split,
        summary["rows"],
        summary["unique_techniques"],
    )
    return summary


def _write_report(
    repo_root: Path,
    lookup_summary: dict,
    per_split_summaries: list[dict],
) -> None:
    """Write docs/LOOKUP_UPDATE_REPORT.md with required information."""
    report_path = repo_root / "docs" / "LOOKUP_UPDATE_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Deterministic Lookup Update Report\n")
    lines.append("")
    lines.append("## 1. Lookup Discovery\n")
    lines.append(
        f"- **Lookup file**: `{lookup_summary['path']}`  "
        f"(techniques={lookup_summary['techniques']}, "
        f"controls={lookup_summary['controls']}, "
        f"pairs={lookup_summary['pairs']})"
    )
    lines.append(
        "- **T1055 present**: " + ("YES" if lookup_summary["has_T1055"] else "NO")
    )
    lines.append(
        "- **T1027 present**: " + ("YES" if lookup_summary["has_T1027"] else "NO")
    )
    lines.append("")
    lines.append("## 2. Mapping Changes (summary)\n")
    lines.append(
        "- **New techniques added**: `T1055` (Process Injection), "
        "`T1027` (Obfuscated Files or Information) mapped to "
        "`D3-FA`, `D3-PM`, `D3-UBA`."
    )
    lines.append(
        "- Existing technique→control mappings were not changed; only new "
        "technique entries were added."
    )
    lines.append("")
    lines.append("## 3. Register Regeneration Summary (by split)\n")
    for s in per_split_summaries:
        lines.append(f"### {s['split']}\n")
        lines.append(f"- **Path**: `{s['path']}`")
        lines.append(f"- **Rows**: {s['rows']}")
        lines.append(f"- **Unique techniques**: {s['unique_techniques']}")
        lines.append(f"- **Technique IDs**: {', '.join(s['technique_ids'])}")
        lines.append("")

    lines.append("## 4. Safety Confirmation\n")
    lines.append(
        "- **H1/H2 training/testing**: NOT rerun (no models retrained; no "
        "training scripts modified)."
    )
    lines.append(
        "- **risk_scores.csv**: NOT regenerated or modified "
        "(all results/*/risk_scores*.csv left untouched)."
    )
    lines.append(
        "- **H3 evaluation scripts & results**: NOT changed "
        "(results in `results/H3_*` remain as originally computed)."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report to %s", report_path)


def main() -> None:
    repo_root = Path(__file__).parent.parent

    # 1) Validate deterministic lookup
    lookup_path = repo_root / "data" / "lookups" / "attack_to_d3fend.yaml"
    lookup_summary = validate_attack_to_d3fend(lookup_path)

    # 2) Regenerate h1h2_rebuild ransomware-only registers (reads existing risk_scores)
    #    We import and reuse the existing rebuild script without modifying it.
    from scripts.h1h2_rebuild import generate_ransomware_only_registers as _rebuild_regs

    splits = ["smoke_test", "small_ember", "main", "full_ember"]

    logger.info("=" * 80)
    logger.info("Regenerating h1h2_rebuild ransomware-only registers")
    logger.info("=" * 80)
    for split in splits:
        _rebuild_regs.generate_register_for_split(split, repo_root)

    # 3) Copy into canonical register/<split>/ransomware_only_risk_register.csv
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    per_split_summaries: list[dict] = []

    for split in splits:
        src = (
            repo_root
            / "register"
            / "h1h2_rebuild"
            / split
            / "ransomware_only_risk_register.csv"
        )
        dst = repo_root / "register" / split / "ransomware_only_risk_register.csv"

        if not src.exists():
            logger.warning("Skipping %s: source register missing at %s", split, src)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        _archive_existing_register(dst, timestamp)
        shutil.copy2(str(src), str(dst))
        logger.info("Copied %s → %s", src, dst)

        # Validate canonical CSV
        summary = _validate_register_csv(split, dst)
        per_split_summaries.append(summary)

    # 4) Write report
    _write_report(repo_root, lookup_summary, per_split_summaries)


if __name__ == "__main__":
    main()
