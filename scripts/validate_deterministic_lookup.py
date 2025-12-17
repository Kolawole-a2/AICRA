#!/usr/bin/env python3
"""
Validate deterministic ATT&CK → D3FEND lookup used for ransomware-only registers.

Checks:
- data/lookups/attack_to_d3fend.yaml exists and is well-formed
- Every technique maps to ≥1 non-empty control ID
- T1055 and T1027 both exist and have ≥1 control

This script is read-only and does NOT touch any models, risk scores, or H3 results.

Usage (from repo root):
    python scripts/validate_deterministic_lookup.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_attack_to_d3fend(path: Path) -> dict:
    """Validate attack_to_d3fend.yaml and return summary stats."""
    if not path.exists():
        raise FileNotFoundError(f"Deterministic lookup not found: {path}")

    logger.info("Validating deterministic lookup: %s", path)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    mappings = data.get("mappings", {})
    if not isinstance(mappings, dict) or not mappings:
        raise ValueError("attack_to_d3fend.yaml: 'mappings' must be a non-empty dict")

    n_pairs = 0
    all_controls: set[str] = set()

    for tech, ctrls in mappings.items():
        if not isinstance(ctrls, list) or not ctrls:
            raise ValueError(f"Technique {tech} has no mapped controls")
        clean_ctrls = [str(c).strip() for c in ctrls if str(c).strip()]
        if not clean_ctrls:
            raise ValueError(f"Technique {tech} has only empty/blank controls")
        n_pairs += len(clean_ctrls)
        all_controls.update(clean_ctrls)

    # Explicitly check T1055 / T1027 presence
    for tech_id in ("T1055", "T1027"):
        ctrls = mappings.get(tech_id)
        if not ctrls or not [c for c in ctrls if str(c).strip()]:
            raise ValueError(f"{tech_id} is missing or has no non-empty controls in lookup")

    summary = {
        "path": str(path),
        "techniques": len(mappings),
        "controls": len(all_controls),
        "pairs": n_pairs,
        "has_T1055": "T1055" in mappings,
        "has_T1027": "T1027" in mappings,
    }

    logger.info(
        "Lookup summary: %d techniques, %d controls, %d technique-control pairs",
        summary["techniques"],
        summary["controls"],
        summary["pairs"],
    )
    logger.info(
        "Presence: T1055=%s, T1027=%s",
        summary["has_T1055"],
        summary["has_T1027"],
    )
    return summary


def main() -> None:
    repo_root = Path(__file__).parent.parent
    lookup_path = repo_root / "data" / "lookups" / "attack_to_d3fend.yaml"
    validate_attack_to_d3fend(lookup_path)


if __name__ == "__main__":
    main()


