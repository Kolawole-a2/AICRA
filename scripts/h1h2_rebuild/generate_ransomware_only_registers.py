#!/usr/bin/env python3
"""
H1/H2 Rebuild – Phase 4: Generate Ransomware-Only Risk Registers

This script consumes the calibrated per-sample `risk_scores.csv` files from:
    results/h1h2_rebuild/<split>/risk_scores.csv

and produces **ransomware-only** risk registers for each split using the
deterministic H3 lookup tables:
    - data/lookups/family_to_attack.yaml
    - data/lookups/attack_to_d3fend.yaml

Outputs:
    register/h1h2_rebuild/<split>/ransomware_only_risk_register.csv

Row grain:
    One row per (sample, ATT&CK technique, D3FEND control).

Columns:
    - sample_id
    - family
    - p_ransomware
    - susceptibility_bucket   (Low/Med/High)
    - impact                  (constant, e.g., 5_000_000)
    - expected_loss           (p_ransomware * impact)
    - attack_technique_id
    - d3fend_control_id
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Static helper to give human-readable names for each D3FEND control ID.
_D3FEND_CONTROL_NAMES: dict[str, str] = {
    # Controls already used in earlier experiments / documentation
    "D3-BDR": "Backup Data Recovery",
    "D3-BAC": "Backup Access Control",
    "D3-SAW": "System Access Workflow",
    "D3-CR": "Command Restriction",
    "D3-AL": "Application Allowlisting",
    "D3-NFP": "Network Filtering Policy",
    "D3-VPM": "Virtual Private Network",
    "D3-AA": "Access Authentication",
    "D3-EDR": "Endpoint Detection and Response",
    "D3-SIEM": "Security Information and Event Management",
    "D3-AV": "Antivirus",
    # Additional D3FEND IDs appearing in deterministic_attack_defense_lookup.csv
    "D3-RA": "Restore Access",
    "D3-DO": "Decoy Object",
    "D3-DE": "Decoy Environment",
    "D3-FA": "File Analysis",
    "D3-PM": "Platform Monitoring",
    "D3-UBA": "User Behavior Analysis",
    "D3-AI": "Asset Inventory",
    "D3-PE": "Process Eviction",
    "D3-AMED": "Access Mediation",
}


def load_family_to_attack(path: Path) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    mappings = data.get("mappings", {})
    # Normalize keys to lower case for robust lookup
    return {str(k).lower(): v for k, v in mappings.items()}


def load_attack_to_d3fend(path: Path) -> dict[str, list[str]]:
    """Load deterministic ATT&CK → D3FEND mapping as {tech_id: [control_ids...]}."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    mappings = data.get("mappings", {})
    clean: dict[str, list[str]] = {}
    for tech, ctrls in mappings.items():
        if not isinstance(ctrls, list):
            continue
        # Strip inline comments and whitespace from each control entry
        cleaned_ctrls: list[str] = []
        for c in ctrls:
            s = str(c)
            # If YAML was loaded as plain strings, comments are already stripped;
            # but if not, split on '#'.
            s = s.split("#", 1)[0].strip()
            if s:
                cleaned_ctrls.append(s)
        if cleaned_ctrls:
            clean[str(tech)] = cleaned_ctrls
    return clean


def susceptibility_bucket(p: float) -> str:
    if p < 0.3:
        return "Low"
    if p < 0.7:
        return "Med"
    return "High"


def generate_register_for_split(
    split_name: str,
    repo_root: Path,
    impact: float = 5_000_000.0,
) -> Path:
    """Generate ransomware-only risk register for a single split."""
    logger.info("=" * 80)
    logger.info(f"Generating ransomware-only register for split: {split_name}")
    logger.info("=" * 80)

    risk_path = repo_root / "results" / "h1h2_rebuild" / split_name / "risk_scores.csv"
    if not risk_path.exists():
        raise FileNotFoundError(
            f"risk_scores.csv not found for split {split_name}: {risk_path}"
        )

    df = pd.read_csv(risk_path)
    logger.info(f"Loaded {len(df)} rows from {risk_path}")

    # Filter to ransomware (label==1)
    df = df[df["true_label"] == 1].copy()
    logger.info(f"  Ransomware rows (label==1): {len(df)}")

    # Drop rows with unknown / missing family
    fam = df["family"].fillna("unknown").astype(str).str.strip().str.lower()
    mask_unknown = fam.isin(["", "unknown", "nan", "none", "null"])
    n_unknown = int(mask_unknown.sum())
    if n_unknown > 0:
        logger.info(f"  Dropping {n_unknown} rows with unknown / missing family")
        df = df[~mask_unknown].copy()
        fam = df["family"].fillna("unknown").astype(str).str.strip().str.lower()

    if len(df) == 0:
        logger.warning("  No ransomware rows with valid families remain; skipping.")
        out_dir = repo_root / "register" / "h1h2_rebuild" / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "ransomware_only_risk_register.csv"
        df.to_csv(out_path, index=False)
        return out_path

    # Load lookups
    fam_to_attack = load_family_to_attack(
        repo_root / "data" / "lookups" / "family_to_attack.yaml"
    )
    attack_to_d3 = load_attack_to_d3fend(
        repo_root / "data" / "lookups" / "attack_to_d3fend.yaml"
    )

    records: list[dict] = []

    for _, row in df.iterrows():
        sample_id = str(row["sample_id"])
        family = str(row["family"]).strip()
        p = float(row["p_ransomware"])
        bucket = susceptibility_bucket(p)
        expected_loss = p * impact

        # Deterministic sample fingerprint (SHA-256 of stable sample identifier).
        # Note: this is a dataset-stable fingerprint, not the original file hash.
        fingerprint = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        # Deterministic family fingerprint (SHA-256 of normalized family name).
        family_fingerprint = hashlib.sha256(family.lower().encode("utf-8")).hexdigest()

        fam_key = family.lower()
        techniques = fam_to_attack.get(fam_key, [])
        if not techniques:
            # Skip families with no mapped techniques (cannot build register rows)
            continue

        for tech in techniques:
            d3_controls = attack_to_d3.get(tech, [])
            if not d3_controls:
                # Still emit row with empty control id to show technique coverage
                records.append(
                    {
                        "sample_id": sample_id,
                        "sample_fingerprint_sha256": fingerprint,
                        "family_fingerprint_sha256": family_fingerprint,
                        "family": family,
                        "p_ransomware": p,
                        "susceptibility_bucket": bucket,
                        "impact": impact,
                        "expected_loss": expected_loss,
                        "attack_technique_id": tech,
                        "d3fend_control_id": "",
                        "d3fend_control_name": "",
                    }
                )
            else:
                for ctrl in d3_controls:
                    records.append(
                        {
                            "sample_id": sample_id,
                            "sample_fingerprint_sha256": fingerprint,
                            "family_fingerprint_sha256": family_fingerprint,
                            "family": family,
                            "p_ransomware": p,
                            "susceptibility_bucket": bucket,
                            "impact": impact,
                            "expected_loss": expected_loss,
                            "attack_technique_id": tech,
                            "d3fend_control_id": ctrl,
                            "d3fend_control_name": _D3FEND_CONTROL_NAMES.get(ctrl, ""),
                        }
                    )

    out_dir = repo_root / "register" / "h1h2_rebuild" / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ransomware_only_risk_register.csv"

    reg_df = pd.DataFrame.from_records(records)
    reg_df.to_csv(out_path, index=False)

    logger.info(f"  ✓ Wrote register for {split_name}: {out_path} ({len(reg_df)} rows)")
    return out_path


def main() -> None:
    repo_root = Path(__file__).parent.parent.parent
    splits = ["smoke_test", "small_ember", "main", "full_ember"]

    logger.info("=" * 80)
    logger.info("H1/H2 Rebuild – Phase 4: Ransomware-only registers")
    logger.info("=" * 80)

    for split in splits:
        try:
            generate_register_for_split(split, repo_root)
        except FileNotFoundError as e:
            logger.warning(f"Skipping split {split}: {e}")
            continue


if __name__ == "__main__":
    main()
