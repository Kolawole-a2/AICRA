"""ATT&CK-informed playbook generation for AICRA."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .mapping import MappingPipeline


def generate_attack_playbook(
    register_df: pd.DataFrame,
    mapping_pipeline: MappingPipeline,
    output_path: Path,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Generate ATT&CK-informed playbook from risk register.

    For each High/Medium risk asset:
    1. Extract technique_id
    2. Lookup ATT&CK technique details
    3. Map to D3FEND controls
    4. Generate prescriptive actions

    Args:
        register_df: Risk register DataFrame with columns:
            - asset_id (or index)
            - susceptibility_bucket (High/Medium/Low)
            - risk_score or probability
            - expected_loss
            - attack_techniques (list of technique IDs)
            - d3fend_controls (list of control IDs)
            - prescriptive_controls (list of control descriptions)
        mapping_pipeline: MappingPipeline instance for technique lookups
        output_path: Path to save playbook JSON
        threshold: Optional threshold to filter high-risk assets

    Returns:
        Dictionary with playbook structure
    """
    playbook = {
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "high_risk_assets": [],
        "medium_risk_assets": [],
        "low_risk_assets": [],
        "summary": {
            "total_assets": len(register_df),
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "total_expected_loss": 0.0,
        },
        "recommendations": [],
    }

    # Filter by threshold if provided
    if threshold is not None:
        risk_col = (
            "risk_score" if "risk_score" in register_df.columns else "probability"
        )
        if risk_col in register_df.columns:
            register_df = register_df[register_df[risk_col] >= threshold].copy()

    # Process High risk assets
    high_risk = register_df[register_df["susceptibility_bucket"] == "High"].copy()
    playbook["summary"]["high_risk_count"] = len(high_risk)

    for idx, row in high_risk.iterrows():
        asset_id = row.get("asset_id", f"asset_{idx}")

        # Extract techniques and controls
        techniques = row.get("attack_techniques", [])
        if isinstance(techniques, str):
            # Handle string representation of list
            import ast

            try:
                techniques = ast.literal_eval(techniques)
            except (ValueError, SyntaxError):
                techniques = []

        controls = row.get("d3fend_controls", [])
        if isinstance(controls, str):
            import ast

            try:
                controls = ast.literal_eval(controls)
            except (ValueError, SyntaxError):
                controls = []

        prescriptive_controls = row.get("prescriptive_controls", [])
        if isinstance(prescriptive_controls, str):
            import ast

            try:
                prescriptive_controls = ast.literal_eval(prescriptive_controls)
            except (ValueError, SyntaxError):
                prescriptive_controls = []

        playbook["high_risk_assets"].append(
            {
                "asset_id": str(asset_id),
                "risk_score": float(row.get("risk_score", row.get("probability", 0.0))),
                "expected_loss": float(row.get("expected_loss", 0.0)),
                "canonical_family": str(row.get("canonical_family", "unknown")),
                "techniques": techniques if isinstance(techniques, list) else [],
                "d3fend_controls": controls if isinstance(controls, list) else [],
                "prescriptive_controls": prescriptive_controls
                if isinstance(prescriptive_controls, list)
                else [],
            }
        )

    # Process Medium risk assets
    medium_risk = register_df[register_df["susceptibility_bucket"] == "Medium"].copy()
    playbook["summary"]["medium_risk_count"] = len(medium_risk)

    for idx, row in medium_risk.iterrows():
        asset_id = row.get("asset_id", f"asset_{idx}")

        techniques = row.get("attack_techniques", [])
        if isinstance(techniques, str):
            import ast

            try:
                techniques = ast.literal_eval(techniques)
            except (ValueError, SyntaxError):
                techniques = []

        controls = row.get("d3fend_controls", [])
        if isinstance(controls, str):
            import ast

            try:
                controls = ast.literal_eval(controls)
            except (ValueError, SyntaxError):
                controls = []

        prescriptive_controls = row.get("prescriptive_controls", [])
        if isinstance(prescriptive_controls, str):
            import ast

            try:
                prescriptive_controls = ast.literal_eval(prescriptive_controls)
            except (ValueError, SyntaxError):
                prescriptive_controls = []

        playbook["medium_risk_assets"].append(
            {
                "asset_id": str(asset_id),
                "risk_score": float(row.get("risk_score", row.get("probability", 0.0))),
                "expected_loss": float(row.get("expected_loss", 0.0)),
                "canonical_family": str(row.get("canonical_family", "unknown")),
                "techniques": techniques if isinstance(techniques, list) else [],
                "d3fend_controls": controls if isinstance(controls, list) else [],
                "prescriptive_controls": prescriptive_controls
                if isinstance(prescriptive_controls, list)
                else [],
            }
        )

    # Process Low risk assets (summary only)
    low_risk = register_df[register_df["susceptibility_bucket"] == "Low"].copy()
    playbook["summary"]["low_risk_count"] = len(low_risk)

    # Compute total expected loss
    if "expected_loss" in register_df.columns:
        playbook["summary"]["total_expected_loss"] = float(
            register_df["expected_loss"].sum()
        )

    # Generate aggregated recommendations
    # Group by technique and generate recommendations
    all_techniques = set()
    for asset in playbook["high_risk_assets"] + playbook["medium_risk_assets"]:
        all_techniques.update(asset.get("techniques", []))

    for technique_id in all_techniques:
        # Get controls for this technique
        technique_controls = []
        for asset in playbook["high_risk_assets"] + playbook["medium_risk_assets"]:
            if technique_id in asset.get("techniques", []):
                technique_controls.extend(asset.get("d3fend_controls", []))

        unique_controls = list(set(technique_controls))

        playbook["recommendations"].append(
            {
                "technique_id": technique_id,
                "priority": "High"
                if technique_id
                in [
                    t
                    for asset in playbook["high_risk_assets"]
                    for t in asset.get("techniques", [])
                ]
                else "Medium",
                "affected_assets": len(
                    [
                        a
                        for a in playbook["high_risk_assets"]
                        + playbook["medium_risk_assets"]
                        if technique_id in a.get("techniques", [])
                    ]
                ),
                "recommended_controls": unique_controls,
            }
        )

    # Save playbook
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(playbook, f, indent=2)

    return playbook
