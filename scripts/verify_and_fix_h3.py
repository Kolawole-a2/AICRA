#!/usr/bin/env python3
"""Quick script to verify and fix H3 mapping diversity."""

from pathlib import Path

import pandas as pd

repo_root = Path(__file__).parent.parent

# Check risk scores
risk_df = pd.read_csv(repo_root / "risk_scores.csv")
risk_techs = set(risk_df["technique_id"].unique())
print(f"Techniques in risk scores: {len(risk_techs)}")
print(f"Sample: {sorted(list(risk_techs))[:10]}")

# Check learned mapping
learned_df = pd.read_csv(repo_root / "data/mappings/learned_mapping.csv")
learned_techs = set(learned_df["technique_id"].unique())
print(f"\nTechniques in learned mapping: {len(learned_techs)}")

# Check deterministic
det_df = pd.read_csv(repo_root / "data/mappings/deterministic_lookup.csv")
det_tech_col = "attack_id" if "attack_id" in det_df.columns else "technique_id"
det_techs = set(det_df[det_tech_col].unique())
print(f"Techniques in deterministic: {len(det_techs)}")

# Check overlap
print(
    f"\nRisk scores covered by learned: {len(risk_techs & learned_techs)}/{len(risk_techs)}"
)
print(
    f"Risk scores covered by deterministic: {len(risk_techs & det_techs)}/{len(risk_techs)}"
)

# Check for techniques in risk scores
missing_in_learned = risk_techs - learned_techs
if missing_in_learned:
    print(f"\n⚠️  Missing in learned mapping: {sorted(list(missing_in_learned))[:10]}")
