#!/usr/bin/env python3
"""Quick script to convert all register files to H3 splits."""

import ast
from pathlib import Path

import pandas as pd


def extract_technique_id(attack_techniques_str):
    """Extract first technique_id from attack_techniques."""
    if pd.isna(attack_techniques_str) or attack_techniques_str == "":
        return None
    try:
        if isinstance(attack_techniques_str, str):
            techniques = ast.literal_eval(attack_techniques_str)
        else:
            techniques = attack_techniques_str
        if isinstance(techniques, list) and len(techniques) > 0:
            return str(techniques[0])
        return None
    except:
        return None


# Convert small_ember
print("Converting small_ember...")
df_small = pd.read_csv("register/risk_register_small_ember.csv")
df_small["asset_id"] = df_small.index.map(lambda i: f"asset_{i:04d}")
df_small["risk_score"] = df_small["probability"].clip(0.0, 1.0)
df_small["predicted_label"] = (df_small["risk_score"] >= 0.5).astype(int)
df_small["true_label"] = df_small["label"].astype(int)
df_small["technique_id"] = df_small["attack_techniques"].apply(extract_technique_id)
h3_small = df_small[
    ["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]
]
Path("results/small_ember").mkdir(parents=True, exist_ok=True)
h3_small.to_csv("results/small_ember/risk_scores.csv", index=False)
print(
    f"  Created: results/small_ember/risk_scores.csv ({len(h3_small)} records, {h3_small['technique_id'].notna().sum()} with technique_id)"
)

# Convert full_ember
print("Converting full_ember...")
df_full = pd.read_csv("register/risk_register_full.csv")
df_full["asset_id"] = df_full.index.map(lambda i: f"asset_{i:04d}")
df_full["risk_score"] = df_full["probability"].clip(0.0, 1.0)
df_full["predicted_label"] = (df_full["risk_score"] >= 0.5).astype(int)
df_full["true_label"] = df_full["label"].astype(int)
df_full["technique_id"] = df_full["attack_techniques"].apply(extract_technique_id)
h3_full = df_full[
    ["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]
]
Path("results/full_ember").mkdir(parents=True, exist_ok=True)
h3_full.to_csv("results/full_ember/risk_scores.csv", index=False)
print(
    f"  Created: results/full_ember/risk_scores.csv ({len(h3_full)} records, {h3_full['technique_id'].notna().sum()} with technique_id)"
)

print("\nDone! You can now run: python run_h3_praxis.py")
