#!/usr/bin/env python3
"""Convert smoke_test register to H3 split."""

import pandas as pd
import ast
from pathlib import Path

def extract_technique_id(x):
    """Extract first technique from attack_techniques."""
    if pd.isna(x) or x == '' or x == '[]':
        return None
    try:
        techs = ast.literal_eval(x) if isinstance(x, str) else x
        return str(techs[0]) if isinstance(techs, list) and len(techs) > 0 else None
    except:
        return None

print("Processing smoke_test...")
df = pd.read_csv("register/smoke_test_register.csv")
print(f"  Loaded {len(df)} records")
print(f"  Columns: {list(df.columns)}")

df["asset_id"] = df.index.map(lambda i: f"asset_{i:04d}")
df["risk_score"] = df["probability"].clip(0.0, 1.0)
df["predicted_label"] = (df["risk_score"] >= 0.5).astype(int)
df["true_label"] = df["label"].astype(int)
df["technique_id"] = df["attack_techniques"].apply(extract_technique_id)

h3 = df[["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]]
Path("results/smoke_test").mkdir(parents=True, exist_ok=True)
h3.to_csv("results/smoke_test/risk_scores.csv", index=False)

print(f"  ✓ Created results/smoke_test/risk_scores.csv")
print(f"    Records: {len(h3)}")
print(f"    With technique_id: {h3['technique_id'].notna().sum()}")
print(f"    Unique techniques: {h3['technique_id'].nunique()}")
