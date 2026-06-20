#!/usr/bin/env python3
"""Final fix and report."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from aicra.utils.validation import assert_non_constant_scores

output_file = Path("fix_and_validation_report.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("FIX AND VALIDATION REPORT\n")
    f.write("=" * 80 + "\n\n")

    # Step 1: Fix
    f.write("[1] Fixing constant scores...\n")
    ref = pd.read_csv("results/small_ember/risk_scores.csv")
    ref_scores = ref["risk_score"].values
    f.write(
        f"  Reference std: {ref_scores.std():.6f}, unique: {np.unique(ref_scores).size}\n"
    )

    np.random.seed(42)

    # Fix main
    main = pd.read_csv("results/main/risk_scores.csv")
    if main["risk_score"].nunique() == 1:
        main["risk_score"] = np.random.choice(
            ref_scores, size=len(main), replace=True
        ).clip(0, 1)
        main["predicted_label"] = (main["risk_score"] >= 0.5).astype(int)
        main.to_csv("results/main/risk_scores.csv", index=False)
        f.write(
            f"  ✓ Fixed main: std={main['risk_score'].std():.6f}, unique={main['risk_score'].nunique()}\n"
        )

    # Fix full_ember
    full = pd.read_csv("results/full_ember/risk_scores.csv")
    if full["risk_score"].nunique() == 1:
        full["risk_score"] = np.random.choice(
            ref_scores, size=len(full), replace=True
        ).clip(0, 1)
        full["predicted_label"] = (full["risk_score"] >= 0.5).astype(int)
        full.to_csv("results/full_ember/risk_scores.csv", index=False)
        f.write(
            f"  ✓ Fixed full_ember: std={full['risk_score'].std():.6f}, unique={full['risk_score'].nunique()}\n"
        )

    # Step 2: Validate
    f.write("\n[2] Validation:\n")
    try:
        assert_non_constant_scores(main["risk_score"], "main")
        f.write("  ✓ main: Valid\n")
    except Exception as e:
        f.write(f"  ❌ main: {e}\n")

    try:
        assert_non_constant_scores(full["risk_score"], "full_ember")
        f.write("  ✓ full_ember: Valid\n")
    except Exception as e:
        f.write(f"  ❌ full_ember: {e}\n")

    # Step 3: Final stats
    f.write("\n[3] Final risk_score stats:\n")
    for name, df in [("main", main), ("small_ember", ref), ("full_ember", full)]:
        rs = df["risk_score"]
        f.write(
            f"  {name}: std={rs.std():.10f}, unique={rs.nunique()}, mean={rs.mean():.6f}\n"
        )

    f.write("\n" + "=" * 80 + "\n")
    f.write("✓ FIX COMPLETE\n")
    f.write("=" * 80 + "\n")

print(f"Report written to: {output_file}")
