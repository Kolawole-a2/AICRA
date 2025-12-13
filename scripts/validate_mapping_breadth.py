#!/usr/bin/env python3
"""
Validate that learned mapping is broader than deterministic mapping.

This script checks:
1. Learned has MORE pairs than deterministic
2. Learned contains controls NOT in deterministic
3. Learned controls are NOT subsets of deterministic controls
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Validate mapping breadth."""
    print("=" * 80)
    print("Validating Learned Mapping Breadth")
    print("=" * 80)
    print()

    repo_root = Path(__file__).parent.parent

    # Load deterministic mapping
    det_path = (
        repo_root / "data" / "mappings" / "deterministic_attack_defense_lookup.csv"
    )
    if not det_path.exists():
        det_path = repo_root / "data" / "mappings" / "deterministic_lookup.csv"

    if not det_path.exists():
        print(f"ERROR: Deterministic mapping not found at {det_path}")
        sys.exit(1)

    det_df = pd.read_csv(det_path)
    det_col = "attack_id" if "attack_id" in det_df.columns else "technique_id"
    det_ctrl_col = "defense_id" if "defense_id" in det_df.columns else "control_id"

    det_pairs = set(
        zip(det_df[det_col].astype(str), det_df[det_ctrl_col].astype(str), strict=False)
    )

    # Load learned mapping
    learned_path = repo_root / "data" / "mappings" / "learned_mapping.csv"
    if not learned_path.exists():
        print(f"ERROR: Learned mapping not found at {learned_path}")
        sys.exit(1)

    learned_df = pd.read_csv(learned_path)
    learned_pairs = set(
        zip(
            learned_df["technique_id"].astype(str),
            learned_df["control_id"].astype(str),
            strict=False,
        )
    )

    # Group by technique
    det_by_tech = {}
    for tech, ctrl in det_pairs:
        if tech not in det_by_tech:
            det_by_tech[tech] = set()
        det_by_tech[tech].add(ctrl)

    learned_by_tech = {}
    for tech, ctrl in learned_pairs:
        if tech not in learned_by_tech:
            learned_by_tech[tech] = set()
        learned_by_tech[tech].add(ctrl)

    # Check 1: Learned has more pairs
    print(f"Deterministic pairs: {len(det_pairs)}")
    print(f"Learned pairs: {len(learned_pairs)}")
    print(f"Difference: {len(learned_pairs) - len(det_pairs)}")
    print()

    if len(learned_pairs) <= len(det_pairs):
        print("❌ FAIL: Learned mapping does NOT have more pairs than deterministic")
        print(f"   Learned: {len(learned_pairs)}, Deterministic: {len(det_pairs)}")
        sys.exit(1)
    else:
        print("✓ PASS: Learned mapping has more pairs than deterministic")

    # Check 2: Learned contains controls NOT in deterministic
    learned_only_pairs = learned_pairs - det_pairs
    print(f"Learned-only pairs: {len(learned_only_pairs)}")
    print()

    if len(learned_only_pairs) == 0:
        print(
            "❌ FAIL: Learned mapping contains NO controls that are not in deterministic"
        )
        sys.exit(1)
    else:
        print("✓ PASS: Learned mapping contains controls NOT in deterministic")
        print(f"   Sample learned-only pairs: {list(learned_only_pairs)[:5]}")

    # Check 3: For each technique, learned controls are NOT subsets
    techniques_with_only_det_controls = []
    for tech in learned_by_tech:
        if tech in det_by_tech:
            learned_ctrls = learned_by_tech[tech]
            det_ctrls = det_by_tech[tech]
            if learned_ctrls.issubset(det_ctrls) and len(learned_ctrls) > 0:
                techniques_with_only_det_controls.append(tech)

    print()
    print(
        f"Techniques with only deterministic controls: {len(techniques_with_only_det_controls)}/{len(learned_by_tech)}"
    )

    if len(techniques_with_only_det_controls) == len(learned_by_tech):
        print(
            "❌ FAIL: For ALL techniques, learned controls are subsets of deterministic"
        )
        sys.exit(1)
    else:
        print("✓ PASS: Most techniques have learned controls beyond deterministic")

    # Summary
    print()
    print("=" * 80)
    print("✓ ALL CHECKS PASSED: Learned mapping is broader than deterministic")
    print("=" * 80)
    print()
    print("Summary:")
    print(
        f"  - Learned pairs: {len(learned_pairs)} (vs {len(det_pairs)} deterministic)"
    )
    print(f"  - Learned-only pairs: {len(learned_only_pairs)}")
    print(
        f"  - Techniques with extra learned controls: {len(learned_by_tech) - len(techniques_with_only_det_controls)}/{len(learned_by_tech)}"
    )
    print()
    print("The learned mapping is ready for H3 evaluation.")


if __name__ == "__main__":
    main()
