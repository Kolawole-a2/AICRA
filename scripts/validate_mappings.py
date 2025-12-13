"""Validate deterministic and learned mappings for H3 comparison."""

import hashlib
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Actual file paths based on project structure
DET_PATH = BASE_DIR / "data" / "mappings" / "deterministic_attack_defense_lookup.csv"
LRN_PATH = BASE_DIR / "data" / "mappings" / "heuristic" / "learned_mapping.csv"


def sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    """Validate both mapping files."""
    # 1. Check existence
    if not DET_PATH.exists():
        raise FileNotFoundError(f"Deterministic mapping missing: {DET_PATH}")
    if not LRN_PATH.exists():
        raise FileNotFoundError(f"Learned mapping missing: {LRN_PATH}")

    print(f"[OK] Found deterministic: {DET_PATH}")
    print(f"[OK] Found learned:       {LRN_PATH}")

    # 2. Hashes (ensure they're not identical by accident)
    det_hash = sha256(DET_PATH)
    lrn_hash = sha256(LRN_PATH)

    print(f"\nDeterministic SHA256: {det_hash}")
    print(f"Learned      SHA256: {lrn_hash}")

    if det_hash == lrn_hash:
        print(
            "[WARN] Files are byte-identical. This suggests learned mapping overwrote deterministic."
        )
    else:
        print("[OK] Files are different on disk.")

    # 3. Schema sanity check
    det_df = pd.read_csv(DET_PATH)
    lrn_df = pd.read_csv(LRN_PATH)

    # Deterministic uses attack_id/defense_id, learned uses technique_id/control_id
    det_required = {"attack_id", "defense_id"}
    lrn_required = {"technique_id", "control_id"}

    if not det_required.issubset(det_df.columns):
        raise ValueError(
            f"Deterministic mapping missing required columns: "
            f"{det_required - set(det_df.columns)}"
        )
    if not lrn_required.issubset(lrn_df.columns):
        raise ValueError(
            f"Learned mapping missing required columns: "
            f"{lrn_required - set(lrn_df.columns)}"
        )

    print(f"\n[OK] Deterministic contains {det_required}")
    print(f"[OK] Learned contains {lrn_required}")

    # Normalize column names for comparison
    det_normalized = det_df.rename(
        columns={"attack_id": "technique_id", "defense_id": "control_id"}
    )
    lrn_normalized = lrn_df.copy()

    print(f"\nDeterministic pairs: {len(det_normalized)}")
    print(f"Learned      pairs: {len(lrn_normalized)}")

    # Check technique coverage
    det_techs = set(det_normalized["technique_id"].unique())
    lrn_techs = set(lrn_normalized["technique_id"].unique())

    print(f"\nDeterministic techniques: {len(det_techs)}")
    print(f"Learned      techniques: {len(lrn_techs)}")
    print(
        f"Overlap: {len(det_techs & lrn_techs)} ({len(det_techs & lrn_techs) / len(det_techs) * 100:.1f}%)"
    )

    # Check if mappings are different
    det_pairs = set(
        zip(det_normalized["technique_id"], det_normalized["control_id"], strict=False)
    )
    lrn_pairs = set(
        zip(lrn_normalized["technique_id"], lrn_normalized["control_id"], strict=False)
    )
    overlap_pairs = det_pairs & lrn_pairs

    print(f"\nDeterministic unique pairs: {len(det_pairs)}")
    print(f"Learned      unique pairs: {len(lrn_pairs)}")
    print(
        f"Overlapping pairs: {len(overlap_pairs)} ({len(overlap_pairs) / len(det_pairs) * 100:.1f}% of deterministic)"
    )

    if len(overlap_pairs) == 0:
        print("[OK] Mappings are completely different (0% overlap)")
    elif len(overlap_pairs) < len(det_pairs) * 0.1:
        print(
            f"[OK] Mappings are sufficiently different ({len(overlap_pairs) / len(det_pairs) * 100:.1f}% overlap)"
        )
    else:
        print(
            f"[WARN] Mappings have significant overlap ({len(overlap_pairs) / len(det_pairs) * 100:.1f}%)"
        )

    print("\n" + "=" * 60)
    print("Sanity check completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
