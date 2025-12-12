"""Check mappings sanity: confirms both mappings exist, no overwrite."""

import hashlib
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

# File paths
DET_PATH = BASE_DIR / "data" / "mappings" / "deterministic_lookup.csv"
LRN_PATH = BASE_DIR / "data" / "mappings" / "learned_mapping.csv"


def sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    """Check mappings sanity."""
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
        print("[WARN] Files are byte-identical. This suggests learned mapping overwrote deterministic.")
    else:
        print("[OK] Files are different on disk.")

    # 3. Schema sanity check
    det_df = pd.read_csv(DET_PATH)
    lrn_df = pd.read_csv(LRN_PATH)

    # Normalize column names
    det_has_tech = "technique_id" in det_df.columns or "attack_id" in det_df.columns
    det_has_ctrl = "control_id" in det_df.columns or "defense_id" in det_df.columns
    lrn_has_tech = "technique_id" in lrn_df.columns
    lrn_has_ctrl = "control_id" in lrn_df.columns

    if not (det_has_tech and det_has_ctrl):
        raise ValueError(
            f"Deterministic mapping missing required columns. "
            f"Found: {list(det_df.columns)}"
        )
    if not (lrn_has_tech and lrn_has_ctrl):
        raise ValueError(
            f"Learned mapping missing required columns. "
            f"Found: {list(lrn_df.columns)}"
        )

    print(f"\n[OK] Both mappings contain required columns")

    # Normalize for comparison
    if "attack_id" in det_df.columns:
        det_df = det_df.rename(columns={"attack_id": "technique_id"})
    if "defense_id" in det_df.columns:
        det_df = det_df.rename(columns={"defense_id": "control_id"})

    print(f"Deterministic pairs: {len(det_df)}")
    print(f"Learned      pairs: {len(lrn_df)}")

    print("\nSanity check completed successfully.")


if __name__ == "__main__":
    main()

