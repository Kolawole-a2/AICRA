"""Compute DAC (Defense-Attack Consistency) metrics for deterministic and learned mappings."""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Actual file paths based on project structure
DET_PATH = BASE_DIR / "data" / "mappings" / "deterministic_attack_defense_lookup.csv"
LRN_PATH = BASE_DIR / "data" / "mappings" / "heuristic" / "learned_mapping.csv"
REF_PATH = BASE_DIR / "data" / "ontology" / "d3fend_reference_pairs.csv"


def load_pairs(path: Path, name: str, normalize_cols: bool = False) -> pd.DataFrame:
    """Load mapping pairs from CSV, handling different column names."""
    if not path.exists():
        raise FileNotFoundError(f"{name} mapping not found: {path}")
    
    df = pd.read_csv(path)
    
    # Normalize column names if needed
    if normalize_cols:
        # Deterministic uses attack_id/defense_id, normalize to technique_id/control_id
        if "attack_id" in df.columns:
            df = df.rename(columns={"attack_id": "technique_id"})
        if "defense_id" in df.columns:
            df = df.rename(columns={"defense_id": "control_id"})
    
    # Check for required columns
    if not {"technique_id", "control_id"}.issubset(df.columns):
        raise ValueError(
            f"{name} mapping missing technique_id/control_id. "
            f"Found columns: {list(df.columns)}"
        )
    
    # Extract pairs
    df = df[["technique_id", "control_id"]].dropna().drop_duplicates()
    return df


def main():
    """Compute DAC metrics."""
    # Load deterministic mapping (normalize column names)
    det = load_pairs(DET_PATH, "Deterministic", normalize_cols=True)
    
    # Load learned mapping
    lrn = load_pairs(LRN_PATH, "Learned", normalize_cols=False)
    
    # Load reference pairs (use deterministic as fallback if not found)
    if REF_PATH.exists():
        ref = load_pairs(REF_PATH, "Reference", normalize_cols=False)
        print(f"[OK] Using reference pairs from: {REF_PATH}")
    else:
        # Use deterministic mapping as reference (gold standard)
        ref = det.copy()
        print(f"[INFO] Reference file not found at {REF_PATH}")
        print(f"[INFO] Using deterministic mapping as reference (gold standard)")
    
    # Convert to sets of tuples for comparison
    det_pairs = set(map(tuple, det.values.tolist()))
    lrn_pairs = set(map(tuple, lrn.values.tolist()))
    ref_pairs = set(map(tuple, ref.values.tolist()))
    
    def coverage(pairs, label):
        """Print coverage statistics."""
        techs = {t for t, _ in pairs}
        ctrls = {c for _, c in pairs}
        print(f"\n{label} mapping:")
        print(f"  Pairs:          {len(pairs)}")
        print(f"  Techniques:    {len(techs)}")
        print(f"  Controls:       {len(ctrls)}")
    
    coverage(det_pairs, "Deterministic")
    coverage(lrn_pairs, "Learned")
    coverage(ref_pairs, "Reference")
    
    # DAC-style consistency summary
    det_inter = det_pairs & ref_pairs
    lrn_inter = lrn_pairs & ref_pairs
    
    def dac(pairs, inter, label):
        """Compute and print DAC metric."""
        dac_val = (len(inter) / len(pairs) * 100.0) if pairs else 0.0
        print(f"\n{label} DAC vs reference: {dac_val:.2f}% ({len(inter)} / {len(pairs)})")
        if pairs:
            missing = pairs - inter
            if missing:
                print(f"  Missing from reference: {len(missing)} pairs")
    
    print("\n" + "=" * 60)
    print("DAC METRICS")
    print("=" * 60)
    dac(det_pairs, det_inter, "Deterministic")
    dac(lrn_pairs, lrn_inter, "Learned")
    
    # Additional comparison: learned vs deterministic
    lrn_det_inter = lrn_pairs & det_pairs
    print(f"\nLearned vs Deterministic overlap: {len(lrn_det_inter)} pairs "
          f"({len(lrn_det_inter)/len(lrn_pairs)*100:.2f}% of learned, "
          f"{len(lrn_det_inter)/len(det_pairs)*100:.2f}% of deterministic)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

