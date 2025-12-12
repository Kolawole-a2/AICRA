"""Compare deterministic and learned mappings with comprehensive summary."""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

# File paths - updated to match AICRA structure
DET_PATH = BASE_DIR / "data" / "mappings" / "deterministic_lookup.csv"
LRN_PATH = BASE_DIR / "data" / "mappings" / "learned_mapping.csv"


def load_mapping(path: Path, name: str, normalize_cols: bool = False) -> pd.DataFrame:
    """Load mapping file and normalize column names if needed."""
    if not path.exists():
        raise FileNotFoundError(f"{name} mapping not found: {path}")
    
    df = pd.read_csv(path)
    
    # Normalize column names
    if normalize_cols:
        if "attack_id" in df.columns:
            df = df.rename(columns={"attack_id": "technique_id"})
        if "defense_id" in df.columns:
            df = df.rename(columns={"defense_id": "control_id"})
    
    return df


def main():
    """Generate comprehensive comparison summary."""
    print("=" * 70)
    print("MAPPING COMPARISON SUMMARY")
    print("=" * 70)
    
    # Load mappings
    det_df = load_mapping(DET_PATH, "Deterministic", normalize_cols=True)
    lrn_df = load_mapping(LRN_PATH, "Learned", normalize_cols=False)
    
    # Extract pairs
    det_pairs = set(zip(det_df["technique_id"], det_df["control_id"]))
    lrn_pairs = set(zip(lrn_df["technique_id"], lrn_df["control_id"]))
    
    # Technique and control sets
    det_techs = set(det_df["technique_id"].unique())
    lrn_techs = set(lrn_df["technique_id"].unique())
    det_ctrls = set(det_df["control_id"].unique())
    lrn_ctrls = set(lrn_df["control_id"].unique())
    
    # Overlaps
    tech_overlap = det_techs & lrn_techs
    ctrl_overlap = det_ctrls & lrn_ctrls
    pair_overlap = det_pairs & lrn_pairs
    
    # Statistics
    print("\n1. BASIC STATISTICS")
    print("-" * 70)
    print(f"{'Metric':<30} {'Deterministic':<20} {'Learned':<20}")
    print("-" * 70)
    print(f"{'Total pairs':<30} {len(det_pairs):<20} {len(lrn_pairs):<20}")
    print(f"{'Unique techniques':<30} {len(det_techs):<20} {len(lrn_techs):<20}")
    print(f"{'Unique controls':<30} {len(det_ctrls):<20} {len(lrn_ctrls):<20}")
    print(f"{'Avg pairs per technique':<30} {len(det_pairs)/len(det_techs):<20.2f} {len(lrn_pairs)/len(lrn_techs):<20.2f}")
    
    # Coverage
    print("\n2. TECHNIQUE COVERAGE")
    print("-" * 70)
    print(f"Deterministic techniques: {len(det_techs)}")
    print(f"Learned techniques:      {len(lrn_techs)}")
    print(f"Overlap:                 {len(tech_overlap)} ({len(tech_overlap)/len(det_techs)*100:.1f}%)")
    
    if len(tech_overlap) == len(det_techs) == len(lrn_techs):
        print("✓ Perfect technique coverage match")
    elif len(tech_overlap) == len(det_techs):
        print("✓ All deterministic techniques covered in learned")
    else:
        missing = det_techs - lrn_techs
        print(f"⚠ Missing {len(missing)} techniques in learned: {sorted(missing)[:10]}")
    
    # Control overlap
    print("\n3. CONTROL OVERLAP")
    print("-" * 70)
    print(f"Deterministic controls: {len(det_ctrls)}")
    print(f"Learned controls:      {len(lrn_ctrls)}")
    print(f"Overlap:               {len(ctrl_overlap)} ({len(ctrl_overlap)/len(det_ctrls)*100:.1f}% of deterministic)")
    
    det_only_ctrls = det_ctrls - lrn_ctrls
    lrn_only_ctrls = lrn_ctrls - det_ctrls
    
    if det_only_ctrls:
        print(f"  Controls only in deterministic: {sorted(det_only_ctrls)}")
    if lrn_only_ctrls:
        print(f"  Controls only in learned: {len(lrn_only_ctrls)} (showing first 10: {sorted(lrn_only_ctrls)[:10]})")
    
    # Pair comparison
    print("\n4. MAPPING PAIR COMPARISON")
    print("-" * 70)
    print(f"Deterministic unique pairs: {len(det_pairs)}")
    print(f"Learned unique pairs:       {len(lrn_pairs)}")
    print(f"Overlapping pairs:          {len(pair_overlap)} ({len(pair_overlap)/len(det_pairs)*100:.1f}% of deterministic)")
    
    det_only_pairs = det_pairs - lrn_pairs
    lrn_only_pairs = lrn_pairs - det_pairs
    
    print(f"Pairs only in deterministic: {len(det_only_pairs)}")
    print(f"Pairs only in learned:       {len(lrn_only_pairs)}")
    
    if len(pair_overlap) == 0:
        print("✓ Mappings are completely different (0% overlap)")
    elif len(pair_overlap) < len(det_pairs) * 0.1:
        print(f"✓ Mappings are sufficiently different ({len(pair_overlap)/len(det_pairs)*100:.1f}% overlap)")
    else:
        print(f"⚠ Mappings have significant overlap ({len(pair_overlap)/len(det_pairs)*100:.1f}%)")
    
    # DAC-style metrics (using deterministic as reference)
    print("\n5. DAC METRICS (Deterministic as Reference)")
    print("-" * 70)
    det_dac = (len(pair_overlap) / len(det_pairs) * 100.0) if det_pairs else 0.0
    lrn_dac = (len(pair_overlap) / len(lrn_pairs) * 100.0) if lrn_pairs else 0.0
    
    print(f"Deterministic self-consistency: {det_dac:.2f}% ({len(pair_overlap)} / {len(det_pairs)})")
    print(f"Learned consistency vs deterministic: {lrn_dac:.2f}% ({len(pair_overlap)} / {len(lrn_pairs)})")
    
    # Technique-level analysis
    print("\n6. TECHNIQUE-LEVEL ANALYSIS")
    print("-" * 70)
    
    # Count mappings per technique
    det_tech_counts = det_df.groupby("technique_id").size()
    lrn_tech_counts = lrn_df.groupby("technique_id").size()
    
    print(f"Techniques with mappings:")
    print(f"  Deterministic: {len(det_tech_counts)} techniques")
    print(f"  Learned:       {len(lrn_tech_counts)} techniques")
    
    # Techniques with most mappings
    print(f"\nTop 5 techniques by mapping count (Deterministic):")
    for tech, count in det_tech_counts.nlargest(5).items():
        lrn_count = lrn_tech_counts.get(tech, 0)
        print(f"  {tech}: {count} mappings (learned: {lrn_count})")
    
    print(f"\nTop 5 techniques by mapping count (Learned):")
    for tech, count in lrn_tech_counts.nlargest(5).items():
        det_count = det_tech_counts.get(tech, 0)
        print(f"  {tech}: {count} mappings (deterministic: {det_count})")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    checks = []
    checks.append(("Technique coverage", len(tech_overlap) == len(det_techs)))
    checks.append(("Similar row count", abs(len(det_pairs) - len(lrn_pairs)) <= 20))
    checks.append(("Different mappings", len(pair_overlap) < len(det_pairs) * 0.1))
    
    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

