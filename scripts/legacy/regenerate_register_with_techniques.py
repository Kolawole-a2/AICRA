#!/usr/bin/env python3
"""
Regenerate register files with proper attack_techniques extraction.

This script reads existing register files and enriches them with attack_techniques
by mapping from family to ATT&CK techniques using the MappingPipeline.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from aicra.config import get_settings
from aicra.pipelines.mapping import MappingPipeline

def regenerate_register(register_path: Path, output_path: Path = None) -> pd.DataFrame:
    """
    Regenerate register file with proper attack_techniques.
    
    Args:
        register_path: Path to existing register CSV
        output_path: Path to save regenerated register (default: overwrite original)
        
    Returns:
        DataFrame with enriched attack_techniques
    """
    print(f"Loading register: {register_path}")
    df = pd.read_csv(register_path)
    print(f"  Loaded {len(df)} records")
    
    # Check if attack_techniques column exists
    if "attack_techniques" not in df.columns:
        print("  Adding attack_techniques column...")
        df["attack_techniques"] = None
    
    # Initialize mapping pipeline
    settings = get_settings()
    mapping_pipeline = MappingPipeline(settings, skip_mlflow=True)
    
    # Map families to attack techniques
    print("  Mapping families to ATT&CK techniques...")
    
    def get_techniques_for_family(family):
        """Get attack techniques for a family."""
        if pd.isna(family) or family == "":
            # Default techniques for unknown/empty families (common ransomware techniques)
            return ["T1486", "T1490", "T1059"]  # Data Encrypted, Inhibit Recovery, Command Scripting
        
        # Normalize family and get techniques
        canonical_family = mapping_pipeline.normalize_family(str(family))
        techniques = mapping_pipeline.family_to_attack(canonical_family)
        
        # If no techniques found, use default ransomware techniques
        if not techniques:
            if canonical_family == "Unknown":
                # For Unknown families, use common ransomware techniques
                techniques = ["T1486", "T1490", "T1059", "T1021", "T1562"]
            else:
                # For other unmapped families, use a smaller default set
                techniques = ["T1486", "T1490", "T1059"]
        
        return techniques
    
    # Apply mapping
    print("  Applying family to technique mapping...")
    df["attack_techniques"] = df["family"].apply(get_techniques_for_family)
    
    # Convert list to string representation for CSV (preserve list format)
    # The CSV expects string representation like "['T1486', 'T1490']"
    df["attack_techniques"] = df["attack_techniques"].apply(
        lambda x: str(x) if isinstance(x, list) else (str([]) if pd.isna(x) else str(x))
    )
    
    # Count statistics
    n_with_techniques = df["attack_techniques"].apply(lambda x: x != '[]' and x != str([])).sum()
    n_total = len(df)
    print(f"  Mapped {n_with_techniques}/{n_total} records to techniques ({n_with_techniques/n_total*100:.1f}%)")
    
    # Show sample of families and their techniques
    print("\n  Sample mappings:")
    sample_df = df[df["attack_techniques"].apply(lambda x: x != '[]' and x != str([]))].head(5)
    for idx, row in sample_df.iterrows():
        print(f"    {row['family']} -> {row['attack_techniques']}")
    
    # Save regenerated register
    output_path = output_path or register_path
    print(f"\n  Saving to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(df)} records")
    
    return df

def main():
    """Regenerate all register files."""
    register_dir = Path("register")
    
    registers_to_regenerate = [
        "risk_register_full.csv",
        "risk_register_small_ember.csv",
        "smoke_test_register.csv",
    ]
    
    print("=" * 80)
    print("REGENERATING REGISTER FILES WITH ATTACK TECHNIQUES")
    print("=" * 80)
    
    for register_name in registers_to_regenerate:
        register_path = register_dir / register_name
        if not register_path.exists():
            print(f"\n⚠️  Skipping {register_name}: file not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing: {register_name}")
        print(f"{'='*80}")
        
        try:
            regenerate_register(register_path)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("REGENERATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Regenerate H3 splits: python create_ember_splits.py")
    print("  2. Re-run H3 evaluation: python run_h3_audited.py")

if __name__ == "__main__":
    main()

