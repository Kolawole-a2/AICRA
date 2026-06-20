#!/usr/bin/env python3
"""Fix register files by directly updating attack_techniques column."""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from aicra.config import get_settings
from aicra.pipelines.mapping import MappingPipeline

def fix_register(register_path: Path):
    """Fix register file by populating attack_techniques."""
    print(f"Loading {register_path}...")
    df = pd.read_csv(register_path)
    print(f"  Loaded {len(df)} records")
    
    # Initialize mapping pipeline
    settings = get_settings()
    mp = MappingPipeline(settings, skip_mlflow=True)
    
    # Default techniques for Unknown/empty families
    default_techs = ["T1486", "T1490", "T1059", "T1021", "T1562"]
    
    def get_techs(family):
        if pd.isna(family) or str(family).lower() in ['unknown', '']:
            return default_techs
        canonical = mp.normalize_family(str(family))
        techs = mp.family_to_attack(canonical)
        return techs if techs else default_techs
    
    # Update attack_techniques
    print("  Updating attack_techniques...")
    df['attack_techniques'] = df['family'].apply(get_techs)
    df['attack_techniques'] = df['attack_techniques'].apply(str)
    
    # Verify
    n_with_tech = df['attack_techniques'].apply(lambda x: x != '[]' and str(x) != '[]').sum()
    print(f"  Records with techniques: {n_with_tech}/{len(df)}")
    print(f"  Sample: {df['attack_techniques'].iloc[0]}")
    
    # Save
    print(f"  Saving to {register_path}...")
    df.to_csv(register_path, index=False)
    print(f"  ✓ Saved")
    
    return df

if __name__ == "__main__":
    register_path = Path("register/risk_register_full.csv")
    if register_path.exists():
        fix_register(register_path)
        print("\n✓ Register fixed! Now run: python create_ember_splits.py")
    else:
        print(f"Register file not found: {register_path}")

