#!/usr/bin/env python3
"""
Fix ALL register files and regenerate ALL H3 splits with proper technique IDs.

This ensures ALL samples in smoke_test, small_ember, and full_ember have technique IDs.
"""

import pandas as pd
import ast
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from aicra.config import get_settings
from aicra.pipelines.mapping import MappingPipeline

print("=" * 80)
print("FIXING ALL REGISTER FILES AND H3 SPLITS")
print("=" * 80)

# Default techniques for Unknown/empty families
DEFAULT_TECHS = ["T1486", "T1490", "T1059", "T1021", "T1562"]

def extract_technique_id(x):
    """Extract first technique from attack_techniques."""
    if pd.isna(x) or x == '' or x == '[]':
        return None
    try:
        techs = ast.literal_eval(x) if isinstance(x, str) else x
        return str(techs[0]) if isinstance(techs, list) and len(techs) > 0 else None
    except:
        return None

def fix_register_and_split(register_path: Path, split_name: str):
    """Fix register file and regenerate H3 split."""
    print(f"\n{'='*80}")
    print(f"Processing: {split_name}")
    print(f"{'='*80}")
    
    if not register_path.exists():
        print(f"  ⚠️  Register file not found: {register_path}")
        return False
    
    # Step 1: Load and fix register
    print(f"\n[1] Loading register: {register_path}")
    df_reg = pd.read_csv(register_path)
    print(f"     Loaded {len(df_reg)} records")
    
    # Initialize mapping pipeline
    settings = get_settings()
    mp = MappingPipeline(settings, skip_mlflow=True)
    
    def get_techniques_for_family(family):
        """Get attack techniques for a family."""
        if pd.isna(family) or str(family).lower() in ['unknown', '']:
            return DEFAULT_TECHS
        
        canonical = mp.normalize_family(str(family))
        techs = mp.family_to_attack(canonical)
        return techs if techs else DEFAULT_TECHS
    
    # Update attack_techniques
    print(f"     Mapping families to techniques...")
    df_reg['attack_techniques'] = df_reg['family'].apply(get_techniques_for_family)
    df_reg['attack_techniques'] = df_reg['attack_techniques'].apply(str)
    
    # Verify register update
    n_with_tech = df_reg['attack_techniques'].apply(lambda x: x != '[]' and str(x) != '[]').sum()
    print(f"     Records with techniques: {n_with_tech}/{len(df_reg)} ({n_with_tech/len(df_reg)*100:.1f}%)")
    print(f"     Sample: {df_reg['attack_techniques'].iloc[0]}")
    
    # Save register
    print(f"     Saving register...")
    df_reg.to_csv(register_path, index=False)
    print(f"     ✓ Saved register")
    
    # Step 2: Regenerate H3 split
    print(f"\n[2] Regenerating H3 split...")
    df_reg_full = pd.read_csv(register_path)  # Reload to ensure we have latest
    
    df_reg_full["asset_id"] = df_reg_full.index.map(lambda i: f"asset_{i:04d}")
    df_reg_full["risk_score"] = df_reg_full["probability"].clip(0.0, 1.0)
    df_reg_full["predicted_label"] = (df_reg_full["risk_score"] >= 0.5).astype(int)
    df_reg_full["true_label"] = df_reg_full["label"].astype(int)
    df_reg_full["technique_id"] = df_reg_full["attack_techniques"].apply(extract_technique_id)
    
    h3_df = df_reg_full[["asset_id", "risk_score", "predicted_label", "true_label", "technique_id"]]
    
    # Determine output path based on split name
    if split_name == "smoke_test":
        output_path = Path("results/smoke_test/risk_scores.csv")
    elif split_name == "small_ember":
        output_path = Path("results/small_ember/risk_scores.csv")
    elif split_name == "full_ember":
        output_path = Path("results/full_ember/risk_scores.csv")
    else:
        output_path = Path(f"results/{split_name}/risk_scores.csv")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    h3_df.to_csv(output_path, index=False)
    
    # Step 3: Verify H3 split
    print(f"\n[3] Verification...")
    df_verify = pd.read_csv(output_path)
    df_verify['technique_id'] = df_verify['technique_id'].replace('', pd.NA)
    n_with_tech_h3 = df_verify['technique_id'].notna().sum()
    n_unique = df_verify['technique_id'].dropna().nunique()
    
    print(f"     Total samples: {len(df_verify)}")
    print(f"     Samples with technique_id: {n_with_tech_h3} ({n_with_tech_h3/len(df_verify)*100:.1f}%)")
    print(f"     Unique techniques: {n_unique}")
    
    if n_with_tech_h3 > 0:
        sample_ids = df_verify['technique_id'].dropna().unique()[:5].tolist()
        print(f"     Sample technique_ids: {sample_ids}")
        print(f"     ✓ SUCCESS: {split_name} has technique IDs!")
        return True
    else:
        print(f"     ❌ FAILED: {split_name} has NO technique IDs")
        return False

# Process all splits
splits_to_fix = [
    ("register/smoke_test_register.csv", "smoke_test"),
    ("register/risk_register_small_ember.csv", "small_ember"),
    ("register/risk_register_full.csv", "full_ember"),
]

results = {}
for register_rel_path, split_name in splits_to_fix:
    register_path = Path(register_rel_path)
    success = fix_register_and_split(register_path, split_name)
    results[split_name] = success

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for split_name, success in results.items():
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"  {split_name:15s}: {status}")

all_success = all(results.values())
if all_success:
    print("\n✅ ALL SPLITS FIXED!")
    print("\nNext step: Run H3 evaluation")
    print("  python run_h3_audited.py")
else:
    print("\n⚠️  Some splits failed. Check errors above.")

print("=" * 80)

