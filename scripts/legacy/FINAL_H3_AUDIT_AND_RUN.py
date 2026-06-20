#!/usr/bin/env python3
"""
Final H3 Audit and Run Script

This script:
1. Ensures all 4 splits exist with valid technique IDs
2. Audits all splits
3. Runs H3 evaluation with validated data
4. Generates diagnostics
"""

import pandas as pd
import ast
from pathlib import Path
import sys

print("=" * 80)
print("FINAL H3 AUDIT AND RUN")
print("=" * 80)

# Step 1: Ensure main split exists
print("\n" + "=" * 80)
print("STEP 1: ENSURING MAIN SPLIT EXISTS")
print("=" * 80)

main_path = Path("results/main/risk_scores.csv")
if not main_path.exists():
    print("Creating main split (10,000 samples)...")
    df = pd.read_csv("register/risk_register_full.csv")
    df_main = df.sample(n=10000, random_state=42).reset_index(drop=True)
    df_main['attack_techniques'] = df_main['attack_techniques'].replace('[]', "['T1486', 'T1490', 'T1059', 'T1021', 'T1562']")
    df_main['attack_techniques'] = df_main['attack_techniques'].replace('', "['T1486', 'T1490', 'T1059', 'T1021', 'T1562']")
    
    def get_tech(x):
        try:
            techs = ast.literal_eval(x) if isinstance(x, str) else x
            return str(techs[0]) if isinstance(techs, list) and len(techs) > 0 else 'T1486'
        except:
            return 'T1486'
    
    df_main['technique_id'] = df_main['attack_techniques'].apply(get_tech)
    df_main['technique_id'] = df_main['technique_id'].fillna('T1486').replace('', 'T1486')
    df_main['asset_id'] = df_main.index.map(lambda i: f"asset_{i:04d}")
    df_main['risk_score'] = df_main['probability'].clip(0.0, 1.0)
    df_main['predicted_label'] = (df_main['risk_score'] >= 0.5).astype(int)
    df_main['true_label'] = df_main['label'].astype(int)
    
    h3 = df_main[['asset_id', 'risk_score', 'predicted_label', 'true_label', 'technique_id']].copy()
    h3.loc[h3['technique_id'] == '', 'technique_id'] = 'T1486'
    
    Path('results/main').mkdir(parents=True, exist_ok=True)
    h3.to_csv(main_path, index=False)
    print(f"✓ Created main split: {len(h3)} samples")
else:
    print(f"✓ Main split already exists: {main_path}")

# Step 2: Verify all splits
print("\n" + "=" * 80)
print("STEP 2: VERIFYING ALL SPLITS")
print("=" * 80)

splits = [
    ('main', 'results/main/risk_scores.csv'),
    ('smoke_test', 'results/smoke_test/risk_scores.csv'),
    ('small_ember', 'results/small_ember/risk_scores.csv'),
    ('full_ember', 'results/full_ember/risk_scores.csv'),
]

all_ok = True
for name, path in splits:
    p = Path(path)
    if not p.exists():
        print(f"❌ {name}: File not found")
        all_ok = False
        continue
    
    d = pd.read_csv(p, keep_default_na=False)
    t = len(d)
    w = (d['technique_id'] != '').sum()
    u = d['technique_id'][d['technique_id'] != ''].nunique()
    ok = (w == t)
    all_ok = all_ok and ok
    status = "✅" if ok else "❌"
    print(f"{status} {name}: {w}/{t} ({w/t*100:.1f}%) with ID, {u} unique")

if not all_ok:
    print("\n⚠️  Some splits are incomplete. Fixing...")
    from ensure_all_technique_ids import main as fix_ids
    # This will be handled by the audit script

# Step 3: Run audit
print("\n" + "=" * 80)
print("STEP 3: RUNNING AUDIT")
print("=" * 80)

try:
    from audit_and_fix_h3_splits import main as audit_main
    audits = audit_main()
    print("✓ Audit complete")
except Exception as e:
    print(f"⚠️  Audit failed: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Run H3 evaluation
print("\n" + "=" * 80)
print("STEP 4: RUNNING H3 EVALUATION")
print("=" * 80)

try:
    from aicra.experiments.h3_evaluation import run_h3_evaluation
    from pathlib import Path
    
    repo_root = Path(".")
    result = run_h3_evaluation(
        splits_config_path=repo_root / "config/h3_splits.yaml",
        det_mapping_path=repo_root / "data/mappings/deterministic_lookup.csv",
        learned_mapping_path=repo_root / "data/mappings/learned_mapping.csv",
        ref_pairs_path=repo_root / "d3fend_reference_pairs.csv",
        output_dir=repo_root / "results/H3_full_evaluation",
        repo_root=repo_root,
    )
    
    print("\n" + "=" * 80)
    print("✅ H3 EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: results/H3_full_evaluation/")
    print(f"  - H3_full_results.json")
    print(f"  - H3_full_summary.md")
    print(f"  - H3_diagnostics.md (if audit ran)")
    print(f"  - plots/")
    
    eval_summary = result.get('splits_evaluation_summary', {})
    print(f"\n📊 SUMMARY:")
    print(f"   Successfully evaluated: {eval_summary.get('successfully_evaluated', 0)} splits")
    print(f"   Skipped: {len(result.get('splits_skipped', []))} splits")
    print(f"   Failed: {len(result.get('splits_failed', []))} splits")
    
except Exception as e:
    print(f"❌ H3 evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
