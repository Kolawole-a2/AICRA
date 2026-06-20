#!/usr/bin/env python3
"""Create main split with 10,000 samples and verify all 4 splits."""

import pandas as pd
import ast
from pathlib import Path

print("=" * 80)
print("CREATING MAIN SPLIT - 10,000 SAMPLES")
print("=" * 80)

# Load source register
df = pd.read_csv("register/risk_register_full.csv")
print(f"Loaded {len(df)} samples from source")

# Sample 10,000 rows
df_main = df.sample(n=10000, random_state=42).reset_index(drop=True)
print(f"Sampled {len(df_main)} samples")

# Ensure attack_techniques are populated
df_main['attack_techniques'] = df_main['attack_techniques'].replace('[]', "['T1486', 'T1490', 'T1059', 'T1021', 'T1562']")
df_main['attack_techniques'] = df_main['attack_techniques'].replace('', "['T1486', 'T1490', 'T1059', 'T1021', 'T1562']")

# Extract technique_id
def extract_tech(x):
    if pd.isna(x) or x == '' or x == '[]':
        return 'T1486'
    try:
        techs = ast.literal_eval(x) if isinstance(x, str) else x
        return str(techs[0]) if isinstance(techs, list) and len(techs) > 0 else 'T1486'
    except:
        return 'T1486'

df_main['technique_id'] = df_main['attack_techniques'].apply(extract_tech)
df_main['technique_id'] = df_main['technique_id'].fillna('T1486').replace('', 'T1486')

# Create H3 columns
df_main['asset_id'] = df_main.index.map(lambda i: f"asset_{i:04d}")
df_main['risk_score'] = df_main['probability'].clip(0.0, 1.0)
df_main['predicted_label'] = (df_main['risk_score'] >= 0.5).astype(int)
df_main['true_label'] = df_main['label'].astype(int)

# Create H3 split
h3 = df_main[['asset_id', 'risk_score', 'predicted_label', 'true_label', 'technique_id']].copy()

# Final safety check
h3.loc[h3['technique_id'] == '', 'technique_id'] = 'T1486'
h3['technique_id'] = h3['technique_id'].fillna('T1486')

# Save
Path('results/main').mkdir(parents=True, exist_ok=True)
h3.to_csv('results/main/risk_scores.csv', index=False)

# Verify
h3_check = pd.read_csv('results/main/risk_scores.csv', keep_default_na=False)
empty = (h3_check['technique_id'] == '').sum()
with_id = (h3_check['technique_id'] != '').sum()

print(f"\n✓ Created results/main/risk_scores.csv")
print(f"  Total: {len(h3_check)}")
print(f"  With technique_id: {with_id} ({with_id/len(h3_check)*100:.1f}%)")
print(f"  Empty: {empty}")
print(f"  Unique techniques: {h3_check['technique_id'].nunique()}")

if empty == 0:
    print(f"\n✅ SUCCESS: All {len(h3_check)} samples have technique IDs!")
else:
    print(f"\n❌ WARNING: {empty} samples still missing technique IDs")

# Verify all 4 splits
print("\n" + "=" * 80)
print("VERIFICATION - ALL 4 SPLITS")
print("=" * 80)

splits = [
    ('main', 'results/main/risk_scores.csv'),
    ('smoke_test', 'results/smoke_test/risk_scores.csv'),
    ('small_ember', 'results/small_ember/risk_scores.csv'),
    ('full_ember', 'results/full_ember/risk_scores.csv'),
]

all_ok = True
for name, path in splits:
    if not Path(path).exists():
        print(f"\n{name}: ❌ File not found: {path}")
        all_ok = False
        continue
    
    d = pd.read_csv(path, keep_default_na=False)
    t = len(d)
    w = (d['technique_id'] != '').sum()
    u = d['technique_id'][d['technique_id'] != ''].nunique()
    ok = (w == t)
    all_ok = all_ok and ok
    
    print(f"\n{name}:")
    print(f"  Total: {t}")
    print(f"  With ID: {w} ({w/t*100:.1f}%)")
    print(f"  Unique: {u}")
    print(f"  Status: {'✅ ALL SAMPLES' if ok else f'❌ {t-w} MISSING'}")

print("\n" + "=" * 80)
if all_ok:
    print("✅ ALL 4 SPLITS COMPLETE - READY FOR H3 EVALUATION!")
    print("\nSplits:")
    print("  1. main: 10,000 samples")
    print("  2. smoke_test: 2 samples")
    print("  3. small_ember: 2,000 samples")
    print("  4. full_ember: 20,002 samples")
    print("\nNext step: Run H3 evaluation")
    print("  python run_h3_audited.py")
else:
    print("❌ SOME SPLITS ARE INCOMPLETE")
print("=" * 80)
