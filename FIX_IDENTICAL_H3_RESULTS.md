# Fix: H3 Producing Identical Results for Deterministic and Learned Mappings

## Problem

You're seeing identical plots and metrics for both deterministic and learned mappings in H3 results. This makes it impossible to determine which is better.

## Root Causes

There are several possible reasons why this happens:

### 1. Mappings Are Actually Identical
- The `learned_mapping.csv` file contains exactly the same pairs as `deterministic_lookup.csv`
- **Solution**: Regenerate the learned mapping

### 2. Mappings Cover Different Techniques
- Deterministic mapping covers techniques A, B, C
- Learned mapping covers techniques X, Y, Z
- Risk scores only contain techniques that are in BOTH (or neither)
- **Solution**: Ensure learned mapping covers the same techniques as deterministic

### 3. Techniques Have Identical Control Mappings
- Both mappings cover the same techniques
- But for each technique, they map to the same controls
- **Solution**: Regenerate learned mapping with different parameters (e.g., top_k=4 or 5)

### 4. Risk Scores Only Contain Techniques with Identical Mappings
- Risk scores only contain techniques where both mappings are identical
- Techniques with different mappings are not in the risk scores
- **Solution**: Use risk scores that include techniques with different mappings

## How to Diagnose

Run the diagnostic script:

```bash
python scripts/diagnose_h3_identical_results.py [path/to/risk_scores.csv]
```

This will tell you:
- If mappings are identical
- Which techniques are covered by each mapping
- Which techniques have identical vs different control mappings
- What techniques are in your risk scores

## How to Fix

### Step 1: Fix the Learned Mapping

Run the fix script to ensure learned mapping:
- Covers all techniques from deterministic mapping
- Uses DIFFERENT controls than deterministic

```bash
python scripts/fix_learned_mapping_for_h3.py
```

This script will:
1. Check if learned mapping covers all deterministic techniques
2. Regenerate it using embeddings if needed
3. Verify mappings are different
4. Save to `data/mappings/learned_mapping.csv`

### Step 2: Verify Mappings Are Different

Check the mappings manually:

```bash
python -c "
import pandas as pd
det = pd.read_csv('data/mappings/deterministic_lookup.csv')
lrn = pd.read_csv('data/mappings/learned_mapping.csv')

# Normalize columns
det_col = 'attack_id' if 'attack_id' in det.columns else 'technique_id'
det_ctrl = 'defense_id' if 'defense_id' in det.columns else 'control_id'

det_pairs = set(zip(det[det_col], det[det_ctrl]))
lrn_pairs = set(zip(lrn['technique_id'], lrn['control_id']))

print(f'Deterministic: {len(det_pairs)} pairs')
print(f'Learned: {len(lrn_pairs)} pairs')
print(f'Intersection: {len(det_pairs & lrn_pairs)}')
print(f'Only in det: {len(det_pairs - lrn_pairs)}')
print(f'Only in learned: {len(lrn_pairs - det_pairs)}')
print(f'Are identical? {det_pairs == lrn_pairs}')
"
```

**Expected output:**
- Mappings should NOT be identical (last line should be `False`)
- Should have pairs "only in det" and "only in learned"
- Intersection should be < 100% of deterministic pairs

### Step 3: Regenerate with More Diversity (if needed)

If mappings are still too similar, regenerate with different parameters:

```python
from aicra.mappings.embedding_learned_mapping import build_learned_embedding_mapping
from pathlib import Path

build_learned_embedding_mapping(
    deterministic_path=Path("data/mappings/deterministic_lookup.csv"),
    output_dir=Path("data/mappings"),
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    top_k=4,  # Increase from 3 to 4 or 5 for more diversity
)
```

Then normalize the output:

```python
import pandas as pd
df = pd.read_csv("data/mappings/learned_embedding_attack_defense_mapping.csv")
df = df.rename(columns={"attack_id": "technique_id", "defense_id": "control_id"})
df[["technique_id", "control_id", "similarity_score"]].to_csv(
    "data/mappings/learned_mapping.csv", index=False
)
```

### Step 4: Re-run H3 Evaluation

After fixing the learned mapping:

```bash
python run_h3_evaluation.py
```

The H3 evaluation will now:
- Detect if mappings are identical (and fail with clear error)
- Show different metrics for deterministic vs learned
- Produce different plots

## Verification

After fixing, you should see:

1. **Different metrics** in `results/H3_full_evaluation/H3_full_results.json`:
   - `delta_precision` ≠ 0
   - `delta_variance_reduction` ≠ 0
   - Different values for deterministic vs learned

2. **Different plots** in `results/H3_full_evaluation/plots/`:
   - Bars should have different heights
   - Lines should not overlap

3. **Validation logs** showing:
   - Mappings are different
   - Overlap percentage < 100%
   - Sample pairs that are unique to each mapping

## Common Issues

### Issue: "Mappings are identical" error
**Cause**: Learned mapping file was copied from deterministic
**Fix**: Run `python scripts/fix_learned_mapping_for_h3.py`

### Issue: Still getting identical results after fix
**Cause**: Risk scores only contain techniques with identical mappings
**Fix**: 
1. Check which techniques are in risk scores: `python scripts/diagnose_h3_identical_results.py results/time_test/risk_scores.csv`
2. Ensure risk scores include techniques where mappings differ
3. Or regenerate learned mapping with top_k=4 or 5 for more diversity

### Issue: Learned mapping doesn't cover all techniques
**Cause**: Embedding generation only covered subset of techniques
**Fix**: The fix script should handle this automatically, but if not, manually regenerate

## Quick Fix Command

If you just want to fix it quickly:

```bash
# 1. Fix learned mapping
python scripts/fix_learned_mapping_for_h3.py

# 2. Verify it's different
python scripts/diagnose_h3_identical_results.py

# 3. Re-run H3
python run_h3_evaluation.py
```

## Expected Results After Fix

You should see in H3 results:

- **Different coverage %**: May be similar but not identical
- **Different DAC %**: Should show difference in consistency
- **Different precision**: `delta_precision` should be non-zero
- **Different variance reduction**: `delta_variance_reduction` should be non-zero
- **Statistical tests**: Should be computable (p-values may or may not be significant)

The deterministic mapping should generally perform better (higher precision, higher DAC) if it's the authoritative curated mapping, but the learned mapping should show measurable differences.
