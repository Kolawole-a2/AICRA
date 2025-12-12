# Quick Fix Guide: H3 Identical Results

## The Problem

H3 shows identical results for deterministic and learned mappings because they're too similar.

## Quick Fix (3 Steps)

### Step 1: Diagnose the Overlap

```bash
python scripts/diagnose_mapping_overlap.py
```

This shows:
- **Jaccard similarity**: How similar the mappings are (0-100%)
  - >90% = Too similar, will show identical results
  - 80-90% = Very similar, results will be almost identical
  - <80% = Reasonable diversity
- **EXACT_MATCH count**: How many techniques have identical control mappings

### Step 2: Regenerate with More Diversity

If Jaccard > 80%, regenerate:

```bash
# Start with top_k=4 (increased from default 3)
python scripts/regenerate_diverse_learned_mapping.py --top-k 4
```

If still too similar, try:
```bash
python scripts/regenerate_diverse_learned_mapping.py --top-k 5
# Or even:
python scripts/regenerate_diverse_learned_mapping.py --top-k 6
```

### Step 3: Re-run H3

```bash
python run_h3_evaluation.py
```

H3 will now:
- Show different metrics for deterministic vs learned
- Produce different plots
- Warn you if mappings are still too similar

## What Each Script Does

### `diagnose_mapping_overlap.py`
- Computes Jaccard similarity between mappings
- Shows which techniques have EXACT_MATCH, PARTIAL_OVERLAP, or DISJOINT mappings
- Restricts analysis to techniques in your risk scores
- Saves detailed JSON to `results/H3_diagnostics/mapping_overlap.json`

### `regenerate_diverse_learned_mapping.py`
- Regenerates learned mapping using embeddings
- Uses higher `top_k` (more controls per technique = more diversity)
- Ensures it covers all techniques from deterministic
- Verifies mappings are different before saving

### H3 Evaluation (enhanced)
- Computes Jaccard similarity automatically
- Emits **CRITICAL WARNING** if Jaccard > 90%
- Emits **WARNING** if Jaccard > 80%
- Shows per-technique exact match statistics
- Provides clear instructions on how to fix

## Expected Results

After fixing, you should see:

✅ **Different metrics**:
- `delta_precision` ≠ 0
- `delta_variance_reduction` ≠ 0
- Different coverage/DAC values

✅ **Different plots**: Bars don't overlap

✅ **Jaccard < 80%**: Mappings have reasonable diversity

## If Still Having Issues

1. **Check which techniques are in risk scores**:
   - Look at `results/H3_diagnostics/mapping_overlap.json`
   - See which techniques have EXACT_MATCH
   - If risk scores only contain EXACT_MATCH techniques, that's the problem

2. **Increase top_k further**:
   - Try top_k=5, 6, or even 7
   - More controls per technique = more diversity

3. **Check H3 evaluation logs**:
   - Look for warnings about high Jaccard similarity
   - Follow the instructions in the warnings

## Files Modified

- ✅ `scripts/diagnose_mapping_overlap.py` - NEW: Diagnostic tool
- ✅ `scripts/regenerate_diverse_learned_mapping.py` - NEW: Regeneration tool
- ✅ `aicra/experiments/h3_evaluation.py` - MODIFIED: Added Jaccard computation and strong warnings

## Summary

1. **Diagnose**: `python scripts/diagnose_mapping_overlap.py`
2. **Fix**: `python scripts/regenerate_diverse_learned_mapping.py --top-k 4`
3. **Verify**: Re-run diagnostic, then re-run H3

This ensures H3 can meaningfully compare the mappings and show which is better.
