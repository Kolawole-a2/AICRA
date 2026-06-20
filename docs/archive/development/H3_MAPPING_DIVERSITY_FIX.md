> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# H3 Mapping Diversity Fix

## Problem

H3 experiment produces almost identical results for deterministic and learned mappings:
- Same coverage, DAC, pairs_count
- Identical actionable precision and variance reduction
- Cannot determine which mapping is better

## Solution

Three components have been created to fix this:

### 1. Diagnostic Script: `scripts/diagnose_mapping_overlap.py`

**Purpose**: Diagnose overlap between deterministic and learned mappings

**What it does**:
- Loads deterministic and learned mappings
- Normalizes to `(technique_id, set(control_id))` format
- Computes global Jaccard similarity: `J = |Det ∩ Learn| / |Det ∪ Learn|`
- Per-technique overlap classification:
  - **EXACT_MATCH**: Controls are identical
  - **PARTIAL_OVERLAP**: Some overlap but not identical
  - **DISJOINT**: No overlap
- Restricts analysis to techniques present in risk scores (from `config/h3_splits.yaml`)
- Generates human-readable report
- Saves JSON summary to `results/H3_diagnostics/mapping_overlap.json`

**Usage**:
```bash
python scripts/diagnose_mapping_overlap.py
```

**Output**:
- Console report showing:
  - Global Jaccard similarity
  - Count of EXACT_MATCH, PARTIAL_OVERLAP, DISJOINT techniques
  - Sample techniques with their control sets
- JSON file with complete diagnostic data

### 2. Regeneration Script: `scripts/regenerate_diverse_learned_mapping.py`

**Purpose**: Regenerate learned mapping with increased diversity (higher top_k)

**What it does**:
- Loads deterministic mapping
- Generates learned mapping using embeddings with increased `top_k` (default: 4, increased from 3)
- Ensures coverage of all techniques from deterministic
- Verifies mappings are different
- Saves to `data/mappings/learned_mapping.csv`

**Usage**:
```bash
# Default: top_k=4
python scripts/regenerate_diverse_learned_mapping.py

# Custom top_k for even more diversity
python scripts/regenerate_diverse_learned_mapping.py --top-k 5

# Or top_k=6 for maximum diversity
python scripts/regenerate_diverse_learned_mapping.py --top-k 6
```

**Options**:
- `--top-k`: Number of top controls per technique (default: 4)
- `--model`: Sentence transformer model (default: sentence-transformers/all-MiniLM-L6-v2)
- `--deterministic`: Path to deterministic mapping (default: data/mappings/deterministic_lookup.csv)
- `--output`: Path to output learned mapping (default: data/mappings/learned_mapping.csv)

### 3. Enhanced H3 Evaluation Warnings

**What was added**:
- **Jaccard similarity computation**: Now computes and logs Jaccard similarity between mappings
- **Strong warnings** when mappings are too similar:
  - **Jaccard > 90%**: CRITICAL WARNING (error-level logging)
    - Warns that H3 will show almost identical results
    - Provides clear solution: regenerate with increased top_k
  - **Jaccard > 80%**: WARNING
    - Warns that results may be very similar
    - Suggests regeneration
- **Per-technique exact match detection**:
  - Counts techniques with EXACT_MATCH control mappings
  - Warns if >80% of common techniques have identical mappings
  - Explains that if risk scores only contain these techniques, results will be identical

**Location**: `aicra/experiments/h3_evaluation.py` (lines ~1040-1080)

## Workflow

### Step 1: Diagnose Current Overlap

```bash
python scripts/diagnose_mapping_overlap.py
```

This will show:
- Global Jaccard similarity
- How many techniques have EXACT_MATCH, PARTIAL_OVERLAP, DISJOINT mappings
- Sample techniques with their control sets

**Interpretation**:
- Jaccard > 90%: Mappings are too similar, H3 will show identical results
- Jaccard 80-90%: Mappings are very similar, results will be very similar
- Jaccard < 80%: Mappings have reasonable diversity

### Step 2: Regenerate with Increased Diversity

If Jaccard is too high (>80%), regenerate:

```bash
# Start with top_k=4
python scripts/regenerate_diverse_learned_mapping.py --top-k 4

# If still too similar, try top_k=5
python scripts/regenerate_diverse_learned_mapping.py --top-k 5

# Or even top_k=6 for maximum diversity
python scripts/regenerate_diverse_learned_mapping.py --top-k 6
```

### Step 3: Verify Improvement

Re-run diagnostic:

```bash
python scripts/diagnose_mapping_overlap.py
```

Check that:
- Jaccard similarity has decreased
- Fewer techniques have EXACT_MATCH
- More techniques have PARTIAL_OVERLAP or DISJOINT

### Step 4: Re-run H3 Evaluation

```bash
python run_h3_evaluation.py
```

The H3 evaluation will now:
- Compute and log Jaccard similarity
- Emit strong warnings if mappings are too similar (>90% Jaccard)
- Show per-technique exact match statistics
- Produce results that show meaningful differences between mappings

## Expected Results After Fix

After regenerating with increased top_k, you should see:

1. **Lower Jaccard similarity**: < 80% (ideally < 70%)
2. **Fewer EXACT_MATCH techniques**: < 50% of common techniques
3. **Different H3 metrics**:
   - `delta_precision` ≠ 0
   - `delta_variance_reduction` ≠ 0
   - Different coverage/DAC values
4. **Different plots**: Bars and lines should not overlap

## Troubleshooting

### Issue: Jaccard still > 90% after regeneration

**Possible causes**:
1. Embedding model is producing very similar similarity scores
2. Deterministic pairs happen to match top-k embedding similarities
3. Risk scores only contain techniques with identical mappings

**Solutions**:
1. Increase top_k further (try 5, 6, or even 7)
2. Check diagnostic output to see which techniques have EXACT_MATCH
3. Verify risk scores contain techniques with different mappings
4. Consider using a different embedding model

### Issue: H3 still shows identical results

**Possible causes**:
1. Risk scores only contain techniques with EXACT_MATCH mappings
2. Mappings were regenerated but H3 is using cached/old results
3. H3 evaluation code has a bug

**Solutions**:
1. Check diagnostic output for techniques in risk scores
2. Delete old H3 result folders and re-run
3. Verify learned_mapping.csv was actually updated
4. Check H3 evaluation logs for warnings

### Issue: Regeneration fails

**Possible causes**:
1. Missing dependencies (sentence-transformers)
2. Deterministic mapping file not found
3. Embedding model download failed

**Solutions**:
1. Install dependencies: `pip install sentence-transformers`
2. Verify deterministic mapping path
3. Check internet connection for model download
4. Try a different model name

## Files Created/Modified

### New Files
1. `scripts/diagnose_mapping_overlap.py` - Diagnostic script
2. `scripts/regenerate_diverse_learned_mapping.py` - Regeneration script

### Modified Files
1. `aicra/experiments/h3_evaluation.py` - Added Jaccard computation and strong warnings

### Output Files
1. `results/H3_diagnostics/mapping_overlap.json` - Diagnostic results (created by diagnostic script)
2. `data/mappings/learned_mapping.csv` - Regenerated learned mapping (overwritten by regeneration script)

## Summary

The three components work together:
1. **Diagnostic** identifies the problem (high overlap)
2. **Regeneration** fixes it (increases diversity)
3. **H3 warnings** prevent future issues (detects and warns about high similarity)

This ensures H3 can meaningfully compare deterministic vs learned mappings and show which is better.
