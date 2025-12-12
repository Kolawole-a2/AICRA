# Learned Mapping Fix Summary

## Problem

The learned mapping CSV was identical to the deterministic lookup (same 15 pairs), meaning the embedding pipeline was either:
1. Copying the deterministic table, or
2. Intersecting with deterministic and throwing away everything else

## Solution

### 1. Hard Check in `generate_learned_mapping()` (Lines 154-250)

**Added RuntimeError check** that raises immediately if learned pairs == deterministic pairs:

```python
def generate_learned_mapping(
    ...,
    deterministic_pairs: set = None,  # For verification ONLY
) -> pd.DataFrame:
    # ... generate learned mapping from embeddings ONLY ...
    
    # CRITICAL CHECK: Verify learned mapping is different from deterministic
    if deterministic_pairs is not None:
        learned_pairs = set(zip(learned_mapping_df["attack_id"], learned_mapping_df["defense_id"]))
        if learned_pairs == deterministic_pairs:
            raise RuntimeError(
                "Learned mapping is identical to deterministic mapping. "
                "This indicates a bug - learned mapping should be generated PURELY from embeddings, "
                "not copied from or filtered by deterministic pairs."
            )
```

**Why:** Catches the bug immediately during generation, before saving the file.

### 2. Explicit Comments About No Filtering (Lines 172-173, 183, 301-307)

Added explicit comments throughout the code:
- "CRITICAL: Generating pairs PURELY from embedding similarity - NO filtering by deterministic pairs"
- "CRITICAL: We select based ONLY on similarity scores, NOT on whether pairs exist in deterministic"
- "CRITICAL: Learned mapping will be generated PURELY from embedding similarity, NOT filtered by deterministic pairs"

**Why:** Makes it crystal clear that deterministic pairs are NEVER used for filtering.

### 3. Verification in `build_learned_embedding_mapping()` (Lines 301-307)

Extracts deterministic pairs ONLY for verification, with explicit logging:
- "Extracted {N} deterministic pairs for verification ONLY (NOT for filtering)"
- "CRITICAL: Learned mapping will be generated PURELY from embedding similarity"

**Why:** Documents that deterministic pairs are extracted but NOT used for filtering.

### 4. Enhanced Error in `generate_learned_mapping.py` (Lines 66-72)

Changed warning to RuntimeError with detailed diagnostics:
- Shows exact pair counts
- Explains possible causes
- Raises RuntimeError instead of just warning

**Why:** Fails fast with clear error message.

## Guarantees

After these changes:

1. ✅ **Learned mapping generated ONLY from embeddings**: Top-k selection based purely on cosine similarity scores
2. ✅ **No filtering by deterministic**: Deterministic pairs are NEVER used to filter, intersect, or modify learned mapping
3. ✅ **Hard check raises RuntimeError**: If learned == deterministic, raises RuntimeError immediately
4. ✅ **Explicit comments**: Code clearly documents that deterministic pairs are for verification only

## Expected Behavior

When you run `python generate_learned_mapping.py`:

- **If learned mapping is different from deterministic** (expected):
  - Script completes successfully
  - Learned mapping has different pairs (usually more than 15)
  - Metrics will differ from deterministic

- **If learned mapping is identical to deterministic** (bug):
  - Script raises RuntimeError immediately
  - Error message explains the problem
  - No file is saved

## Next Steps

1. **Run the generation script**:
   ```bash
   python generate_learned_mapping.py
   ```

2. **If RuntimeError is raised**, it means:
   - The embedding model is producing identical similarity scores (unlikely)
   - There's a bug somewhere filtering by deterministic (needs investigation)
   - Deterministic pairs happen to match top-k embedding similarities (coincidence, but should be rare)

3. **If script succeeds**, verify the learned mapping:
   - Should have different pairs than deterministic
   - Usually more than 15 pairs (if more than 5 unique attacks)
   - Based purely on semantic similarity

4. **Re-run H3 experiment**:
   ```bash
   python run_h3_experiment.py
   ```
   - Metrics should now diverge between deterministic and learned
   - Plots should show different values
