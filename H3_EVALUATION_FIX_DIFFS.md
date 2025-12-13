# H3 Evaluation Fix - Unified Diffs

## Summary

Fixed `run_h3_experiment.py` to ensure deterministic and learned mappings are kept separate and used correctly for all metrics. The key changes ensure:

1. **Separate DataFrames**: Created `det_pairs_df` and `lrn_pairs_df` as separate copies that are never modified
2. **No Intersection Overwrite**: Intersection is computed separately for DAC but never replaces `lrn_pairs_df`
3. **Critical Error Check**: If both "only in deterministic" and "only in learned" are 0, the script exits with error
4. **Consistent Usage**: All metric functions use the correct DataFrame (`det_pairs_df` for deterministic, `lrn_pairs_df` for learned)

## Key Changes

### 1. Separate DataFrame Creation (Lines 214-241)

**BEFORE:**
```python
det = pd.read_csv(IN_DET)
# ... rename columns ...
det = det[["technique_id", "control_id"]].drop_duplicates()

lrn = pd.read_csv(IN_LRN)
# ... rename columns ...
lrn = lrn[["technique_id", "control_id"]].drop_duplicates()

det_pairs_df = det[["technique_id", "control_id"]].drop_duplicates()
lrn_pairs_df = lrn[["technique_id", "control_id"]].drop_duplicates()
```

**AFTER:**
```python
det_raw = pd.read_csv(IN_DET)
# ... rename columns ...

# CRITICAL: Create separate copy for deterministic pairs - DO NOT modify after this
det_pairs_df = det_raw[["technique_id", "control_id"]].drop_duplicates().copy()

lrn_raw = pd.read_csv(IN_LRN)
# ... rename columns ...

# CRITICAL: Create separate copy for learned pairs - DO NOT modify after this
# This is the learned mapping AS-IS, NOT intersected with deterministic or reference
lrn_pairs_df = lrn_raw[["technique_id", "control_id"]].drop_duplicates().copy()
```

**Why:** Using `.copy()` ensures we have independent DataFrames that won't be accidentally modified.

---

### 2. Critical Error Check for Identical Mappings (Lines 261-271)

**BEFORE:**
```python
if det_pairs == lrn_pairs:
    print(f"\n⚠️  WARNING: Mappings are IDENTICAL!")
    # ... continue execution ...
```

**AFTER:**
```python
# CRITICAL CHECK: If both are 0, mappings are identical - this is an error
if len(only_in_det) == 0 and len(only_in_learned) == 0:
    print(f"\n❌ CRITICAL ERROR: Mappings are IDENTICAL!")
    print(f"Both 'only in deterministic' and 'only in learned' are 0.")
    print(f"This means the learned mapping file contains EXACTLY the same pairs as deterministic.")
    print(f"Results will be identical. This indicates a problem with the learned mapping generation.")
    print(f"Deterministic file: {IN_DET} (SHA256: {file_hash(IN_DET)[:16]}...)")
    print(f"Learned file: {IN_LRN} (SHA256: {file_hash(IN_LRN)[:16]}...)")
    print(f"\nSOLUTION: Regenerate the embedding-based learned mapping:")
    print(f"  python -m aicra.mappings.embedding_learned_mapping")
    sys.exit(1)
```

**Why:** If both "only in" sets are 0, the mappings are identical, which is an error condition that should stop execution.

---

### 3. Mapping Metrics - Explicit DataFrame Usage (Lines 326-347)

**BEFORE:**
```python
mm_det = mapping_metrics(det, refp)
mm_lrn = mapping_metrics(lrn, refp)
```

**AFTER:**
```python
# CRITICAL: Use det_pairs_df for deterministic metrics, lrn_pairs_df for learned metrics
# mapping_metrics computes consistency by comparing with refp, but does NOT modify the input DataFrames
# For deterministic: use det_pairs_df
mm_det = mapping_metrics(det_pairs_df, refp)
# For learned: use lrn_pairs_df (NOT intersection, NOT reference, NOT deterministic)
mm_lrn = mapping_metrics(lrn_pairs_df, refp)

# ... logging ...

# Compute intersection separately for DAC (but DO NOT replace learned_pairs_df with it)
intersection_pairs = det_pairs & lrn_pairs
dac_overall = (len(intersection_pairs) / len(det_pairs) * 100.0) if len(det_pairs) > 0 else 0.0
print(f"\nDAC (overall): {dac_overall:.2f}% ({len(intersection_pairs)}/{len(det_pairs)} pairs)")
print(f"NOTE: Intersection computed separately for DAC - learned_pairs_df remains unchanged")
```

**Why:** 
- Explicitly uses `det_pairs_df` for deterministic and `lrn_pairs_df` for learned
- Computes intersection separately for DAC but never replaces `lrn_pairs_df`
- Adds logging to verify intersection is computed correctly

---

### 4. Actionable Precision - Explicit DataFrame Usage (Lines 349-356)

**BEFORE:**
```python
ap_det = actionable_precision(risk, det, refp)
ap_lrn = actionable_precision(risk, lrn, refp)
```

**AFTER:**
```python
# CRITICAL: 
# - For deterministic metrics: use det_pairs_df
# - For learned metrics: use lrn_pairs_df (NOT intersection, NOT reference, NOT deterministic)
# actionable_precision uses the mapping to determine actionable positives but does NOT modify the mapping
ap_det = actionable_precision(risk, det_pairs_df, refp)
ap_lrn = actionable_precision(risk, lrn_pairs_df, refp)
```

**Why:** Explicitly uses the correct DataFrame for each metric branch.

---

### 5. Variance Consistency - Explicit DataFrame Usage (Lines 407-413)

**BEFORE:**
```python
vc_det = variance_consistency(risk, det, DEMOTION_FACTOR)
vc_lrn = variance_consistency(risk, lrn, DEMOTION_FACTOR)
```

**AFTER:**
```python
# CRITICAL:
# - For deterministic metrics: use det_pairs_df
# - For learned metrics: use lrn_pairs_df (NOT intersection, NOT reference, NOT deterministic)
vc_det = variance_consistency(risk, det_pairs_df, DEMOTION_FACTOR)
vc_lrn = variance_consistency(risk, lrn_pairs_df, DEMOTION_FACTOR)
```

**Why:** Explicitly uses the correct DataFrame for each metric branch.

---

### 6. Final Verification (Lines 440-460)

**BEFORE:**
```python
# No final verification
```

**AFTER:**
```python
# ------------------ FINAL VERIFICATION ------------------
# CRITICAL: Verify that learned pairs have NOT been overwritten or intersected
final_det_pairs = set(zip(det_pairs_df["technique_id"], det_pairs_df["control_id"]))
final_lrn_pairs = set(zip(lrn_pairs_df["technique_id"], lrn_pairs_df["control_id"]))
if final_det_pairs != det_pairs:
    print(f"\n❌ ERROR: Deterministic pairs were modified during computation!")
    print(f"  Before: {len(det_pairs)} pairs")
    print(f"  After: {len(final_det_pairs)} pairs")
    sys.exit(1)
if final_lrn_pairs != lrn_pairs:
    print(f"\n❌ ERROR: Learned pairs were modified during computation!")
    print(f"  Before: {len(lrn_pairs)} pairs")
    print(f"  After: {len(final_lrn_pairs)} pairs")
    sys.exit(1)

# Verify they're still different
final_only_in_det = final_det_pairs - final_lrn_pairs
final_only_in_learned = final_lrn_pairs - final_det_pairs
if len(final_only_in_det) == 0 and len(final_only_in_learned) == 0:
    print(f"\n❌ ERROR: After computation, mappings became identical!")
    print(f"  This should not happen - learned pairs must remain separate.")
    sys.exit(1)

print(f"\n✓ Verification passed:")
print(f"  Deterministic pairs: {len(final_det_pairs)} (unchanged)")
print(f"  Learned pairs: {len(final_lrn_pairs)} (unchanged)")
print(f"  Only in deterministic: {len(final_only_in_det)}")
print(f"  Only in learned: {len(final_only_in_learned)}")
```

**Why:** 
- Verifies that DataFrames were not modified during computation
- Verifies that mappings remain different after all computations
- Exits with error if any modification is detected

---

## Guarantees

After these fixes:

1. ✅ **Deterministic metrics** use ONLY `det_pairs_df` (from deterministic file)
2. ✅ **Learned metrics** use ONLY `lrn_pairs_df` (from learned file, NOT intersected)
3. ✅ **Intersection** is computed separately for DAC but never replaces `lrn_pairs_df`
4. ✅ **Error exit** if mappings are identical (both "only in" sets are 0)
5. ✅ **Final verification** ensures pairs remain unchanged and different

## Testing

Run the experiment:
```bash
python run_h3_experiment.py
```

The script will:
- Show detailed mapping comparison at start
- Exit with error if mappings are identical
- Use correct DataFrames for each metric branch
- Verify pairs remain unchanged at the end
- Show different results unless mappings are truly identical












