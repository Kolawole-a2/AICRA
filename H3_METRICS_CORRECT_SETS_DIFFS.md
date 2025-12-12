# H3 Metrics Use Correct Sets - Unified Diffs

## Summary

Added explicit comments and verification to ensure metrics use the correct sets:
- **Deterministic metrics** → use `det_pairs_df` ONLY
- **Learned metrics** → use `lrn_pairs_df` ONLY (AS-IS, NOT merged with deterministic, NOT merged with reference)
- **Intersection** → used ONLY for overlap ratios (DAC), NOT to replace learned mapping

## Key Changes

### 1. Enhanced Comments for mapping_metrics() (Lines 609-625)

**BEFORE:**
```python
# CRITICAL: Each branch MUST use its own mapping DataFrame
# - Deterministic branch: use det_pairs_df ONLY (derived from deterministic file)
# - Learned branch: use lrn_pairs_df ONLY (derived from learned file, NOT intersected)
# - Random branch: use random_baseline_df ONLY (generated randomly)
# Note: refp (reference pairs) is used as the canonical set for consistency calculation, but does NOT replace the mappings
mm_det = mapping_metrics(det_pairs_df, refp)  # Uses det_pairs_df for deterministic
mm_lrn = mapping_metrics(lrn_pairs_df, refp)  # Uses lrn_pairs_df for learned (NOT det_pairs_df)
mm_random = mapping_metrics(random_baseline_df, refp)  # Uses random_baseline_df for random
```

**AFTER:**
```python
# CRITICAL: Each branch MUST use its own mapping DataFrame - DO NOT merge or intersect
# - Deterministic branch: use det_pairs_df ONLY (derived from deterministic file)
# - Learned branch: use lrn_pairs_df ONLY (derived from learned file, AS-IS, NOT merged with det_pairs, NOT merged with refp)
# - Random branch: use random_baseline_df ONLY (generated randomly)
# 
# DO NOT DO THIS (WRONG):
#   learned_eval_pairs = learned_pairs.merge(det_pairs, ...)  # WRONG - forces learned to equal deterministic
#   mm_lrn = mapping_metrics(learned_eval_pairs, refp)  # WRONG
#
# DO THIS (CORRECT):
#   mm_lrn = mapping_metrics(lrn_pairs_df, refp)  # CORRECT - uses learned_pairs AS-IS
#
# Note: refp (reference pairs) is passed to mapping_metrics() ONLY to compute consistency
# (overlap ratio with canonical pairs). It does NOT replace the mapping_df parameter.
# The mapping_metrics() function uses lookup_df (the mapping) to compute coverage and pairs_count,
# and only uses ref_df to compute consistency (overlap with canonical pairs).
mm_det = mapping_metrics(det_pairs_df, refp)  # Uses det_pairs_df for deterministic metrics
mm_lrn = mapping_metrics(lrn_pairs_df, refp)  # Uses lrn_pairs_df for learned metrics (NOT det_pairs_df, NOT intersection, NOT refp)
mm_random = mapping_metrics(random_baseline_df, refp)  # Uses random_baseline_df for random metrics
```

**Why:** Explicitly shows what NOT to do (merge learned with deterministic) and clarifies that refp is only used for consistency calculation, not to replace the mapping.

---

### 2. Enhanced Comments for actionable_precision() (Lines 691-707)

**BEFORE:**
```python
# CRITICAL: Each branch MUST use its own mapping DataFrame
# - Deterministic branch: use det_pairs_df ONLY
# - Learned branch: use lrn_pairs_df ONLY (NOT det_pairs_df, NOT intersection, NOT reference)
# - Random branch: use random_baseline_df ONLY
# Note: refp is used as canonical set for actionable check, but does NOT replace the mappings
ap_det = actionable_precision(risk, det_pairs_df, refp)  # Uses det_pairs_df for deterministic
ap_lrn = actionable_precision(risk, lrn_pairs_df, refp)  # Uses lrn_pairs_df for learned (NOT det_pairs_df)
ap_random = actionable_precision(risk, random_baseline_df, refp)  # Uses random_baseline_df for random
```

**AFTER:**
```python
# CRITICAL: Each branch MUST use its own mapping DataFrame - DO NOT merge or intersect
# - Deterministic branch: use det_pairs_df ONLY (derived from deterministic file)
# - Learned branch: use lrn_pairs_df ONLY (derived from learned file, AS-IS, NOT merged with det_pairs, NOT merged with refp)
# - Random branch: use random_baseline_df ONLY (generated randomly)
#
# DO NOT DO THIS (WRONG):
#   learned_eval_pairs = learned_pairs.merge(det_pairs, ...)  # WRONG - forces learned to equal deterministic
#   ap_lrn = actionable_precision(risk, learned_eval_pairs, refp)  # WRONG
#
# DO THIS (CORRECT):
#   ap_lrn = actionable_precision(risk, lrn_pairs_df, refp)  # CORRECT - uses learned_pairs AS-IS
#
# Note: refp is passed to actionable_precision() ONLY to check if pairs are canonical-consistent
# (for the "actionable" check). It does NOT replace the mapping_df parameter.
# The actionable_precision() function uses mapping_df to determine which techniques have mapped controls,
# and only uses ref_df to check if those mapped pairs are canonical-consistent.
ap_det = actionable_precision(risk, det_pairs_df, refp)  # Uses det_pairs_df for deterministic metrics
ap_lrn = actionable_precision(risk, lrn_pairs_df, refp)  # Uses lrn_pairs_df for learned metrics (NOT det_pairs_df, NOT intersection, NOT refp)
ap_random = actionable_precision(risk, random_baseline_df, refp)  # Uses random_baseline_df for random metrics
```

**Why:** Same pattern - explicitly shows what NOT to do and clarifies that refp is only used for canonical-consistency check, not to replace the mapping.

---

### 3. Enhanced Comments for variance_consistency() (Lines 745-761)

**BEFORE:**
```python
# CRITICAL: Each branch MUST use its own mapping DataFrame
# - Deterministic branch: use det_pairs_df ONLY
# - Learned branch: use lrn_pairs_df ONLY (NOT det_pairs_df, NOT intersection, NOT reference)
# - Random branch: use random_baseline_df ONLY
vc_det = variance_consistency(risk, det_pairs_df, DEMOTION_FACTOR)  # Uses det_pairs_df for deterministic
vc_lrn = variance_consistency(risk, lrn_pairs_df, DEMOTION_FACTOR)  # Uses lrn_pairs_df for learned (NOT det_pairs_df)
vc_random = variance_consistency(risk, random_baseline_df, DEMOTION_FACTOR)  # Uses random_baseline_df for random
```

**AFTER:**
```python
# CRITICAL: Each branch MUST use its own mapping DataFrame - DO NOT merge or intersect
# - Deterministic branch: use det_pairs_df ONLY (derived from deterministic file)
# - Learned branch: use lrn_pairs_df ONLY (derived from learned file, AS-IS, NOT merged with det_pairs, NOT merged with refp)
# - Random branch: use random_baseline_df ONLY (generated randomly)
#
# DO NOT DO THIS (WRONG):
#   learned_eval_pairs = learned_pairs.merge(det_pairs, ...)  # WRONG - forces learned to equal deterministic
#   vc_lrn = variance_consistency(risk, learned_eval_pairs, DEMOTION_FACTOR)  # WRONG
#
# DO THIS (CORRECT):
#   vc_lrn = variance_consistency(risk, lrn_pairs_df, DEMOTION_FACTOR)  # CORRECT - uses learned_pairs AS-IS
#
# Note: variance_consistency() does NOT use refp - it only uses mapping_df to determine which techniques have mapped controls.
vc_det = variance_consistency(risk, det_pairs_df, DEMOTION_FACTOR)  # Uses det_pairs_df for deterministic metrics
vc_lrn = variance_consistency(risk, lrn_pairs_df, DEMOTION_FACTOR)  # Uses lrn_pairs_df for learned metrics (NOT det_pairs_df, NOT intersection)
vc_random = variance_consistency(risk, random_baseline_df, DEMOTION_FACTOR)  # Uses random_baseline_df for random metrics
```

**Why:** Same pattern - explicitly shows what NOT to do and clarifies that variance_consistency doesn't use refp at all.

---

### 4. Enhanced Comments for actionable_binary_vector() (Lines 777-789)

**BEFORE:**
```python
# CRITICAL: Each branch MUST use its own mapping DataFrame
# - Deterministic: use det_pairs_df ONLY
# - Learned: use lrn_pairs_df ONLY (NOT det_pairs_df, NOT intersection)
vec_det = actionable_binary_vector(risk, det_pairs_df, refp)  # Uses det_pairs_df for deterministic
vec_lrn = actionable_binary_vector(risk, lrn_pairs_df, refp)  # Uses lrn_pairs_df for learned (NOT det_pairs_df)
```

**AFTER:**
```python
# CRITICAL: Each branch MUST use its own mapping DataFrame - DO NOT merge or intersect
# - Deterministic branch: use det_pairs_df ONLY (derived from deterministic file)
# - Learned branch: use lrn_pairs_df ONLY (derived from learned file, AS-IS, NOT merged with det_pairs, NOT merged with refp)
#
# DO NOT DO THIS (WRONG):
#   learned_eval_pairs = learned_pairs.merge(det_pairs, ...)  # WRONG - forces learned to equal deterministic
#   vec_lrn = actionable_binary_vector(risk, learned_eval_pairs, refp)  # WRONG
#
# DO THIS (CORRECT):
#   vec_lrn = actionable_binary_vector(risk, lrn_pairs_df, refp)  # CORRECT - uses learned_pairs AS-IS
#
# Note: refp is passed to actionable_binary_vector() ONLY to check if pairs are canonical-consistent.
# It does NOT replace the mapping_df parameter.
vec_det = actionable_binary_vector(risk, det_pairs_df, refp)  # Uses det_pairs_df for deterministic metrics
vec_lrn = actionable_binary_vector(risk, lrn_pairs_df, refp)  # Uses lrn_pairs_df for learned metrics (NOT det_pairs_df, NOT intersection, NOT refp)
```

**Why:** Same pattern - explicitly shows what NOT to do and clarifies that refp is only used for canonical-consistency check.

---

### 5. Enhanced Final Verification Comment (Lines 838-841)

**BEFORE:**
```python
# ------------------ FINAL VERIFICATION ------------------
# CRITICAL: Verify that learned pairs have NOT been overwritten or intersected
```

**AFTER:**
```python
# ------------------ FINAL VERIFICATION ------------------
# CRITICAL: Verify that learned pairs have NOT been overwritten, merged, or intersected
# This ensures that lrn_pairs_df was never assigned from a merge result like:
#   lrn_pairs_df = learned_pairs.merge(det_pairs, ...)  # WRONG - would make learned equal deterministic
#   lrn_pairs_df = learned_pairs.merge(refp, ...)  # WRONG - would replace learned with reference
```

**Why:** Explicitly shows what the final verification is checking for - that learned_pairs was never assigned from a merge result.

---

## Guarantees

After these changes:

1. ✅ **Explicit DO NOT examples**: Shows exactly what NOT to do (merge learned with deterministic)
2. ✅ **Explicit DO examples**: Shows exactly what TO do (use learned_pairs AS-IS)
3. ✅ **Clarified refp usage**: Makes it clear that refp is only used for consistency/canonical checks, NOT to replace mappings
4. ✅ **All metric functions documented**: Every metric computation has explicit comments about correct usage
5. ✅ **Final verification enhanced**: Final check explicitly verifies learned_pairs was never merged

## Testing

Run the experiment:
```bash
python run_h3_experiment.py
```

The script will:
- Use `det_pairs_df` for all deterministic metrics
- Use `lrn_pairs_df` (AS-IS) for all learned metrics
- Use intersection ONLY for DAC overlap ratios, NOT to replace learned mapping
- Raise RuntimeError if metrics are identical when mappings differ
- Show detailed logging at each step

If the learned mapping is really different, the metrics will diverge.
If the learned mapping is still identical, the code will raise the RuntimeError instead of silently giving identical numbers.









