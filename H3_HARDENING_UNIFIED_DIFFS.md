# H3 Evaluation Hardening - Unified Diffs

## Summary

Hardened the H3 evaluation to ensure metrics use the correct sets, added random baseline mapping, computed DAC for three comparisons, and added sample output for visual verification.

## Key Changes

### 1. Replaced Random Baseline Function (Lines 213-226)

**BEFORE:**
```python
def generate_random_baseline_mapping(det_pairs_df, lrn_pairs_df, seed=42):
    """
    Generate a random baseline mapping for contrast.
    Uses same techniques as deterministic/learned but randomly assigns controls.
    """
    rng = np.random.default_rng(seed)
    
    # Get all unique techniques and controls from both mappings
    all_techniques = set(det_pairs_df["technique_id"].unique()) | set(lrn_pairs_df["technique_id"].unique())
    all_controls = set(det_pairs_df["control_id"].unique()) | set(lrn_pairs_df["control_id"].unique())
    
    # For each technique, randomly assign k controls (where k is similar to learned mapping)
    # Get average number of controls per technique in learned mapping
    avg_controls_per_tech = len(lrn_pairs_df) / len(all_techniques) if len(all_techniques) > 0 else 3
    k = max(1, int(round(avg_controls_per_tech)))
    
    random_pairs = []
    all_controls_list = list(all_controls)
    
    for tech in all_techniques:
        # Randomly select k controls for this technique
        selected_controls = rng.choice(all_controls_list, size=min(k, len(all_controls_list)), replace=False)
        for ctrl in selected_controls:
            random_pairs.append({"technique_id": tech, "control_id": ctrl})
    
    random_df = pd.DataFrame(random_pairs)
    # ... logging ...
    return random_df
```

**AFTER:**
```python
def build_random_mapping(det_pairs: pd.DataFrame, n_defenses: int, seed: int = 17) -> pd.DataFrame:
    """
    Build a random baseline mapping for contrast.
    Uses same attacks as deterministic but randomly assigns defenses.
    """
    rng = np.random.default_rng(seed)
    unique_attacks = det_pairs["technique_id"].unique()  # Use technique_id (same as attack_id after rename)
    unique_defenses = det_pairs["control_id"].unique()  # Use control_id (same as defense_id after rename)
    
    rows = []
    for attack_id in unique_attacks:
        # Choose 1 random defense per attack
        defense_id = rng.choice(unique_defenses)
        rows.append({"technique_id": attack_id, "control_id": defense_id})
    
    return pd.DataFrame(rows).drop_duplicates()
```

**Why:** Simpler function that assigns exactly 1 random defense per attack, providing a clear baseline for comparison.

---

### 2. Added Sample Output for Visual Check (Lines 447-454)

**BEFORE:**
```python
# No sample output
```

**AFTER:**
```python
# ------------------ OUTPUT SAMPLES FOR VISUAL CHECK ------------------
print(f"\n{'='*80}")
print("SAMPLE ROWS FOR VISUAL CHECK")
print(f"{'='*80}")
LOGGER.info("Sample deterministic pairs:\n%s", det_pairs_df.head(5).to_string(index=False))
LOGGER.info("Sample learned pairs:\n%s", lrn_pairs_df.head(5).to_string(index=False))
LOGGER.info("Sample random pairs:\n%s", random_baseline_df.head(5).to_string(index=False))
print(f"{'='*80}\n")
```

**Why:** Allows visual verification that the three mappings are genuinely different before computing metrics.

---

### 3. Compute DAC for Three Sets (Lines 500-520)

**BEFORE:**
```python
# Compute DAC using DataFrame-based intersection (already computed above)
dac_overall = (len(intersection) / len(det_pairs_for_merge) * 100.0) if len(det_pairs_for_merge) > 0 else 0.0
print(f"\nDAC (overall): {dac_overall:.2f}% ({len(intersection)}/{len(det_pairs_for_merge)} pairs)")
print(f"NOTE: Intersection computed using DataFrame merge - learned_pairs_df remains unchanged")
```

**AFTER:**
```python
# ------------------ COMPUTE DAC FOR THREE SETS ------------------
print(f"\n{'='*80}")
print("COMPUTING DAC FOR THREE COMPARISONS")
print(f"{'='*80}")

# DAC 1: Deterministic vs itself (should be 1.0)
det_vs_det_intersection = det_pairs_for_merge.merge(det_pairs_for_merge, on=["technique_id", "control_id"], how="inner")
dac_det_vs_det = (len(det_vs_det_intersection) / len(det_pairs_for_merge) * 100.0) if len(det_pairs_for_merge) > 0 else 0.0
print(f"DAC (deterministic vs itself): {dac_det_vs_det:.2f}% (should be 100.00%)")

# DAC 2: Learned vs deterministic (already computed above as intersection)
dac_learned_vs_det = (len(intersection) / len(det_pairs_for_merge) * 100.0) if len(det_pairs_for_merge) > 0 else 0.0
print(f"DAC (learned vs deterministic): {dac_learned_vs_det:.2f}% ({len(intersection)}/{len(det_pairs_for_merge)} pairs)")

# DAC 3: Random vs deterministic
random_vs_det_intersection = random_baseline_df.merge(det_pairs_for_merge, on=["technique_id", "control_id"], how="inner")
dac_random_vs_det = (len(random_vs_det_intersection) / len(det_pairs_for_merge) * 100.0) if len(det_pairs_for_merge) > 0 else 0.0
print(f"DAC (random vs deterministic): {dac_random_vs_det:.2f}% ({len(random_vs_det_intersection)}/{len(det_pairs_for_merge)} pairs)")

print(f"NOTE: All intersections computed using DataFrame merge - original DataFrames remain unchanged")
print(f"{'='*80}\n")
```

**Why:** Computes DAC for all three comparisons (deterministic vs itself, learned vs deterministic, random vs deterministic) to provide comprehensive comparison.

---

### 4. Enhanced Metric Comments to Ensure Correct Set Usage (Lines 471-480, 503-520, 559-560, 578-580)

**BEFORE:**
```python
# CRITICAL: Use det_pairs_df for deterministic metrics, lrn_pairs_df for learned metrics
mm_det = mapping_metrics(det_pairs_df, refp)
mm_lrn = mapping_metrics(lrn_pairs_df, refp)
mm_random = mapping_metrics(random_baseline_df, refp)
```

**AFTER:**
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

**Why:** Explicit comments ensure each metric branch uses the correct DataFrame and clarify that `refp` is only used as a reference, not to replace mappings.

---

### 5. Updated JSON Output to Include Three DAC Values (Lines 650-662)

**BEFORE:**
```python
"mapping_comparison": {
    "deterministic_pairs": len(det_pairs_for_merge),
    "learned_pairs": len(learned_pairs_for_merge),
    "intersection_pairs": len(intersection),
    "overlap_pairs": len(overlap),
    "only_in_deterministic": len(only_in_det_df),
    "only_in_learned": len(only_in_learned_df),
    "similarity_det": round(det_similarity, 4),
    "similarity_lrn": round(lrn_similarity, 4),
    "dac_overall": round(dac_overall, 2),
    "debug_counts": debug_counts
},
```

**AFTER:**
```python
"mapping_comparison": {
    "deterministic_pairs": len(det_pairs_for_merge),
    "learned_pairs": len(learned_pairs_for_merge),
    "random_baseline_pairs": len(random_baseline_df),
    "intersection_pairs": len(intersection),
    "overlap_pairs": len(overlap),
    "only_in_deterministic": len(only_in_det_df),
    "only_in_learned": len(only_in_learned_df),
    "similarity_det": round(det_similarity, 4),
    "similarity_lrn": round(lrn_similarity, 4),
    "dac_deterministic_vs_itself": round(dac_det_vs_det, 2),
    "dac_learned_vs_deterministic": round(dac_learned_vs_det, 2),
    "dac_random_vs_deterministic": round(dac_random_vs_det, 2),
    "debug_counts": debug_counts
},
```

**Why:** Includes all three DAC values and random baseline pair count in the output.

---

### 6. Enhanced Summary Markdown with DAC Section (Lines 750-765)

**BEFORE:**
```markdown
## Mapping Comparison

- Deterministic pairs: **{count}**
- Learned pairs: **{count}**
- Overlap: **{count}** ({dac}% DAC)
```

**AFTER:**
```markdown
## Mapping Comparison

- Deterministic pairs: **{count}**
- Learned pairs: **{count}**
- Random baseline pairs: **{count}**
- Overlap (deterministic ∩ learned): **{count}**
- Only in deterministic: **{count}**
- Only in learned: **{count}**
- Similarity (overlap/det): **{value}**
- Similarity (overlap/learned): **{value}**

## DAC (Defense-Attack Consistency)

- DAC (deterministic vs itself): **{value}%** (should be 100.00%)
- DAC (learned vs deterministic): **{value}%**
- DAC (random vs deterministic): **{value}%**
```

**Why:** Provides comprehensive comparison information including all three DAC values.

---

## Guarantees

After these changes:

1. ✅ **Metrics use correct sets**: Each branch (deterministic, learned, random) uses ONLY its own DataFrame
2. ✅ **Random baseline included**: Random mapping provides contrast to verify evaluation works
3. ✅ **Three DAC values computed**: Deterministic vs itself, learned vs deterministic, random vs deterministic
4. ✅ **Sample output logged**: Visual verification that mappings are different
5. ✅ **RuntimeError on identical mappings**: Code raises exception instead of silently proceeding
6. ✅ **Plots show three series**: All plots include deterministic, learned, and random baseline

## Verification

Run the experiment:
```bash
python run_h3_experiment.py
```

Check the output for:
- `DEBUG_DAC_COUNTS` showing non-zero values for `only_in_det` and `only_in_learned`
- Sample rows showing different pairs for deterministic, learned, and random
- Three DAC values in the output
- Three-way bar plots with different values

If `only_in_det = 0` and `only_in_learned = 0`, the learned mapping file itself is identical to deterministic, and the embedding mapping generator needs to be fixed.













