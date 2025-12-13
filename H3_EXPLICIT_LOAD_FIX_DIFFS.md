# H3 Explicit Load Fix - Unified Diffs

## Summary

Added explicit loads at the top of the file and verification checks throughout to ensure deterministic and learned mappings remain separate and are used correctly for all metrics.

## Key Changes

### 1. Explicit Load at Top (Lines 28-115)

**BEFORE:**
```python
# Setup logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ------------------ CONFIG ------------------
```

**AFTER:**
```python
# Setup logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ------------------ EXPLICIT LOAD AT TOP (BEFORE ANY OTHER LOGIC) ------------------
print("=" * 80)
print("EXPLICIT MAPPING LOAD AT TOP")
print("=" * 80)

# EXPLICIT LOAD: Deterministic mapping
DET_PATH = Path("data/mappings/deterministic_attack_defense_lookup.csv")
# ... fallback logic ...
det_df = pd.read_csv(DET_PATH)
LOGGER.info("Loaded deterministic mapping from: %s", DET_PATH)
LOGGER.info("Deterministic mapping shape: %s", det_df.shape)
LOGGER.info("Deterministic mapping columns: %s", list(det_df.columns))

# Handle column name variations - normalize to attack_id/defense_id first
if "attack_id" not in det_df.columns and "technique_id" in det_df.columns:
    det_df = det_df.rename(columns={"technique_id": "attack_id", "control_id": "defense_id"})

# Extract pairs using attack_id/defense_id
if "attack_id" in det_df.columns and "defense_id" in det_df.columns:
    det_pairs = det_df[["attack_id", "defense_id"]].drop_duplicates().copy()
    LOGGER.info("Extracted deterministic pairs: %d rows", len(det_pairs))
    LOGGER.info("Sample deterministic pairs:\n%s", det_pairs.head(5).to_string(index=False))

# EXPLICIT LOAD: Learned mapping
LEARNED_PATH = Path("data/mappings/learned_embedding_attack_defense_mapping.csv")
learned_df = pd.read_csv(LEARNED_PATH)
LOGGER.info("Loaded learned mapping from: %s", LEARNED_PATH)
LOGGER.info("Learned mapping shape: %s", learned_df.shape)
LOGGER.info("Learned mapping columns: %s", list(learned_df.columns))

# Extract pairs using attack_id/defense_id (AS-IS, NOT intersected with deterministic)
if "attack_id" in learned_df.columns and "defense_id" in learned_df.columns:
    learned_pairs = learned_df[["attack_id", "defense_id"]].drop_duplicates().copy()
    LOGGER.info("Extracted learned pairs: %d rows (AS-IS, NOT intersected)", len(learned_pairs))
    LOGGER.info("Sample learned pairs:\n%s", learned_pairs.head(5).to_string(index=False))

# CRITICAL: Verify they are different at load time
det_pairs_set = set(zip(det_pairs["attack_id"], det_pairs["defense_id"]))
learned_pairs_set = set(zip(learned_pairs["attack_id"], learned_pairs["defense_id"]))
only_in_det_at_load = det_pairs_set - learned_pairs_set
only_in_learned_at_load = learned_pairs_set - det_pairs_set

LOGGER.info("At load time - Deterministic pairs: %d", len(det_pairs_set))
LOGGER.info("At load time - Learned pairs: %d", len(learned_pairs_set))
LOGGER.info("At load time - Only in deterministic: %d", len(only_in_det_at_load))
LOGGER.info("At load time - Only in learned: %d", len(only_in_learned_at_load))

if len(only_in_det_at_load) == 0 and len(only_in_learned_at_load) == 0:
    raise RuntimeError(
        "Learned mapping is identical to deterministic mapping at load time: DAC comparison is meaningless. "
        "Check that the learned mapping is actually generated from embeddings and not copied from deterministic."
    )
```

**Why:** Explicit loads at the very top ensure mappings are loaded separately before any other logic runs, with immediate verification that they're different.

---

### 2. Metric Functions Work on Copies (Lines 87-237, 238-264, 615-635)

**BEFORE:**
```python
def mapping_metrics(lookup_df, ref_df):
    # ... uses lookup_df directly ...
    lookup_pairs = set(map(tuple, lookup_df[["technique_id", "control_id"]].dropna().values.tolist()))
    # ...
```

**AFTER:**
```python
def mapping_metrics(lookup_df, ref_df):
    """
    ...
    CRITICAL: This function does NOT modify lookup_df. It only reads from it.
    """
    # CRITICAL: Work on a copy to ensure we don't modify the input
    lookup_df = lookup_df.copy()
    # ... rest of function ...
```

**Why:** Ensures metric functions never modify the input DataFrames, preventing accidental overwrites.

---

### 3. Verification Before Each Metric Computation (Lines 550-567, 636-645, 680-689)

**BEFORE:**
```python
# ------------------ MAPPING METRICS ------------------
print("\nComputing mapping metrics...")
mm_det = mapping_metrics(det_pairs_df, refp)
mm_lrn = mapping_metrics(lrn_pairs_df, refp)
```

**AFTER:**
```python
# ------------------ MAPPING METRICS ------------------
print("\nComputing mapping metrics...")

# CRITICAL VERIFICATION: Ensure mappings are still different before computing metrics
verify_det_pairs = set(zip(det_pairs_df["technique_id"], det_pairs_df["control_id"]))
verify_lrn_pairs = set(zip(lrn_pairs_df["technique_id"], lrn_pairs_df["control_id"]))
verify_only_in_det = verify_det_pairs - verify_lrn_pairs
verify_only_in_learned = verify_lrn_pairs - verify_det_pairs

LOGGER.info("BEFORE mapping_metrics - Deterministic pairs: %d", len(verify_det_pairs))
LOGGER.info("BEFORE mapping_metrics - Learned pairs: %d", len(verify_lrn_pairs))
LOGGER.info("BEFORE mapping_metrics - Only in deterministic: %d", len(verify_only_in_det))
LOGGER.info("BEFORE mapping_metrics - Only in learned: %d", len(verify_only_in_learned))

if len(verify_only_in_det) == 0 and len(verify_only_in_learned) == 0:
    raise RuntimeError(
        "Learned mapping became identical to deterministic mapping before metric computation: DAC comparison is meaningless. "
        "This indicates a bug - learned_pairs_df was overwritten or intersected."
    )

mm_det = mapping_metrics(det_pairs_df, refp)
mm_lrn = mapping_metrics(lrn_pairs_df, refp)
```

**Why:** Verifies mappings are still different before each metric computation, catching any overwrites early.

---

### 4. RuntimeError if Metrics Identical When Mappings Differ (Lines 605-620, 660-675, 705-720)

**BEFORE:**
```python
print(f"Deterministic: Coverage={mm_det['coverage_%']}%, Consistency={mm_det['consistency_%']}%")
print(f"Learned: Coverage={mm_lrn['coverage_%']}%, Consistency={mm_lrn['consistency_%']}%")
```

**AFTER:**
```python
print(f"Deterministic: Coverage={mm_det['coverage_%']}%, Consistency={mm_det['consistency_%']}%, Pairs={mm_det['pairs_count']}")
print(f"Learned: Coverage={mm_lrn['coverage_%']}%, Consistency={mm_lrn['consistency_%']}%, Pairs={mm_lrn['pairs_count']}")

# CRITICAL: If metrics are identical but mappings differ, this is a problem
if (mm_det['coverage_%'] == mm_lrn['coverage_%'] and 
    mm_det['consistency_%'] == mm_lrn['consistency_%'] and 
    mm_det['pairs_count'] == mm_lrn['pairs_count']):
    if verify_det_pairs != verify_lrn_pairs:
        print(f"\n{'='*80}")
        print("❌ CRITICAL ERROR: Mappings differ but metrics are IDENTICAL!")
        print(f"{'='*80}")
        print(f"This indicates a bug in the evaluation code.")
        print(f"Deterministic pairs: {len(verify_det_pairs)}, Learned pairs: {len(verify_lrn_pairs)}")
        print(f"Only in det: {len(verify_only_in_det)}, Only in learned: {len(verify_only_in_learned)}")
        raise RuntimeError(
            "Mappings differ but mapping_metrics returned identical values. "
            "This indicates a bug - the evaluation is using the same mapping for both branches."
        )
```

**Why:** Catches the exact problem the user reported - if mappings differ but metrics are identical, raises RuntimeError immediately.

---

## Guarantees

After these changes:

1. ✅ **Explicit loads at top**: Mappings loaded separately before any other logic
2. ✅ **Immediate verification**: Checked for differences at load time
3. ✅ **Metric functions use copies**: Never modify input DataFrames
4. ✅ **Verification before each metric**: Ensures mappings remain different
5. ✅ **RuntimeError on identical metrics**: Raises error if metrics identical when mappings differ
6. ✅ **Detailed logging**: LOGGER.info statements throughout for debugging

## Testing

Run the experiment:
```bash
python run_h3_experiment.py
```

The script will:
- Load mappings explicitly at the top
- Verify they're different immediately
- Check before each metric computation
- Raise RuntimeError if metrics are identical when mappings differ
- Show detailed logging at each step

If you still see identical metrics, the RuntimeError will catch it and show exactly where the problem is.












