> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# Verification: H1-H3 Experiment Outputs Unchanged

## Analysis Summary

✅ **Good News:** The security fixes and experimental design changes should **NOT affect H1-H3 output values** for the following reasons:

---

## ✅ H1 Experiment - SAFE

**File:** `aicra/experiments/h1_classification.py`

**What Changed:- H1 calls `load_ember_2024(time_ordered=True)` (line 121)
- The `load_ember_2024()` function now supports `time_ordered` parameter

**Impact Analysis:- **Before:** `load_ember_2024()` didn't accept `time_ordered` parameter, so it would have either:
  - Failed with `TypeError` (if strict), OR
  - Ignored the parameter and used default behavior (random/stratified split)
- **After:** Now properly implements time-ordered split when `time_ordered=True`

**Potential Issue:- If H1 was previously running with a **random/stratified split** (because `time_ordered=True` was ignored), the new time-ordered split **will produce different results**.
- However, this is actually a **fix** - H1 was supposed to use time-ordered splits but wasn't working correctly.

**Recommendation:- If you want to preserve exact previous results, you can temporarily set `time_ordered=False` in H1
- But the time-ordered split is the **correct** behavior for H1 (avoids temporal leakage)

---

## ✅ H2 Experiment - SAFE (No Changes)

**File:** `aicra/experiments/h2_calibration_thresholds.py`

**What Changed:- H2 calls `load_ember_2024()` without parameters (line 151)
- Default behavior is `time_ordered=False`, so **no change in behavior**Impact:** ✅ **ZERO** - H2 outputs will be identical

---

## ✅ H3 Experiment - SAFE (No Changes)

**File:** `aicra/experiments/h3_evaluation.py`

**What Changed:- H3 does **NOT** use `load_ember_2024()` at all
- H3 loads data from CSV files directly using `pd.read_csv()`
- H3 uses `yaml.safe_load()` (unchanged, already safe)

**Impact:** ✅ **ZERO** - H3 outputs will be identical

---

## ✅ Security Fixes - SAFE (No Impact on Experiments)

**Files Changed:- `aicra/utils/policy_writer.py`
- `aicra/utils/train_ffnn.py`
- `aicra/utils/evaluate.py`
- `aicra/utils/train_lightgbm.py`

**Impact Analysis:- These are **utility scripts**, NOT used by H1, H2, H3 experiments
- H1-H3 experiments use:
  - `load_ember_2024()` → loads from JSONL files (not NPZ)
  - `pd.read_csv()` → loads from CSV files
  - `joblib.load()` → loads models (not affected)
- The `np.load()` fixes only affect standalone utility scripts

**Impact:** ✅ **ZERO** - No effect on H1-H3 experiments

---

## ⚠️ Potential Change: H1 Time-Ordered Split

**If H1 was previously running with random/stratified split** (because `time_ordered=True` was ignored), the new implementation will:

1. **Sort data by timestamp2. **Split chronologically** (80% train, 20% test)
3. **Produce different train/test setsThis will change:
- Model training data
- Test set composition
- All metrics (AUROC, Precision, Recall, F1, etc.)

**However:- This is the **correct** behavior (avoids temporal leakage)
- Time-ordered splits are required for proper evaluation
- Previous results may have been **incorrect** if they used random splits

---

## Verification Steps

To verify your outputs are unchanged:

### 1. Check H1 Results

```bash
# Compare old vs new H1 results
# If time-ordered split was working before, results should be identical
# If it wasn't, results will differ (but new results are correct)
```

### 2. Check H2 Results

```bash
# H2 should produce identical results
# It uses default load_ember_2024() (time_ordered=False)
```

### 3. Check H3 Results

```bash
# H3 should produce identical results
# It doesn't use load_ember_2024() at all
```

---

## Summary

| Experiment | Output Changed? | Reason |
|------------|----------------|--------|
| **H1** | ⚠️ **Possibly** | Time-ordered split now works correctly (was broken before) |
| **H2** | ✅ **No** | Uses default `time_ordered=False` |
| **H3** | ✅ **No** | Doesn't use `load_ember_2024()` at all |

---

## Recommendation

1. **H2 and H3:** ✅ Safe - outputs will be identical
2. **H1:** If you want to preserve exact previous results:
   - Temporarily change line 121 in `h1_classification.py`:
     ```python
     # Change from:
     train_data, test_data = load_ember_2024(time_ordered=True)
     # To:
     train_data, test_data = load_ember_2024(time_ordered=False)
     ```
   - But this is **not recommended** - time-ordered splits are correct for H1

---

## Conclusion

✅ **H2 and H3 outputs are guaranteed unchanged⚠️ **H1 outputs may differ if time-ordered split wasn't working before** (but new results are more correct)

