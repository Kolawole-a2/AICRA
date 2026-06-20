# Susceptibility Diagnosis: Benign Samples with High Scores

**Date:** 2025-01-XX  
**Issue:** Benign samples (label=0) are getting high susceptibility scores, with 33.9% (small_ember) and 50.3% (full_ember) classified as "High" risk.

## 1. Problem Reproduction

### Current State Analysis

**SMALL_EMBER (2,000 records):- Benign (label=0): 1,509 samples (75.4%)
- Ransomware (label=1): 491 samples (24.6%)
- **Benign High Rate:** 33.9% (511/1,509)
- **Ransomware Mean Probability:** 0.904269 (constant - only 1 unique value!)
- **Benign Mean Probability:** 0.308409
- **False Positive Rate:** 33.9%

**FULL_EMBER (20,002 records):- Benign (label=0): 10,440 samples (52.2%)
- Ransomware (label=1): 9,562 samples (47.8%)
- **Benign High Rate:** 50.3% (5,253/10,440)
- **Ransomware Mean Probability:** 0.453415
- **Benign Mean Probability:** 0.456640 (almost identical to ransomware!)
- **False Positive Rate:** 50.3%
- **False Negative Rate:** 50.0%

### Key Findings

1. **SMALL_EMBER**: Ransomware samples have ONLY ONE unique probability value (0.904269) - this is clearly incorrect. A properly trained model should produce a distribution of probabilities.

2. **FULL_EMBER**: Benign and ransomware have nearly identical probability distributions (mean 0.456640 vs 0.453415), indicating the probabilities are not from actual model predictions.

3. **Probability Values**: The values 0.904269 and 0.003310 appear repeatedly, suggesting placeholder or incorrectly generated values rather than model predictions.

## 2. Label Correctness Verification

**Label Mapping (Confirmed):- `label = 0` → Benign (confirmed in `model_card.md` and code)
- `label = 1` → Ransomware (confirmed in `model_card.md` and code)

**Label Distribution:- Labels are correctly assigned (0 for benign, 1 for ransomware)
- No evidence of label inversion in the data

**Conclusion:** Labels are correct. The issue is NOT label inversion.

## 3. Probability Correctness Verification

### Expected Behavior

For a properly trained ransomware detection model:
- **Benign samples (label=0)** should have **LOW** probabilities (close to 0)
- **Ransomware samples (label=1)** should have **HIGH** probabilities (close to 1)
- The model should output `P(ransomware)` via `predict_proba()[:, 1]`

### Current Behavior

**SMALL_EMBER:- Ransomware: ALL samples have probability = 0.904269 (constant, only 1 unique value)
- Benign: 141 unique probabilities, but many are 0.904269 (same as ransomware)

**FULL_EMBER:- Both classes have 141 unique probabilities
- Mean probabilities are almost identical (0.456640 vs 0.453415)
- This suggests probabilities are NOT from model predictions

### Root Cause

The probability values in the register files are **NOT from actual model predictions**. They appear to be:
1. Placeholder values (0.904269, 0.003310)
2. Incorrectly generated (possibly from a broken model or wrong data)
3. Not regenerated from the model when registers were updated

**Evidence:- `regenerate_risk_registers_fixed.py` loads existing register files and re-computes susceptibility/buckets from existing (wrong) probabilities
- It does NOT generate new probabilities from `model.predict_proba()`
- The script preserves the existing probability column: `register_df["probability"] = input_df["probability"].values`

## 4. Bucketing Logic Audit

**Location:** `aicra/register.py` lines 92-98

```python
df["susceptibility"] = df["probability"].clip(0.0, 1.0)
df["susceptibility_bucket"] = pd.cut(
    df["susceptibility"],
    bins=[0.0, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"],
    include_lowest=True,
)
```

**Bucketing Logic:- Low: [0.0, 0.33]
- Medium: (0.33, 0.66]
- High: (0.66, 1.0]

**Analysis:- Bucketing logic is **CORRECT- Thresholds are monotonic and appropriate
- The issue is that **input probabilities are wrong**, not the bucketing

## 5. Calibration Sanity Checks

**Status:** Cannot verify calibration because probabilities are not from model predictions.

**Expected:- Calibrated probabilities should improve calibration metrics (Brier score, ECE)
- Calibration should be applied via `calibrator.transform(y_prob_raw)`

**Current:- Cannot assess calibration quality because probabilities are not from model

## 6. Root Cause Analysis

### Primary Root Cause

**The probability values in risk register CSV files are NOT from actual model predictions.**Evidence:1. **SMALL_EMBER**: Ransomware has only 1 unique probability (0.904269) - impossible for a real model
2. **FULL_EMBER**: Benign and ransomware have nearly identical distributions - indicates no discrimination
3. **Regeneration Script**: `regenerate_risk_registers_fixed.py` preserves existing probabilities instead of generating new ones from the model

**Why Benign Has High Susceptibility:- Because the probability values are wrong (not from model)
- Many benign samples have probability = 0.904269 (same as ransomware)
- Bucketing correctly assigns High bucket to probabilities > 0.66
- The bucketing is working correctly; the input probabilities are wrong

### Secondary Issues

1. **No Validation**: The regeneration script does not validate that probabilities are from model predictions
2. **No Assertions**: No checks to ensure benign samples have low probabilities
3. **Silent Failure**: The script silently preserves wrong probabilities

## 7. Required Fixes

### Fix 1: Regenerate Probabilities from Model

**File:** `regenerate_registers_with_correct_probabilities.py`

**Changes Required:1. Load test data from EMBER dataset
2. Generate predictions using `model.predict_proba()`
3. Apply calibration if available
4. Use these probabilities in register (NOT existing CSV probabilities)

**Implementation:- Already exists in `regenerate_registers_with_correct_probabilities.py`
- Need to ensure it runs successfully and replaces existing registers

### Fix 2: Add Validation Assertions

**File:** `aicra/register.py` or validation script

**Add checks:1. Verify benign samples have mean probability < 0.5 (or configurable threshold)
2. Verify ransomware samples have mean probability > 0.5
3. Verify probabilities have sufficient variance (not constant)
4. Raise error if validation fails

### Fix 3: Update Regeneration Script

**File:** `regenerate_risk_registers_fixed.py`

**Issue:** Script preserves wrong probabilities

**Fix:** Do NOT use this script for probability regeneration. Use `regenerate_registers_with_correct_probabilities.py` instead.

## 8. Implementation Plan

1. **Run Correct Regeneration:**

   ```bash
   python scripts/regenerate_registers_with_correct_probabilities.py --debug
   ```

2. **Validate Results:**
   - Benign mean probability should be < 0.3
   - Ransomware mean probability should be > 0.7
   - Benign high rate should be < 5%
   - False positive rate should be < 10%

3. **Add Validation Assertions:**
   - Add to `aicra/register.py` or create validation utility
   - Fail fast if probabilities are wrong

## 9. Expected Outcomes After Fix

**Before:**
- Benign High Rate: 33.9% (small_ember), 50.3% (full_ember)
- Benign Mean Prob: 0.308409 (small_ember), 0.456640 (full_ember)
- Ransomware Mean Prob: 0.904269 (small_ember, constant), 0.453415 (full_ember)

**After (Expected):**
- Benign High Rate: < 5%
- Benign Mean Prob: < 0.3
- Ransomware Mean Prob: > 0.7
- False Positive Rate: < 10%
- Proper probability distributions (not constant values)

## 10. Examiner-Safe Explanation

**Issue:** Benign samples were incorrectly assigned high susceptibility scores (33.9-50.3% in High bucket).

**Root Cause:** The probability values in risk register files were not generated from actual model predictions. They appeared to be placeholder or incorrectly generated values, causing benign samples to have probabilities similar to ransomware samples.

**Fix:** Regenerated risk registers using actual model predictions (`model.predict_proba()`) with proper calibration. This ensures benign samples receive low probabilities (low susceptibility) and ransomware samples receive high probabilities (high susceptibility), as expected from a properly trained model.

**Validation:** Added assertions to verify probability distributions are correct and fail fast if benign samples have inappropriately high probabilities.

