# Critical: H1/H2 Rebuild Pipeline vs Canonical Experiments - NOT EQUIVALENT

## ⚠️ IMPORTANT FINDING

**The H1/H2 rebuild pipeline outputs are NOT the same as the canonical H1/H2 experiment outputs.## Key Differences

### Canonical H1 Experiment

**Data Split:- Uses time-ordered split: **Train (40,004 samples) / Test (10,001 samples)- Evaluates on **test set only** (10,001 samples)

**Results** (canonical `results/H1_classification/H1_full_results.json`, banking threshold **0.0248**, FN:FP = **100:1**):
- AUROC: **0.9796** (full_ember)
- PR-AUC: **0.9767**
- Precision: **0.6478**
- Recall: **0.9985**
- F1: **0.7858**
- Brier: **0.0551**
- ECE: **0.0079**

**Location:** `results/H1_classification/H1_full_results.json`

---

### Rebuild Pipeline (full_ember)

**Data Split:- Loads time-ordered split, then **COMBINES train+test** for evaluation
- Evaluates on **combined dataset** (50,006 samples = train + test)

**Results:- AUROC: **0.9980** (different!)
- PR-AUC: **0.9979** (different!)
- Precision: **0.9847** (different!)
- Recall: **0.9809** (different!)
- F1: **0.9828** (different!)
- Brier: **0.0142** (different!)
- ECE: **0.0122** (different!)

**Location:** `results/h1h2_rebuild/full_ember/metrics.json`

---

## Why They're Different

1. **Different Evaluation Datasets:   - Canonical H1: Evaluates on **test set only** (10,001 samples)
   - Rebuild Pipeline: Evaluates on **train+test combined** (50,006 samples)

2. **Data Leakage in Rebuild Pipeline:   - The rebuild pipeline combines train and test data, then evaluates on the combined set
   - This includes training data in the evaluation, which inflates metrics
   - This is **NOT appropriate for hypothesis validation3. **Different Sample Sizes:   - Canonical: 10,001 test samples
   - Rebuild: 50,006 samples (5x larger, includes training data)

4. **Different Metrics:   - All metrics are different because they're evaluated on different datasets
   - Rebuild metrics are higher because they include training data

---

## ⚠️ CORRECTED RECOMMENDATION

### ❌ DO NOT Use Rebuild Pipeline Plots for Canonical H1/H2 Results

**Reason:** The rebuild pipeline evaluates on different data (train+test combined) and produces different metrics. Using these plots to represent canonical results would be **misleading** because:

1. The metrics don't match
2. The evaluation includes training data (data leakage)
3. The sample sizes are different
4. This violates proper experimental methodology

### ✅ Correct Approach for Hypothesis Validation

**For H1/H2 Hypothesis Validation:1. **Use Canonical Results Only:   - `results/H1_classification/H1_full_results.json`
   - `results/H1_classification/H1_summary.md`
   - `results/H2_calibration_thresholds/H2_full_results.json`
   - `results/H2_calibration_thresholds/H2_summary.md`

2. **Generate Plots from Canonical Data (If Needed):   - If you need plots, you should generate them from the canonical experiment's test set predictions
   - OR clearly label rebuild plots as "demonstration only" and explain they use different data

3. **If Using Rebuild Plots:   - **MUST clearly label** them as "Optional Rebuild Pipeline - Different Dataset"
   - **MUST explain** that they evaluate on train+test combined (not just test)
   - **MUST state** that metrics are different and these are for demonstration only
   - **MUST NOT** use them to represent canonical H1/H2 validation results

---

## What the Rebuild Pipeline is Actually For

The rebuild pipeline is designed for:
- ✅ Generating **operational artifacts** (risk registers)
- ✅ Demonstrating **per-sample outputs** across multiple splits
- ✅ Showing **end-to-end pipeline** operation
- ❌ **NOT** for hypothesis validation
- ❌ **NOT** equivalent to canonical experiments

---

## Comparison Table

| Aspect | Canonical H1/H2 | Rebuild Pipeline |
|--------|----------------|------------------|
| **Evaluation Dataset** | Test set only (10,001) | Train+test combined (50,006) |
| **Data Leakage** | ❌ No (proper train/test split) | ⚠️ Yes (includes training data) |
| **AUROC** | 0.9796 | 0.9980 (different!) |
| **Precision** | 0.6478 | 0.9847 (different!) |
| **Purpose** | Hypothesis validation | Operational demonstration |
| **Appropriate for Validation?** | ✅ Yes | ❌ No |

---

## Corrected Defense Strategy

### When Asked: "Can I use rebuild pipeline plots for H1/H2 validation?"

**Answer:"No, I cannot use rebuild pipeline plots to represent canonical H1/H2 validation results because:

1. **Different datasets**: Canonical H1 evaluates on test set only (10,001 samples), while rebuild pipeline evaluates on train+test combined (50,006 samples)

2. **Different metrics**: The metrics are different (e.g., AUROC 0.9796 vs 0.9980) because they're evaluated on different data

3. **Data leakage**: The rebuild pipeline includes training data in evaluation, which is not appropriate for hypothesis validation

4. **Different purpose**: Canonical experiments are for hypothesis validation; rebuild pipeline is for operational demonstration

For hypothesis validation, I use only the canonical H1/H2 results. If I need plots, I would generate them from the canonical experiment's test set predictions, or clearly label rebuild plots as demonstration-only with appropriate disclaimers."

---

## Action Items

1. ✅ **Use canonical H1/H2 results** for hypothesis validation
2. ❌ **Do NOT use rebuild pipeline plots** to represent canonical results
3. ⚠️ **If you must use rebuild plots**, clearly label them and explain the differences
4. 📊 **Consider generating plots** directly from canonical experiment outputs if visualization is needed

---

## Summary

**The rebuild pipeline outputs are NOT equivalent to canonical H1/H2 experiments.- Different evaluation datasets
- Different metrics
- Different purposes
- **Do NOT use rebuild plots to represent canonical validation results**For hypothesis validation, use only canonical H1/H2 results.