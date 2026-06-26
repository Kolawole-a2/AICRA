# H1/H2 Hypothesis Validation: Output Files and Plots Guide

## Answer: Which Outputs to Show for Hypothesis Validation

**For validating research hypotheses H1 and H2, you should use:1. **Primary Results**: Canonical H1/H2 experiment outputs (JSON + Markdown summaries) ⭐ **REQUIRED2. **Plots**: ⚠️ **IMPORTANT**: The rebuild pipeline plots are **NOT equivalent** to canonical results (see warning below)

**⚠️ CRITICAL WARNING**: The rebuild pipeline evaluates on **different data** (train+test combined) than canonical experiments (test-only), producing **different metrics**. Do NOT use rebuild plots to represent canonical validation results without clear disclaimers.

**See `docs/H1_H2_REBUILD_VS_CANONICAL_COMPARISON.md` for detailed comparison.--## Canonical H1/H2 Experiment Outputs (Primary Research Results)

### H1: Static PE Classification

**Location**: `results/H1_classification/`

#### Required Files for Hypothesis Validation:

1. **`H1_full_results.json`** ⭐ **PRIMARY   - Complete metrics including:
     - `metrics.auroc` - Area Under ROC Curve
     - `metrics.pr_auc` - Precision-Recall AUC
     - `metrics.precision`, `metrics.recall`, `metrics.f1`
     - `metrics.brier_score` - Calibration quality
     - `metrics.ece` - Expected Calibration Error
     - `metrics.baseline` - Baseline comparison metrics
     - `metrics.improvement` - % improvements over baseline
     - `metrics.alert_fatigue_reduction` - Alert fatigue metrics
   
2. **`H1_summary.md`** ⭐ **PRIMARY   - Human-readable summary report
   - Baseline comparison section
   - AICRA improvements over baseline
   - Alert fatigue reduction
   - Canonical improvement statement

3. **`metrics.json`** (Backward compatibility)
   - Same data as H1_full_results.json, different format

#### What These Files Show:
- ✅ **Hypothesis validation**: AUROC ≥ 0.95 (target met?)
- ✅ **Baseline comparison**: Improvement over logistic regression/majority classifier
- ✅ **Operational metrics**: Precision, Recall, F1 at operational threshold
- ✅ **Calibration quality**: Brier score, ECE
- ✅ **Alert fatigue reduction**: Estimated reduction percentage

#### Plots from Canonical H1:
❌ **The canonical H1 experiment does NOT generate plots directly--### H2: Calibration & Cost-Aware Thresholding

**Location**: `results/H2_calibration_thresholds/`

#### Required Files for Hypothesis Validation:

1. **`H2_full_results.json`** ⭐ **PRIMARY   - Complete metrics including:
     - `metrics.calibration.brier_before` / `brier_after` - Calibration improvement
     - `metrics.calibration.ece_before` / `ece_after` - ECE improvement
     - `metrics.calibration.brier_improvement_pct` - % improvement
     - `metrics.calibration.ece_improvement_pct` - % improvement
     - `metrics.calibration.baseline_brier` - Baseline Brier value
     - `metrics.calibration.baseline_ece` - Baseline ECE value
     - `metrics.thresholds.f1_optimal` - F1-optimized threshold
     - `metrics.thresholds.cost_optimal` - Cost-optimal threshold
     - `metrics.expected_loss.f1_optimized` - Expected loss at F1 threshold
     - `metrics.expected_loss.cost_optimal` - Expected loss at cost-optimal threshold
     - `metrics.improvement_statement` - Canonical statement

2. **`H2_summary.md`** ⭐ **PRIMARY   - Human-readable summary report
   - Calibration results (before/after)
   - Comparison vs typical baseline
   - Threshold comparison (F1-optimized vs cost-optimal)
   - Expected loss comparison
   - Canonical improvement statement

3. **`metrics.json`** (Backward compatibility)
   - Same data as H2_full_results.json, different format

#### What These Files Show:
- ✅ **Hypothesis validation**: Calibration improves Brier/ECE
- ✅ **Baseline comparison**: Improvement over uncalibrated baseline
- ✅ **Cost-aware thresholding**: Cost-optimal vs F1-optimized comparison
- ✅ **Expected loss reduction**: Banking-specific cost structure optimization

#### Plots from Canonical H2:
❌ **The canonical H2 experiment does NOT generate plots directly--## ⚠️ Rebuild Pipeline Plots (NOT Equivalent to Canonical Results)

**Location**: `results/h1h2_rebuild/<split>/plots/`

**⚠️ IMPORTANT**: These plots are **NOT equivalent** to canonical H1/H2 results because:
- They evaluate on **train+test combined** (not test-only)
- They produce **different metrics** (e.g., AUROC 0.9980 vs 0.9866)
- They include **training data in evaluation** (data leakage)

**Do NOT use these plots to represent canonical validation results without clear disclaimers.### Available Plots (For Demonstration Only):

1. **`roc.png`** - ROC Curve
   - ⚠️ Different AUROC than canonical (0.9980 vs 0.9866)
   - Evaluates on train+test combined, not test-only

2. **`pr.png`** - Precision-Recall Curve
   - ⚠️ Different PR-AUC than canonical
   - Evaluates on train+test combined

3. **`confusion.png`** - Confusion Matrix
   - ⚠️ Different confusion matrix than canonical
   - Based on different evaluation dataset

4. **`reliability.png`** - Reliability (Calibration) Diagram
   - ⚠️ Different calibration metrics than canonical H2
   - Evaluates on train+test combined

### Available Splits:
- `smoke_test/` - Small test set (200 samples)
- `small_ember/` - Small EMBER subset (2,000 samples)
- `main/` - Main split (10,000 samples)
### Canonical H1 splits (test-only evaluation):
- `full_ember/` - Full temporal test holdout (**10,001 samples**)
- `main/` - Main split (10,000 samples)
- `small_ember/` - Small EMBER subset (2,000 samples)
- `smoke_test/` - Smoke test (200 samples)

### Rebuild pipeline splits (different dataset — use with caution):
- `full_ember/` - Train+test combined (**50,005+ samples**) ⚠️ **Not canonical H1 test holdout**

### Metrics Files:
- `results/h1h2_rebuild/<split>/metrics.json` - Per-split metrics (different from canonical)
- `results/h1h2_rebuild/metrics_summary.json` - Aggregated metrics (different from canonical)

---

## Recommended Approach for Praxis Defense

### For Hypothesis Validation (Required):

1. **Show Primary Results**:
   ```
   results/H1_classification/H1_full_results.json
   results/H1_classification/H1_summary.md
   results/H2_calibration_thresholds/H2_full_results.json
   results/H2_calibration_thresholds/H2_summary.md
   ```

2. **Reference Key Metrics**:
   - H1: AUROC, PR-AUC, Precision, Recall, F1, Brier, ECE
   - H2: Brier improvement %, ECE improvement %, Expected loss reduction

### For Visualization (Supporting Material):

3. **⚠️ Rebuild Pipeline Plots** (Use with CAUTION):
   ```
   results/h1h2_rebuild/full_ember/plots/roc.png
   results/h1h2_rebuild/full_ember/plots/pr.png
   results/h1h2_rebuild/full_ember/plots/confusion.png
   results/h1h2_rebuild/full_ember/plots/reliability.png
   ```

   **⚠️ Important Disclaimers if Using These Plots**:
   - **MUST clearly label** as "Optional Rebuild Pipeline - Different Dataset"
   - **MUST explain** that they evaluate on train+test combined (not test-only)
   - **MUST state** that metrics are different from canonical results
   - **MUST NOT** use them to represent canonical validation results
   - **RECOMMENDED**: Generate plots directly from canonical experiment outputs instead

---

## File Structure Summary

```
results/
├── H1_classification/              ⭐ PRIMARY FOR H1 VALIDATION
│   ├── H1_full_results.json        ← Show this
│   ├── H1_summary.md                ← Show this
│   └── metrics.json                (backward compatibility)
│
├── H2_calibration_thresholds/      ⭐ PRIMARY FOR H2 VALIDATION
│   ├── H2_full_results.json         ← Show this
│   ├── H2_summary.md                ← Show this
│   └── metrics.json                (backward compatibility)
│
└── h1h2_rebuild/                   ⚠️ OPTIONAL (for plots/visualization)
    ├── full_ember/
    │   ├── metrics.json
    │   └── plots/
    │       ├── roc.png              ← Can use for visualization
    │       ├── pr.png               ← Can use for visualization
    │       ├── confusion.png        ← Can use for visualization
    │       └── reliability.png      ← Can use for visualization
    └── metrics_summary.json
```

---

## Quick Reference: What to Show in Your Praxis

### H1 Hypothesis Validation:

**Required (Primary)**:
- ✅ `results/H1_classification/H1_full_results.json` - Metrics
- ✅ `results/H1_classification/H1_summary.md` - Summary report

**Optional (Visualization - Use with Disclaimers)**:
- ⚠️ `results/h1h2_rebuild/full_ember/plots/roc.png` - ROC curve (different dataset!)
- ⚠️ `results/h1h2_rebuild/full_ember/plots/pr.png` - PR curve (different dataset!)
- ⚠️ `results/h1h2_rebuild/full_ember/plots/confusion.png` - Confusion matrix (different dataset!)

### H2 Hypothesis Validation:

**Required (Primary)**:
- ✅ `results/H2_calibration_thresholds/H2_full_results.json` - Metrics
- ✅ `results/H2_calibration_thresholds/H2_summary.md` - Summary report

**Optional (Visualization - Use with Disclaimers)**:
- ⚠️ `results/h1h2_rebuild/full_ember/plots/reliability.png` - Calibration diagram (different dataset!)
- ⚠️ `results/h1h2_rebuild/full_ember/plots/roc.png` - ROC curve (different dataset!)

---

## Defense Strategy

### When Asked: "Where are your H1/H2 validation results?"

**Answer**:
"My primary hypothesis validation results are in:
- `results/H1_classification/H1_full_results.json` and `H1_summary.md`
- `results/H2_calibration_thresholds/H2_full_results.json` and `H2_summary.md`

These contain the complete metrics that validate my hypotheses. For visualization, I also have plots from the optional rebuild pipeline that demonstrate the same model performance visually."

### When Asked: "Why do you use plots from the rebuild pipeline?"

**Answer**:
"I do NOT use rebuild pipeline plots to represent canonical H1/H2 validation results because they evaluate on different data (train+test combined vs test-only) and produce different metrics. The canonical H1/H2 experiments produce comprehensive JSON metrics and markdown summaries. If I need plots for visualization, I would generate them directly from the canonical experiment's test set predictions, or clearly label rebuild plots as demonstration-only with appropriate disclaimers explaining they use different evaluation data."

---

## Summary

✅ **For Hypothesis Validation**: Use canonical H1/H2 results (JSON + Markdown)
- `results/H1_classification/H1_full_results.json` + `H1_summary.md`
- `results/H2_calibration_thresholds/H2_full_results.json` + `H2_summary.md`

⚠️ **For Plots/Visualization**: 
- **RECOMMENDED**: Generate plots directly from canonical experiment test set predictions
- **If using rebuild plots**: MUST clearly label and explain they use different data (train+test combined vs test-only) and different metrics
- `results/h1h2_rebuild/full_ember/plots/*.png` - Use with disclaimers only

**⚠️ CRITICAL**: The optional rebuild pipeline does NOT produce equivalent results to canonical experiments. It evaluates on different data and produces different metrics. Do NOT use rebuild plots to represent canonical validation results without clear disclaimers.

**See `docs/H1_H2_REBUILD_VS_CANONICAL_COMPARISON.md` for detailed comparison.