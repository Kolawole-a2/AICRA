# H3 Variance Reduction Reconciliation Report

**Date:** 2024-12-19  
**Purpose:** Reconcile inconsistencies between README claims and actual H3 experimental results for variance reduction metrics.

---

## Executive Summary

**Issue Identified:** README claims H3 variance reduction of **-47%**, but actual experimental results show **0.0%** across all evaluation splits.

**Root Cause:** The variance reduction computation returns 0.0 because:
1. All ATT&CK techniques in the evaluation splits have mapped D3FEND controls in both deterministic and learned mappings
2. The score adjustment logic (`compute_score_consistency`) only demotes scores for unmapped positives
3. Since all techniques are mapped, no score adjustments occur, resulting in `baseline_variance == mapped_variance`
4. Therefore, `variance_reduction = baseline_var - adjusted_var = 0.0`

**Resolution:** Update README and all documentation to reflect the actual computed value of **0.0%** variance reduction, with an explanation of why this occurs.

---

## 1. Inventory of Variance Reduction Claims

### 1.1 Claims in Documentation

| File | Line/Section | Claimed Variance Reduction | Context | Source |
|------|--------------|---------------------------|---------|--------|
| `README.md` | Line 393 | **-47%** | Combined baseline comparison | Manual/theoretical |
| `README.md` | Line 533 | **47%** | "reduces risk-score variance by 47%" | Manual/theoretical |
| `README.md` | Line 535 | **40-50%** | Variance reduction range | Manual/theoretical |
| `docs/RESULTS_SUMMARY.md` | Line 92 | **0.0** | Variance Reduction Delta | Computed from experiments |
| `results/H3_full_evaluation/H3_full_summary.md` | Line 96 | **0.000000** | Mean Variance Reduction | Computed from experiments |
| `results/H3_full_evaluation/H3_full_results.json` | Multiple | **0.0** | Per-split variance_reduction | Computed from experiments |
| `SOURCE_CONTRIBUTION_AND_AICRA_IMPROVEMENTS.md` | Line 93 | **-47%** | Combined Baselines comparison | Theoretical/expected |

### 1.2 Actual Computed Values

**Source:** `results/H3_full_evaluation/H3_full_results.json`

| Split | Deterministic Variance Reduction | Learned Variance Reduction | Delta |
|-------|----------------------------------|----------------------------|-------|
| main | 0.0 | 0.0 | 0.0 |
| small_ember | 0.0 | 0.0 | 0.0 |
| full_ember | 0.0 | 0.0 | 0.0 |
| smoke_test | 0.0 | 0.0 | 0.0 |
| **Aggregated Mean** | **0.0** (SD: 0.0) | **0.0** (SD: 0.0) | **0.0** (SD: 0.0) |

**Source:** `results/H3_full_evaluation/H3_full_summary.md` (lines 94-102):
- **Deterministic:** 0.000000 (SD: 0.000000)
- **Learned:** 0.000000 (SD: 0.000000)
- **Mean Δ Variance Reduction:** 0.000000 (SD: 0.000000)
- **95% CI for Δ Variance Reduction:** [0.000000, 0.000000]

---

## 2. Code Analysis: Variance Reduction Computation

### 2.1 Computation Logic

**File:** `aicra/experiments/h3_evaluation.py` (lines 528-575)

```python
def compute_score_consistency(
    risk_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    demotion_factor: float = 0.90
) -> Dict:
    """
    Compute score consistency metrics (variance and IQR reduction).
    
    For positives whose technique has NO mapped controls, demote risk_score by demotion_factor.
    """
    # Techniques with ANY control in mapping
    techniques_with_controls = set(mapping_df["technique_id"].dropna().unique())
    
    # Adjust scores for unmapped positives
    def adjust_score(row):
        if row["predicted_label"] == 1 and row["technique_id"] not in techniques_with_controls:
            return row["risk_score"] * demotion_factor
        return row["risk_score"]  # No adjustment if technique has controls
    
    risk_df["risk_score_adjusted"] = risk_df.apply(adjust_score, axis=1)
    
    # Compute baseline metrics
    baseline_var = float(np.var(risk_df["risk_score"], ddof=1))
    
    # Compute adjusted metrics
    adjusted_var = float(np.var(risk_df["risk_score_adjusted"], ddof=1))
    
    # Compute reductions
    variance_reduction = baseline_var - adjusted_var  # This is 0.0 if no adjustments occurred
```

### 2.2 Why Variance Reduction is 0.0

**Root Cause Analysis:1. **All techniques have mapped controls:** In all evaluation splits (main, small_ember, full_ember, smoke_test), every ATT&CK technique present in the risk scores has at least one mapped D3FEND control in both the deterministic and learned mappings.

2. **No score adjustments occur:** Because all techniques have controls, the `adjust_score` function never applies the `demotion_factor`. It always returns `row["risk_score"]` unchanged.

3. **Identical variances:** Since `risk_score == risk_score_adjusted` for all rows, `baseline_var == adjusted_var`, resulting in `variance_reduction = 0.0`.

**Evidence from Results:- Deterministic mapping: 173 pairs covering 46 unique techniques
- Learned mapping: 190 pairs covering 47 unique techniques
- All techniques in evaluation splits are covered by both mappings

### 2.3 Percentage Improvement Calculation Bug

**File:** `aicra/experiments/h3_evaluation.py` (lines 1257-1279)

**Issue:** The code passes `variance_reduction` values (which are 0.0) to `compute_h3_improvements`, but that function expects actual variance values to compute percentage improvements.

```python
deterministic_variance = aggregated["deterministic"]["variance_reduction"]["mean"]  # 0.0
learned_variance = aggregated["learned"]["variance_reduction"]["mean"]  # 0.0

h3_improvements = compute_h3_improvements(
    deterministic_variance=deterministic_variance,  # Wrong: passing 0.0 instead of actual variance
    learned_variance=learned_variance,  # Wrong: passing 0.0 instead of actual variance
    ...
)
```

**File:** `aicra/core/benchmarks.py` (line 354)

```python
# Variance reduction (lower is better)
variance_reduction_pct = 100 * (learned_variance - deterministic_variance) / learned_variance if learned_variance > 0 else 0.0
```

Since both `learned_variance` and `deterministic_variance` are 0.0, the result is `0.0 / 0.0 = 0.0` (handled by the `if learned_variance > 0` guard).

**Note:** This bug doesn't affect the result (both would be 0.0 anyway), but it indicates a conceptual mismatch: the function expects variance values, not variance reduction values.

---

## 3. Origin of the -47% Claim

### 3.1 Source Analysis

The **-47%** claim appears in:
- `README.md` (line 393)
- `SOURCE_CONTRIBUTION_AND_AICRA_IMPROVEMENTS.md` (line 93)

**From `SOURCE_CONTRIBUTION_AND_AICRA_IMPROVEMENTS.md`:```
| **Combined Baselines** | Risk Score Variance | High (learned mapping) | Low (deterministic) | -47% variance | **47% Variance Reduction** |
```

**Interpretation:** This appears to be a **theoretical/expected value** based on the assumption that:
- Learned mappings produce "high" variance (more inconsistent risk scores)
- Deterministic mappings produce "low" variance (more consistent risk scores)
- The difference is estimated at 47%

However, this is **not computed from actual experimental data**. It appears to be a placeholder or expected value that was never updated after running the actual experiments.

### 3.2 Why the Theoretical Value Doesn't Match Reality

The theoretical assumption that deterministic mappings reduce variance doesn't hold in this evaluation because:

1. **All techniques are mapped:** Both deterministic and learned mappings provide controls for all techniques, so no score demotion occurs.

2. **Variance reduction requires unmapped techniques:** The current implementation only reduces variance when some techniques lack controls, causing score demotion. Since all techniques have controls, no variance reduction occurs.

3. **Different variance sources:** The theoretical -47% may have assumed variance reduction from mapping consistency (fewer/noisier controls = more variance), but the actual implementation measures variance reduction from score demotion (unmapped techniques = lower scores = different variance).

---

## 4. Reconciliation Decision

### 4.1 What to Fix

**Decision:** Update documentation to reflect actual computed values (**0.0%** variance reduction) rather than theoretical values (**-47%**).

**Rationale:1. **Scientific accuracy:** Documentation must match experimental results
2. **Reproducibility:** Readers should be able to verify claims against actual outputs
3. **Transparency:** The explanation of why variance reduction is 0.0 is operationally meaningful (all techniques are mapped, so no score adjustments occur)

### 4.2 What NOT to Fix

**Decision:** Do NOT modify the variance reduction computation code.

**Rationale:1. **No bug in computation:** The code correctly computes `variance_reduction = baseline_var - adjusted_var`. The result is 0.0 because no adjustments occur, which is mathematically correct.
2. **Design choice:** The current implementation measures variance reduction from score demotion, not from mapping consistency. This is a valid design choice, even if it doesn't produce the expected variance reduction in this evaluation.
3. **Future work:** If variance reduction from mapping consistency is desired, it would require a different computation (e.g., comparing variance of risk scores when using deterministic vs learned mappings directly, not after score adjustment).

---

## 5. Updated Claims

### 5.1 Corrected Variance Reduction Statement

**Old Claim (README.md line 533):> "Deterministic mapping increases ATT&CK–D3FEND mapping coverage by 30% and reduces risk-score variance by 47%."

**Corrected Claim:> "Deterministic mapping increases ATT&CK–D3FEND mapping coverage by 48.1% and achieves 100% Defense-Attack Consistency (DAC). Variance reduction is 0.0% because all ATT&CK techniques in the evaluation splits have mapped D3FEND controls in both deterministic and learned mappings, so no score adjustments occur."

### 5.2 Corrected Table Entry

**Old Claim (README.md line 393):```
| **H3** | Combined | 100% | Variance reduction | **-47%** |
```

**Corrected Claim:```
| **H3** | Combined | 100% | Variance reduction | **0.0%** |
```

**Note:** Add explanation: "Variance reduction is 0.0% because all techniques have mapped controls, so no score demotion occurs. See `docs/H3_RECONCILIATION_REPORT.md` for details."

---

## 6. README Accuracy Audit for H1-H3

### 6.1 H1 Claims Verification

| README Section | Claim | Verified by | Status | Notes |
|----------------|-------|-------------|--------|-------|
| Line 39 | AUROC >= 0.95 | `results/H1_classification/H1_full_results.json` | ✅ OK | Actual: 0.9866 |
| Line 54 | Precision, Recall, F1 at threshold 0.5 | `results/H1_classification/H1_full_results.json` | ✅ OK | Verified |
| Line 48 | Time-ordered evaluation | Code: `aicra/experiments/h1_classification.py` | ✅ OK | Implemented |
| Line 49 | Out-of-family evaluation | Code: `aicra/experiments/h1_out_of_sample_eval.py` | ✅ OK | Implemented |

### 6.2 H2 Claims Verification

| README Section | Claim | Verified by | Status | Notes |
|----------------|-------|-------------|--------|-------|
| Line 65 | Brier score and ECE reduction | `results/H2_calibration_thresholds/H2_full_results.json` | ✅ OK | Verified |
| Line 74 | Isotonic regression | Code: `aicra/pipelines/calibration.py` | ✅ OK | Implemented |
| Line 80 | Expected loss minimization | `results/H2_calibration_thresholds/H2_full_results.json` | ✅ OK | Verified |
| Line 391 | Expected Loss reduction: -65.4% | `artifacts/benchmark_improvements.json` | ✅ OK | Verified |

### 6.3 H3 Claims Verification

| README Section | Claim | Verified by | Status | Notes |
|----------------|-------|-------------|--------|-------|
| Line 89 | Deterministic vs learned mapping comparison | `results/H3_full_evaluation/H3_full_results.json` | ✅ OK | Verified |
| Line 393 | Variance reduction: -47% | `results/H3_full_evaluation/H3_full_results.json` | ❌ **NEEDS FIX** | Actual: 0.0% |
| Line 391 | Coverage improvement: +48.1% | `results/H3_full_evaluation/H3_full_results.json` | ✅ OK | Verified |
| Line 392 | Consistency improvement: +60.0% | `results/H3_full_evaluation/H3_full_results.json` | ✅ OK | Verified |
| Line 533 | "reduces risk-score variance by 47%" | `results/H3_full_evaluation/H3_full_results.json` | ❌ **NEEDS FIX** | Actual: 0.0% |
| Line 535 | "Variance reduction: 40-50%" | `results/H3_full_evaluation/H3_full_results.json` | ❌ **NEEDS FIX** | Actual: 0.0% |

---

## 7. Reproducibility Instructions

### 7.1 How to Reproduce Variance Reduction = 0.0

**Command:```bash
python experiments/h3_mapping_compare.py
```

**Expected Output:- `results/H3_full_evaluation/H3_full_results.json`
- Check `aggregated_metrics.deterministic.variance_reduction.mean` → Should be 0.0
- Check `aggregated_metrics.learned.variance_reduction.mean` → Should be 0.0

**Verification:```bash
# Extract variance reduction values
python -c "
import json
with open('results/H3_full_evaluation/H3_full_results.json') as f:
    data = json.load(f)
    det_var = data['aggregated_metrics']['deterministic']['variance_reduction']['mean']
    lrn_var = data['aggregated_metrics']['learned']['variance_reduction']['mean']
    print(f'Deterministic variance reduction: {det_var}')
    print(f'Learned variance reduction: {lrn_var}')
"
```

**Expected Output:```
Deterministic variance reduction: 0.0
Learned variance reduction: 0.0
```

### 7.2 Why Variance Reduction is 0.0

**Explanation:1. All ATT&CK techniques in the evaluation splits have mapped D3FEND controls in both deterministic and learned mappings.
2. The `compute_score_consistency` function only demotes risk scores for unmapped positives (techniques with no controls).
3. Since all techniques are mapped, no score demotion occurs: `risk_score == risk_score_adjusted` for all rows.
4. Therefore, `baseline_variance == mapped_variance`, resulting in `variance_reduction = 0.0`.

**Code Reference:- `aicra/experiments/h3_evaluation.py`, lines 528-575: `compute_score_consistency()`
- `results/H3_full_evaluation/H3_full_results.json`: Per-split `consistency_metrics.variance_reduction` values

---

## 8. Summary

### 8.1 What Changed

- **Old claim:** H3 variance reduction = -47% (theoretical/expected value)
- **Correct claim:** H3 variance reduction = 0.0% (actual computed value)

### 8.2 Why It Changed

- The -47% value was a theoretical estimate that doesn't match actual experimental results
- Actual experiments show 0.0% variance reduction because all techniques have mapped controls, so no score adjustments occur

### 8.3 How to Reproduce

- Run `python experiments/h3_mapping_compare.py`
- Check `results/H3_full_evaluation/H3_full_results.json` → `aggregated_metrics.deterministic.variance_reduction.mean` = 0.0

### 8.4 Files to Update

1. `README.md` (lines 393, 533, 535)
2. `SOURCE_CONTRIBUTION_AND_AICRA_IMPROVEMENTS.md` (line 93, 156) - Note: This file may be intentionally theoretical; add a note clarifying it's an expected value, not computed

--**Report Generated:** 2024-12-19  
**Next Steps:** Apply README updates as documented in Section 5.

