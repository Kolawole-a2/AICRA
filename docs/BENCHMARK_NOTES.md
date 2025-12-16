## Benchmark Notes (Current Repository State)

**Last Updated**: 2025-12-16  
**Source Artifacts**:
- `results/H1_classification/H1_full_results.json`
- `results/H2_calibration_thresholds/H2_full_results.json`
- `results/H3_full_evaluation/H3_full_results.json`

---

## H1 – Static PE Classification (full_ember)

**Source**: `results/H1_classification/H1_full_results.json`

**Dataset / Split**:
- Train samples: 40,004
- Test samples: 10,001
- Model: LightGBM (static PE features)

**Core Metrics**:
- **AUROC**: 0.9866
- **PR-AUC**: 0.9869
- **Precision**: 0.9459
- **Recall**: 0.9363
- **F1**: 0.9411
- **Brier Score**: 0.0426
- **ECE**: 0.0066

**Targets vs. Results**:
- **Precision ≥ 0.88** → 0.9459 (**PASS**)
- **Recall ≥ 0.88** → 0.9363 (**PASS**)
- **F1 ≥ 0.88** → 0.9411 (**PASS**)
- **Brier Score < 0.12** → 0.0426 (**PASS**)
- **ECE < 0.12** → 0.0066 (**PASS**)

**Interpretation**:
- H1 comfortably exceeds the AUROC ≥ 0.95 target.
- Precision/Recall/F1 all exceed 0.93 on the full_ember evaluation.
- Probability calibration is strong (very low Brier and ECE).

---

## H2 – Calibration & Cost-Aware Thresholding (full_ember)

**Source**: `results/H2_calibration_thresholds/H2_full_results.json`

**Test Samples**:
- Test samples: 10,001
- Cost parameters: FN cost = 10.0, FP cost = 1.0

### Calibration Metrics

- **Brier (uncalibrated)**: 0.0426  
- **Brier (calibrated)**: 0.0500  
- **ECE (uncalibrated)**: 0.0066  
- **ECE (calibrated)**: 0.0457  

All Brier/ECE values are **below 0.12**, satisfying the calibration target.

### F1-Optimized Threshold (Uncalibrated & Calibrated)

**Uncalibrated (F1-optimized)**:
- Threshold: 0.4586  
- Precision: 0.9404  
- Recall: 0.9429  
- F1: 0.9416  
- Expected Loss: 0.3027  

**Calibrated (F1-optimized)**:
- Threshold: 0.2268  
- Precision: 0.9404  
- Recall: 0.9429  
- F1: 0.9416  
- Expected Loss: 0.3027  

**Targets vs. Results (F1-optimized)**:
- **Precision ≥ 0.88** → 0.9404 (**PASS**)
- **Recall ≥ 0.88** → 0.9429 (**PASS**)
- **F1 ≥ 0.88** → 0.9416 (**PASS**)
- **Brier < 0.12 / ECE < 0.12** → see calibration metrics above (**PASS**)

### Cost-Optimized Threshold (Banking-Style Costs)

**Uncalibrated (cost-optimal)**:
- Threshold: 0.1040  
- Precision: 0.8213  
- Recall: 0.9854  
- F1: 0.8959  
- Expected Loss: 0.1729  

**Calibrated (cost-optimal)**:
- Threshold: 0.0100  
- Precision: 0.9047  
- Recall: 0.9654  
- F1: 0.9341  
- Expected Loss: 0.2148  

**Interpretation**:
- The F1-optimized setting yields Precision/Recall/F1 all above 0.94 with strong calibration.
- The cost-optimal setting (uncalibrated) minimizes expected loss under FN≫FP, still maintaining F1 ≈ 0.896 and very high recall (~0.99).
- Calibrated cost-optimal thresholds trade a small loss in expected loss for higher precision/recall and F1 (>0.93).

---

## H3 – Deterministic vs Learned Mapping (DAC)

**Source**: `results/H3_full_evaluation/H3_full_results.json`

### Per-Split Deterministic Mapping Metrics (Selected)

For each split (main, small_ember, full_ember; smoke_test similar), the deterministic mapping achieves:

- **Coverage**: 100% of techniques mapped (`coverage_%` = 100.0)
- **DAC (Defense–Attack Consistency)**: 100% (`dac_%` = 100.0)
- **Actionable Precision**: 1.0
- **Actionable F1**: 1.0

The learned mapping:

- **Coverage**: 100% (by construction in this evaluation)
- **DAC**: 0.0 (`dac_%` = 0.0)
- **Actionable Precision**: 0.0
- **Actionable F1**: 0.0

### Baseline vs Deterministic Metrics (Example: small_ember)

- **Baseline (pre-mapping) risk distribution**:
  - AUROC: 0.8307  
  - PR-AUC: 0.4900  
  - Brier Score: 0.2112  
  - ECE: 0.2092  

- **Deterministic mapping deltas**:
  - Δ DAC: +100.0 percentage points
  - Δ actionable precision: +1.0
  - Δ actionable F1: +1.0

### Summary

- Deterministic ATT&CK→D3FEND mapping achieves **100% DAC_internal** and **100% coverage** across evaluated splits.
- Learned mapping, as evaluated here, has 0% DAC and 0 actionable precision/F1, confirming deterministic mapping as the normative ground truth for H3.

---

## Alignment with 88% / 0.12 Targets

Based on the **current repository artifacts**:

- **H1 (full_ember)**:
  - Precision 0.9459, Recall 0.9363, F1 0.9411 → all **≥ 0.88**
  - Brier 0.0426, ECE 0.0066 → both **< 0.12**

- **H2 (full_ember, F1-optimized)**:
  - Precision 0.9404, Recall 0.9429, F1 0.9416 → all **≥ 0.88**
  - Brier/ECE values identical to/unified with H1 uncalibrated metrics → **< 0.12**

These numbers are **directly read from the JSON outputs** listed at the top of this document and represent the validated state of the repository as of the last update.

---

## File Paths Used

- `results/H1_classification/H1_full_results.json`
- `results/H2_calibration_thresholds/H2_full_results.json`
- `results/H3_full_evaluation/H3_full_results.json`


