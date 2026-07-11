## Benchmark Notes (Current Repository State)

**Last Updated**: 2025-12-16  
**Source Artifacts**:
- `results/H1_classification/H1_full_results.json`
- `results/H2_calibration_thresholds/H2_full_results.json`
- `results/H3_full_evaluation/H3_full_results.json`

---

## H1 – Static PE Classification (full_ember)

**Source**: `results/H1_classification/H1_full_results.json` → `metrics.per_split_results[full_ember]`

**Dataset / Split**:
- Train samples: 40,004
- Test samples: 10,001
- Model: LightGBM (static PE features)
- **Banking threshold**: 0.0248 (FN cost = 100, FP cost = 1)

**Core Metrics** (full_ember split @ banking threshold):
- **AUROC**: 0.9796
- **PR-AUC**: 0.9767
- **Precision**: 0.6478
- **Recall**: 0.9985
- **F1**: 0.7858
- **Brier Score**: 0.0551
- **ECE**: 0.0079

**Targets vs. Results**:
- **AUROC > 0.88** (reliability benchmark) → 0.9796 (**PASS**)
- **AUROC ≥ 0.95** (design target) → 0.9796 (**PASS**)
- **Brier Score < 0.12** → 0.0551 (**PASS**)
- **ECE < 0.12** → 0.0079 (**PASS**)
- **Precision ≥ 0.88** → 0.6478 (**intentional trade-off** at banking threshold; see `docs/PRECISION_RECALL_TRADE_OFF_BANKING.md`)

**Interpretation**:
- H1 exceeds the **> 0.88 AUROC reliability benchmark** and the ≥ 0.95 design target on full_ember.
- Validated across **time-ordered** train/test, **multi-split** evaluation (mean AUROC 0.9610), and supplementary **OOF** (AUROC 0.9616).
- Empirical logistic baseline AUROC **0.7811** on the same split (+**25.4%** lift vs full_ember AICRA 0.9796).
- Banking threshold prioritizes recall (0.9985) over precision (0.6478) under 100:1 FN:FP costs.

---

## H2 – Post-Hoc Calibration Test & Cost-Aware Thresholding (full_ember)

**Source**: `results/H2_calibration_thresholds/H2_full_results.json`

**Test Samples**:
- Test samples: 10,001
- Cost parameters: FN cost = 10.0, FP cost = 1.0

### Calibration Metrics (help test — not assumed benefit)

Platt/isotonic regression applied post hoc **to test whether calibration improves** reported metrics. Finding: model already well-calibrated from H1; calibration does not improve expected loss (primary H2 comparison remains cost-optimal vs F1-optimal thresholds).

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
- Primary H2 finding: cost-optimal (uncalibrated) minimizes expected loss under FN≫FP while maintaining high recall (~0.99).
- Post-hoc calibration **help test**: does not improve expected loss on this model (already well-calibrated from H1).
- Calibrated cost-optimal thresholds trade a small change in expected loss for different precision/recall balance.

---

## H3 – Deterministic vs Learned Mapping (DAC)

**Variance note**: Across all splits, deterministic mapping is always correct (100% DAC_internal) and learned is always extraneous (0%). Variance reduction is **0.0 for both**; t-test, Wilcoxon, and Shapiro–Wilk on variance reduction are **not applicable**. H3 validated via perfect separation and deterministic dominance.

**Source**: `results/H3_full_evaluation/H3_full_results.json`

### Per-Split Deterministic Mapping Metrics (Selected)

For each production-scale split (main, small_ember, full_ember), the deterministic mapping achieves:

- **Coverage**: 100% of techniques mapped (`coverage_%` = 100.0)
- **DAC_internal**: 100% (`dac_%` = 100.0)
- **Actionable Precision**: 1.0

On **smoke_test** (n=2), deterministic actionable precision is **0.0** (no actionable alerted rows); mean across four splits is **0.75**.

The learned mapping on all splits:

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

- **H1 (full_ember @ banking threshold 0.0248)**:
  - AUROC 0.9796 → exceeds **> 0.88** benchmark and **≥ 0.95** design target
  - Precision 0.6478, Recall 0.9985 — recall-first banking trade-off (precision below 0.88 by design)
  - Brier 0.0551, ECE 0.0079 → both **< 0.12**
- **H2 (full_ember, F1-optimized, uncalibrated)**:
  - Precision 0.9404, Recall 0.9429, F1 0.9416 → all **≥ 0.88**
  - Brier 0.0426, ECE 0.0066 (uncalibrated probabilities) → both **< 0.12**
- **H2 (full_ember, cost-optimal, uncalibrated, 10:1 costs)**:
  - Expected loss 0.1729 vs F1-optimal 0.3027 (**42.9%** reduction on split)

These numbers are read from the JSON outputs listed at the top of this document.

---

## File Paths Used

- `results/H1_classification/H1_full_results.json`
- `results/H2_calibration_thresholds/H2_full_results.json`
- `results/H3_full_evaluation/H3_full_results.json`

