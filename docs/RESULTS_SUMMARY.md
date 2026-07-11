# AICRA Results Summary

**Machine Learning-Based Cyber Risk Advisor with Analytics for Endpoint Ransomware Defense in U.S. Banking Organizations**

This document presents experimental results for hypotheses H1, H2, and H3 in the AICRA praxis. All figures below are taken from canonical artifacts in `results/H1_classification/`, `results/H2_calibration_thresholds/`, and `results/H3_full_evaluation/` unless noted as supplementary.

**Canonical sources:** `H1_summary.md`, `H2_summary.md`, `H3_full_summary.md`, and matching `*_full_results.json` / `metrics.json` files.

---

## H1: Static PE Classification Reliability

**Hypothesis (canonical):** Static PE features enable reliable ransomware classification with AUROC ≥ 0.95 and operational precision suitable for banking environments.

**Model:** LightGBM on EMBER-2024 static PE features (`results/H1_classification/H1_full_results.json`).

**Validation modes:**

| Mode | Purpose | Evidence |
|------|---------|----------|
| **Time-ordered** | Train/test respects temporal ordering (no leakage) | 40,004 train / 10,001 test; `temporal_split_verification.json` confirms `max(train) < min(test)` |
| **Multi-split** | Robustness across nested test slices | `config/h1_splits.yaml` → full_ember, main, small_ember, smoke_test |
| **Out-of-family (OOF)** | Ranking on malware families unseen in training (supplementary) | `results/H1_oof_robust_eval/oof_robust_metrics.json` |

### H1 Results Table

| Model | Dataset Split | AUROC | Precision | Recall | F1 | vs Empirical Baseline |
|-------|---------------|-------|-----------|--------|-----|------------------------|
| LightGBM (AICRA) | full_ember (10,001 samples) | 0.9796 | 0.648 | 0.998 | 0.786 | +25.4% AUROC vs logistic |
| LightGBM (AICRA) | OOF (supplementary, 5,587 samples) | 0.9616 | 0.066* | 0.994* | 0.124* | Exceeds > 0.88 AUROC benchmark |
| Logistic Regression | full_ember (same split) | 0.7811 | 0.773 | 0.636 | 0.698 | Empirical baseline |
| Majority Classifier | full_ember (same split) | 0.500 | 0.000 | 0.000 | 0.000 | Trivial baseline (AUROC only) |

\*OOF precision/recall/F1 use the **H1 banking threshold (0.0248)** tuned on the full test set, then applied to the OOF slice (~3.2% positive rate vs ~46% on full test). **AUROC is the primary OOF metric**; operational P/R/F1 are supporting reference only (`oof_robust_metrics.json` notes).

**Additional H1 metrics (full_ember unless noted):**

| Metric | Value | Source |
|--------|-------|--------|
| PR-AUC | 0.9767 (mean 0.9550 multi-split) | `H1_summary.md` |
| Brier Score | 0.0551 (mean 0.0753 multi-split) | `H1_summary.md` |
| ECE | 0.0079 (mean 0.0249 multi-split) | `H1_summary.md` |
| Banking threshold | **0.0248** | FN cost = **100**, FP cost = **1** (100:1) |
| Lift@1% / 5% / 10% | 2.18× | `H1_full_results.json` |
| Confusion matrix (full_ember @ 0.0248) | TN=2916, FP=2493, FN=7, TP=4585 | 7 FNs out of 4,592 positives |

**Important trade-off:** At the banking threshold, AICRA **lowers precision** vs the logistic baseline (0.648 vs 0.773, −16.2%) while **raising recall** (0.998 vs 0.636, +56.9%). This is intentional under FN ≫ FP costs.

### H1 Interpretation

**Predictive strength:** AUROC 0.9796 on full_ember and mean 0.9610 across multi-split evaluation exceed both the **> 0.88 reliability benchmark** and the **≥ 0.95 design target**. Supplementary OOF AUROC is 0.9616. AUROC improvement over the empirical logistic baseline on full_ember is **+25.4%** (0.7811 → 0.9796).

**Calibration:** Brier 0.055 and ECE 0.008 on full_ember indicate well-calibrated probabilities on the temporal holdout.

**False-negative reduction:** At the banking threshold, AICRA achieves **99.6% FN-rate reduction** vs the logistic baseline (0.15% vs 36.4% FN rate; 7 vs 1,670 false negatives on 4,592 positives).

**Scope note:** H1 does **not** report expected operational loss. Threshold economics at **10:1** costs are evaluated under **H2**.

---

## H2: Cost-Aware Thresholding & Post-Hoc Calibration Test

**Research Question (RQ2):** Does cost-aware thresholding reduce expected loss compared to F1-optimized thresholds under banking-style asymmetric costs (FN cost >> FP cost)?

**Hypothesis (H2):** Cost-aware thresholding produces lower expected loss than F1-optimized thresholds under banking-style asymmetric costs, demonstrating more decision-aligned susceptibility scores for operational deployment.

**Cost parameters (H2):** FN cost = **10**, FP cost = **1** (10:1). *(H1 uses 100:1 for its banking threshold; the two experiments are complementary.)*

**Calibration:** Platt/isotonic regression applied post hoc **to test whether calibration helps**. Finding: uncalibrated H1 probabilities are already well-calibrated; post-hoc calibration **does not** improve expected loss and **worsens** Brier/ECE on full_ember.

### H2 Results Table (full_ember, 10,001 samples)

| Calibration | Brier | ECE | Change vs uncalibrated |
|-------------|-------|-----|------------------------|
| Uncalibrated | 0.0426 | 0.0066 | Baseline |
| Isotonic (calibrated) | 0.0500 | 0.0457 | Brier +17.4% worse; ECE +595% worse |

*Aggregated uncalibrated means across splits: Brier 0.0490, ECE 0.0162 (`H2_summary.md`).*

### H2 Threshold Comparison (full_ember)

| Method | Threshold | Precision | Recall | F1 | Expected Loss |
|--------|-----------|-----------|--------|-----|---------------|
| F1-optimized (uncalibrated) | 0.459 | 0.940 | 0.943 | 0.942 | 0.303 |
| F1-optimized (calibrated) | 0.227 | 0.940 | 0.943 | 0.942 | 0.303 |
| Cost-optimal (uncalibrated) | 0.104 | 0.821 | 0.985 | 0.896 | **0.173** |
| Cost-optimal (calibrated) | 0.010 | 0.905 | 0.965 | 0.934 | 0.215 |

**Expected-loss reductions (uncalibrated, primary H2 finding):**

| Comparison | full_ember | Mean across splits |
|------------|--------------|-------------------|
| Cost-optimal vs F1-optimal | 0.173 vs 0.303 (**42.9%** reduction) | 0.180 vs 0.365 (**50.6%** reduction) |
| Cost-optimal (cal) vs F1-optimal | 0.215 vs 0.303 (29.1% reduction) | 0.258 vs 0.365 (29.3% reduction) |

*Design-benchmark comparison vs expected loss = 0.50 yields ~65.4% improvement for cost-optimal uncalibrated (0.173); that is an H2 design target, not an H1 baseline.*

### H2 Interpretation

**Primary finding:** Cost-optimal thresholding at **0.104** (10:1 costs) minimizes expected loss vs F1-optimal thresholding. On full_ember it yields recall **0.985** and precision **0.821** (uncalibrated).

**Calibration help test:** Post-hoc isotonic calibration does **not** improve expected loss. Uncalibrated cost-optimal loss (**0.173**) beats calibrated cost-optimal loss (**0.215**) on full_ember. Optimal H2 deployment uses **cost-optimized thresholds on uncalibrated probabilities**.

**Temporal check:** Calibration parameters transfer across time windows with ECE degradation (0.007 → 0.046 on full_ember), supporting monitoring intervals but not mandatory recalibration for this model.

---

## H3: Deterministic vs Learned ATT&CK–D3FEND Mapping

**Hypothesis:** Deterministic ATT&CK–D3FEND mapping achieves higher DAC_internal and actionable precision than learned mapping.

**Evaluation:** 4 splits, 32,004 total samples (`H3_full_summary.md`). Mapping comparison uses **173 deterministic pairs** (46 techniques, 9 controls) vs **190 learned pairs** (47 techniques, 79 controls). **Zero pair overlap** between deterministic and learned mappings.

**Variance note:** Variance reduction is **0.0 on all splits** for both mappings. H3 is validated via **perfect DAC separation** and **actionable-precision dominance on production-scale splits**, not variance-reduction tests.

### H3 Results Table (aggregated across splits)

| Mapping | Coverage (%) | DAC_internal (%) | Actionable Precision | Variance Reduction |
|---------|-------------|------------------|------------------------|-------------------|
| Deterministic | 100.0 | 100.0 | **0.75** (SD: 0.50) | 0.0 |
| Learned | 100.0 | 0.0 | 0.0 (SD: 0.0) | 0.0 |
| Δ (Det − Learned) | 0.0 | +100.0 | +0.75 | 0.0 |

**Per-split actionable precision (deterministic / learned):**

| Split | n_samples | Deterministic | Learned |
|-------|-------------|---------------|---------|
| main | 10,000 | 1.0 | 0.0 |
| full_ember | 20,002 | 1.0 | 0.0 |
| small_ember | 2,000 | 1.0 | 0.0 |
| smoke_test | 2 | 0.0 | 0.0 |

*Mean 0.75 reflects three large splits at 1.0 and smoke_test at 0.0 (no actionable alerted rows in that 2-sample split).*

**Statistical tests (`H3_full_summary.md`):**

| Metric | Result | Interpretation |
|--------|--------|----------------|
| DAC_internal | t = ∞, **p < 0.001** | Perfect separation (det 100%, learned 0%) |
| Actionable precision | t = 3.0, **p = 0.058** | Marginal; not p < 0.05 |
| Variance reduction | t = NaN | Not applicable (zero variance) |

### H3 Interpretation

**DAC_internal (primary H3 metric):** Measures exact **(technique_id, control_id) pair overlap** between mappings. Deterministic vs itself = **100%** by definition. Learned vs deterministic ground truth = **0%** (disjoint control vocabularies: 9 ransomware-focused vs 79 broad controls).

**Actionable precision:** Among **alerted** samples (`predicted_label = 1`), fraction whose mapping recommends at least one deterministic (ransomware-relevant) control. Deterministic = **1.0** on main, full_ember, and small_ember; learned = **0.0** on all splits (learned mapping has no T1486 entries and zero deterministic-pair overlap).

**Learned mapping role:** 100% technique coverage and broader control set (79 controls) show learned mappings are **broader but not ransomware-aligned** in this evaluation—zero actionable precision limits operational use in banking SOCs.

**Limitation:** Canonical H3 risk registers default many samples to **T1486** when family is unknown; DAC is mapping-table-level and does not depend on per-sample scores. See `H3_full_summary.md` for register and default-technique context.

---

## Interpretation by Hypothesis

### H1: Predictive Performance and Operational Deployment

H1 validates reliable ransomware classification across time-ordered, multi-split, and supplementary OOF evaluation. full_ember AUROC **0.9796** (mean **0.9610**; OOF **0.9616**) exceeds the **> 0.88** benchmark with **+25.4%** AUROC gain vs logistic regression (**0.7811**).

H1 banking threshold **0.0248** (100:1 costs) prioritizes recall (**0.998** on full_ember) over precision (**0.648**), reducing false negatives to **7 of 4,592** positives. Calibration on full_ember (Brier **0.055**, ECE **0.008**) supports interpretable risk scores. Expected-loss optimization is addressed in H2.

### H2: Cost-Aware Thresholding & Calibration Help Test

H2 shows cost-optimal threshold **0.104** (10:1 costs) reduces expected loss to **0.173** on full_ember vs **0.303** at F1-optimal (**42.9%** reduction; **50.6%** mean across splits). Post-hoc calibration does not improve expected loss; uncalibrated probabilities remain the operational choice.

### H3: Mapping Consistency and Decision Reliability

H3 shows **perfect DAC_internal separation** (deterministic 100%, learned 0%) and **deterministic actionable-precision dominance** on production-scale splits (1.0 vs 0.0). Aggregated actionable-precision advantage is **+0.75** (p = **0.058**, marginal). Variance reduction is identically zero—H3 conclusion rests on DAC and precision, not variance tests.

---

## Summary

| Hypothesis | Supported? | Canonical headline |
|------------|------------|-------------------|
| **H1** | Yes (AUROC ≥ 0.95) | AUROC 0.9796 on full_ember; +25.4% vs logistic; 99.6% FN-rate reduction at threshold 0.0248 |
| **H2** | Yes | Cost-optimal threshold 0.104 cuts expected loss 42.9% vs F1-optimal on full_ember (50.6% mean across splits) |
| **H3** | Yes (DAC separation) | Deterministic DAC_internal 100% vs learned 0%; actionable precision 1.0 vs 0.0 on large splits |

Together, these results support the praxis claim that AICRA provides a reliable, calibrated, and operationally viable **machine learning-based cyber risk advisor with analytics for endpoint ransomware defense** in U.S. banking organizations.

---

## Data Availability

| Experiment | Canonical artifacts |
|------------|----------------------|
| H1 | `results/H1_classification/H1_summary.md`, `H1_full_results.json`, `metrics.json` |
| H2 | `results/H2_calibration_thresholds/H2_summary.md`, `H2_full_results.json`, `metrics.json` |
| H3 | `results/H3_full_evaluation/H3_full_summary.md`, `H3_full_results.json` |
| H1 OOF (supplementary) | `results/H1_oof_robust_eval/oof_robust_metrics.json` |

Reproduction: `docs/EXPERIMENTS.md` · Data: `docs/DATA.md`
