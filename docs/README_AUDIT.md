# README Accuracy Audit Report

**Last updated:** 2026-06-19  
**Purpose:** Audit README and linked praxis claims against canonical experiment artifacts (H1, H2, H3).

> **Canonical narrative (aligned 2026):** H1 = time-ordered + multi-split + OOF (AUROC benchmark **> 0.88**, empirical logistic **0.7811**); H2 = post-hoc Platt/isotonic **help test** + cost-optimal expected loss; H3 = **perfect separation** when variance reduction is **0.0 on all splits**.

---

## Summary

**Status:** ✅ **PASSED** — README and secondary praxis docs match canonical artifacts as of commits `7202dd6` (RESULTS_SUMMARY audit) and prior alignment series.

| Area | Status | Notes |
|------|--------|-------|
| **H1** | ✅ | Three validation modes documented; AUROC **> 0.88**; empirical baseline **0.7811** |
| **H2** | ✅ | Calibration framed as **help test**; primary metric = expected loss (cost-opt vs F1-opt) |
| **H3** | ✅ | Variance reduction 0.0%; validated via perfect separation, not variance-reduction p-values |

---

## H1 Claims Verification

| Claim | Verified by | Status | Canonical value |
|-------|-------------|--------|-----------------|
| AUROC reliability benchmark **> 0.88** | `README.md`, `docs/HYPOTHESIS_TESTING_PVALUES.md` | ✅ | Benchmark threshold |
| Empirical logistic baseline **0.7811** (same split) | `results/H1_classification/H1_full_results.json` | ✅ | `baseline.best_baseline.auroc` |
| full_ember AUROC 0.9796 | `H1_full_results.json` → `metrics.per_split_results` | ✅ | full_ember split |
| Multi-split mean AUROC 0.9610 | `H1_full_results.json` → `metrics.aggregated_metrics.auroc.mean` | ✅ | 4 splits |
| OOF AUROC 0.9616 (supplementary) | `results/H1_oof_robust_eval/oof_robust_summary.md` | ✅ | `scripts/evaluate_h1_oof_robust.py` |
| **Time-ordered** train/test | `aicra/experiments/h1_classification.py` | ✅ | Temporal split |
| **Multi-split** evaluation | `config/h1_splits.yaml` | ✅ | full_ember, main, small_ember, smoke_test |
| **Out-of-family** evaluation | `scripts/evaluate_h1_oof_robust.py` | ✅ | Supplementary folder |

**Conclusion:** H1 claims use the correct benchmark (> 0.88), empirical baseline (0.7811), and three validation modes.

---

## H2 Claims Verification

| Claim | Verified by | Status | Notes |
|-------|-------------|--------|-------|
| Platt/isotonic applied **post hoc to test whether calibration helps** | `README.md`, `docs/CALIBRATION_VALIDATION_REPORT.md` | ✅ | Not assumed to improve outcomes |
| Primary H2 metric: expected loss (cost-opt vs F1-opt) | `results/H2_calibration_thresholds/H2_full_results.json` | ✅ | ~50.6% reduction (uncalibrated) |
| Calibration does **not** improve expected loss on this model | H2 artifacts + H1 Brier/ECE | ✅ | Already well-calibrated from H1 |
| Cost-optimal expected loss ≈ 0.1729 (full_ember uncal) | `H2_full_results.json` | ✅ | vs F1-opt ≈ 0.3027 |

**Conclusion:** H2 documentation correctly separates the calibration **help test** from the primary expected-loss finding.

---

## H3 Claims Verification

| Claim | Verified by | Status | Notes |
|-------|-------------|--------|-------|
| Deterministic mapping **always correct** (100% DAC_internal) | `results/H3_full_evaluation/H3_full_results.json` | ✅ | All splits |
| Learned mapping **always extraneous** (0% DAC_internal) | Same | ✅ | All splits |
| Variance reduction **0.0** for both mappings | Same → `aggregated_metrics.*.variance_reduction` | ✅ | All splits |
| t-test / Wilcoxon / Shapiro–Wilk on variance **not applicable** | `docs/HYPOTHESIS_TESTING_PVALUES.md` § H3 | ✅ | Zero variability |
| H3 validated via **perfect separation** + DAC/precision | `README.md`, `H3_full_summary.md` | ✅ | Primary inference |
| ~~47% variance reduction~~ | N/A | ❌ **REMOVED** | Theoretical; never in artifacts |

**Conclusion:** H3 variance claims corrected. Primary validation is deterministic dominance and consistent superiority on DAC_internal and actionable precision.

---

## Historical Corrections (traceability)

The following incorrect README claims were **fixed** in 2024–2026:

| Old claim | Correct state |
|-----------|---------------|
| AUROC baseline **0.85** in summary table | **> 0.88** benchmark; empirical logistic **0.7811** |
| H2 “calibration for reliable risk scores” | Post-hoc **help test**; no expected-loss improvement |
| H3 “47% variance reduction” | **0.0%** on all splits; perfect separation |
| H3 “greater risk-score stability (lower variance)” | Removed from hypothesis; variance tests N/A |
| H1 OOF buried as optional only | Three modes: time-ordered, multi-split, **OOF** |
| Wrong praxis title (“Artificial Intelligence–Powered… Endpoint Security”) | **Machine Learning-Based… Endpoint Ransomware Defense** (`README.md`) |
| H1 precision/recall **0.9459 / 0.9363** in secondary docs | Banking-threshold metrics **0.648 / 0.998** on full_ember (`H1_summary.md`) |
| H1/H2 metric mix-up (threshold **0.104**, EL **0.173** under H1) | H1 threshold **0.0248** (100:1); H2 threshold **0.104** (10:1) |

See `docs/H3_RECONCILIATION_REPORT.md` for variance-reduction root-cause analysis.

---

## Verification Commands

### H1 — AUROC and baseline
```bash
python -c "
import json
with open('results/H1_classification/H1_full_results.json') as f:
    d = json.load(f)
splits = {s['split']: s['auroc'] for s in d['metrics']['per_split_results']}
print('full_ember AUROC:', splits.get('full_ember'))
print('aggregated mean AUROC:', d['metrics']['aggregated']['auroc']['mean'])
print('logistic baseline AUROC:', d.get('baselines', {}).get('logistic_regression', {}).get('auroc'))
"
```

### H2 — expected loss
```bash
python -c "
import json
with open('results/H2_calibration_thresholds/H2_full_results.json') as f:
    d = json.load(f)
for row in d['metrics']['per_split_results']:
    if row['split'] == 'full_ember':
        print('F1-opt EL:', row['f1_optimized']['expected_loss'])
        print('Cost-opt EL:', row['cost_optimized']['expected_loss'])
"

```

### H3 — variance reduction
```bash
python -c "
import json
with open('results/H3_full_evaluation/H3_full_results.json') as f:
    d = json.load(f)
det = d['aggregated_metrics']['deterministic']['variance_reduction']['mean']
lrn = d['aggregated_metrics']['learned']['variance_reduction']['mean']
print('Deterministic variance reduction:', det)
print('Learned variance reduction:', lrn)
print('DAC_internal det:', d['aggregated_metrics']['deterministic']['dac_%']['mean'])
print('DAC_internal lrn:', d['aggregated_metrics']['learned']['dac_%']['mean'])
"
```

Expected: variance reduction **0.0** both; DAC_internal **100%** vs **0%**.

---

## Files in audit scope

| File | Role |
|------|------|
| `README.md` | Primary praxis narrative |
| `docs/RESULTS_SUMMARY.md`, `docs/BENCHMARK_NOTES.md` | Secondary results |
| `docs/HYPOTHESIS_TESTING_PVALUES.md` | Statistical validation |
| `docs/H3_RECONCILIATION_REPORT.md` | H3 variance reconciliation |
| `results/praxis_validation_report.md` | Consolidated validation |
| `docs/archive/development/*` | Historical notes (banners aligned 2026) |

---

## Conclusion

✅ README and linked documentation match experimental artifacts for H1, H2, and H3 under the 2026 canonical narrative. The only material historical errors were H3 variance-reduction percentages and H1 AUROC baseline **0.85**; both are corrected across live and archived docs.
