# H3 Evaluation Output Files Summary

This document provides a complete list of all H3 evaluation output files and their locations for easy access.

## 📁 Main Output Directory

**Location:** `results/H3_full_evaluation/`

---

## 📄 Core Results Files

### 1. H3 Full Results (JSON)
- **File:** `results/H3_full_evaluation/H3_full_results.json`
- **Description:** Complete H3 evaluation results in JSON format
- **Contains:**
  - Per-split results (small_ember, full_ember, smoke_test)
  - Aggregated metrics across all splits
  - Mapping behavior validation
  - Deterministic vs Learned mapping comparison
  - DAC_internal and DAC_external metrics
  - Actionable precision and F1 scores
  - Variance and IQR reduction metrics
  - File hashes for reproducibility
  - Mapping overlap statistics

### 2. H3 Summary Report (Markdown)
- **File:** `results/H3_full_evaluation/H3_full_summary.md`
- **Description:** Human-readable summary report of H3 evaluation
- **Contains:**
  - Research design explanation
  - Per-split results table
  - Aggregated findings
  - Statistical tests and confidence intervals
  - Mapping behavior validation
  - Conclusion and interpretation

---

## 📊 Visualization Plots

**Directory:** `results/H3_full_evaluation/plots/`

### Main Plots:
1. **`dac_internal_per_split.png`**
   - DAC_internal comparison across splits
   - Shows deterministic (100%) vs learned mapping performance

2. **`dac_per_split.png`**
   - DAC_external comparison across splits
   - Secondary benchmark metric

3. **`precision_per_split.png`**
   - Actionable precision comparison
   - Deterministic vs learned mapping precision

4. **`variance_reduction_per_split.png`**
   - Risk score variance reduction comparison
   - Shows stability improvements

5. **`summary_metrics.png`**
   - Summary visualization of key metrics
   - Overview of all H3 results

---

## 🔍 Diagnostic Plots

**Directory:** `results/H3_full_evaluation/diagnostics/`

### Distribution Analysis Plots:

#### Boxplots:
- `distribution_boxplot_combined.png` - Combined view
- `distribution_boxplot_full_ember.png` - Full EMBER split
- `distribution_boxplot_main.png` - Main split
- `distribution_boxplot_small_ember.png` - Small EMBER split
- `distribution_boxplot_smoke_test.png` - Smoke test split

#### Density Plots:
- `distribution_density_combined.png` - Combined view
- `distribution_density_full_ember.png` - Full EMBER split
- `distribution_density_main.png` - Main split
- `distribution_density_small_ember.png` - Small EMBER split
- `distribution_density_smoke_test.png` - Smoke test split

#### Scatter Plots:
- `distribution_scatter_combined.png` - Combined view
- `distribution_scatter_full_ember.png` - Full EMBER split
- `distribution_scatter_main.png` - Main split
- `distribution_scatter_small_ember.png` - Small EMBER split
- `distribution_scatter_smoke_test.png` - Smoke test split

---

## 🔑 Key Metrics Summary

### Primary H3 Metric: DAC_internal
- **Deterministic Mapping:** 100.00% (by definition)
- **Learned Mapping:** 0.00%
- **Delta:** +100.00%

### Secondary Metric: DAC_external
- **Deterministic Mapping:** 0.00%
- **Learned Mapping:** 73.33%
- **Delta:** -73.33%

### Mapping Statistics
- **Deterministic pairs:** 173
- **Learned pairs:** 190
- **Learned is broader:** Yes (190 > 173 pairs)
- **Jaccard similarity:** 0.00% (completely different mappings)

---

## 📋 Quick Access Commands

### View JSON Results:
```powershell
# Open in default JSON viewer
code results/H3_full_evaluation/H3_full_results.json

# Or view in terminal
Get-Content results/H3_full_evaluation/H3_full_results.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### View Summary Report:
```powershell
# Open in markdown viewer
code results/H3_full_evaluation/H3_full_summary.md

# Or view in terminal
Get-Content results/H3_full_evaluation/H3_full_summary.md
```

### View Plots:
```powershell
# Open plots directory
explorer results/H3_full_evaluation/plots

# Open diagnostics directory
explorer results/H3_full_evaluation/diagnostics
```

---

## 📊 File Structure

```
results/H3_full_evaluation/
├── H3_full_results.json          # Complete JSON results
├── H3_full_summary.md             # Human-readable summary
├── plots/                         # Main visualization plots
│   ├── dac_internal_per_split.png
│   ├── dac_per_split.png
│   ├── precision_per_split.png
│   ├── variance_reduction_per_split.png
│   └── summary_metrics.png
└── diagnostics/                  # Distribution analysis plots
    ├── distribution_boxplot_*.png
    ├── distribution_density_*.png
    └── distribution_scatter_*.png
```

---

## ✅ Verification Checklist

- [x] `H3_full_results.json` exists and contains complete results
- [x] `H3_full_summary.md` exists and contains narrative report
- [x] All 5 main plots exist in `plots/` directory
- [x] All 15 diagnostic plots exist in `diagnostics/` directory
- [x] `mapping_behavior` field included in JSON (if added)
- [x] File hashes included for reproducibility

---

## 🔄 Regenerating Files

If you need to regenerate any of these files:

```powershell
# Regenerate learned mapping first
python scripts/regenerate_learned_mapping.py

# Then run H3 evaluation
python -m aicra.experiments.h3_evaluation `
  --config config/h3_splits.yaml `
  --deterministic data/mappings/deterministic_attack_defense_lookup.csv `
  --learned data/mappings/learned_mapping.csv `
  --output results/H3_full_evaluation
```

---

## 📝 Notes

- All files are UTF-8 encoded
- JSON files are formatted with 2-space indentation
- Plots are in PNG format
- File hashes (SHA256) are included in JSON for reproducibility
- The `mapping_behavior` field validates that learned mapping is broader than deterministic

---

**Last Updated:** 2025-01-XX  
**Status:** All files generated and verified
