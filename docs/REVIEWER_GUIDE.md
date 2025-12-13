# Reviewer and Examiner Guide

**AICRA: Artificial Intelligence–Powered Cyber Risk Advisor for Endpoint Security in U.S. Banking Organizations**

This guide assists reviewers and examiners in navigating the AICRA repository, understanding experimental design, and reproducing results.

---

## How to Navigate the Repository

### Code Structure

**Experiment Entry Points**:
- `experiments/h1_train_eval.py` — H1: Static PE Classification
- `experiments/h2_calibration_eval.py` — H2: Calibration and Thresholding
- `experiments/h3_mapping_compare.py` — H3: Mapping Comparison

**Core Implementation**:
- `aicra/experiments/h1_classification.py` — H1 experiment logic
- `aicra/experiments/h2_calibration_thresholds.py` — H2 experiment logic
- `aicra/experiments/h3_evaluation.py` — H3 experiment logic
- `aicra/core/benchmarks.py` — Baseline computation and % improvement calculations
- `aicra/utils/data_paths.py` — Data path management (uses `AICRA_EMBER2024_DIR`)

**Supporting Modules**:
- `aicra/pipelines/training.py` — Model training pipeline
- `aicra/pipelines/calibration.py` — Calibration pipeline
- `aicra/core/data.py` — Data loading with time-ordered splits
- `aicra/utils/benchmark_reporter.py` — Consolidated benchmark report generation

### Results Storage

**H1 Results**:
- `results/H1_classification/metrics.json` — Complete metrics (AUROC, Precision, Recall, F1, baselines, improvements)
- `results/H1_classification/summary.md` — Human-readable summary
- `artifacts/H1_classification/` — Alternative output location (if using standardized entrypoints)

**H2 Results**:
- `results/H2_calibration_thresholds/metrics.json` — Complete metrics (Brier, ECE, thresholds, improvements)
- `results/H2_calibration_thresholds/summary.md` — Human-readable summary
- `artifacts/H2_calibration_thresholds/` — Alternative output location

**H3 Results**:
- `results/H3_full_evaluation/H3_full_results.json` — Complete results (per-split and aggregated metrics)
- `results/H3_full_evaluation/H3_full_summary.md` — Comprehensive markdown report
- `results/H3_full_evaluation/h3_mapping_integrity.json` — Mapping integrity check (learned vs deterministic)
- `results/H3_full_evaluation/plots/` — Visualization plots

**Risk Registers**:
- `register/risk_register_small_ember.csv` and `.json` — Small EMBER risk register (included in Git)
- `register/risk_register_full.csv` and `.json` — Full EMBER risk register (if generated)
- `results/*/risk_scores.csv` — Risk scores per evaluation split (used by H3)

**Benchmark Reports**:
- `artifacts/benchmark_improvements.csv` — Machine-readable table of all % improvements
- `artifacts/benchmark_improvements.md` — Human-readable summary

### Documentation

**Primary Documentation**:
- `README.md` — Main repository documentation
- `docs/EXPERIMENTS.md` — Step-by-step reproduction guide
- `docs/DATA.md` — Data availability and exclusion rationale
- `docs/RESULTS_SUMMARY.md` — Research-ready results tables and interpretation
- `docs/THREATS_TO_VALIDITY.md` — Validity threats and mitigations
- `docs/REVIEWER_GUIDE.md` — This document

**Supporting Documentation**:
- `HYPOTHESIS_EXPERIMENTS_GUIDE.md` — Detailed experiment guide
- `docs/DATA.md` — Data management policy

---

## How to Reproduce Results

### Prerequisites

1. **Data Setup**: EMBER-2024 dataset must be available locally
   ```bash
   # Check data availability
   bash scripts/fetch_data.sh  # Linux/Mac
   .\scripts\fetch_data.ps1    # Windows
   
   # Or set environment variable
   export AICRA_EMBER2024_DIR=/path/to/ember2024_real  # Linux/Mac
   $env:AICRA_EMBER2024_DIR = "C:\path\to\ember2024_real"  # Windows
   ```

2. **Dependencies**: Install requirements
   ```bash
   pip install -r requirements-dev.txt
   ```

### Running Experiments

**H1: Static PE Classification**
```bash
# Standardized entrypoint (recommended)
python experiments/h1_train_eval.py

# Expected outputs:
# - artifacts/H1_classification/metrics.json
# - artifacts/H1_classification/summary.md
# - artifacts/models/h1_lgbm.joblib (trained model)
```

**H2: Calibration and Thresholding**
```bash
# Requires H1 to be run first (uses trained model)
python experiments/h2_calibration_eval.py

# Expected outputs:
# - artifacts/H2_calibration_thresholds/metrics.json
# - artifacts/H2_calibration_thresholds/summary.md
```

**H3: Mapping Comparison**
```bash
# Requires H1/H2 risk scores (stored in results/*/risk_scores.csv)
python experiments/h3_mapping_compare.py

# Expected outputs:
# - artifacts/H3_full_evaluation/H3_full_results.json
# - artifacts/H3_full_evaluation/H3_full_summary.md
# - artifacts/H3_full_evaluation/h3_mapping_integrity.json
```

**Generate Benchmark Report**
```bash
# Automatically generated after each experiment, or manually:
python -m aicra.utils.benchmark_reporter

# Expected outputs:
# - artifacts/benchmark_improvements.csv
# - artifacts/benchmark_improvements.md
```

### Expected Outputs

Each experiment produces:
1. **JSON metrics file** — Complete numerical results
2. **Markdown summary** — Human-readable interpretation
3. **Experiment metadata** — Seeds, configs, timestamps (in `experiment_metadata.json`)

### Notes on Excluded Raw Data

**Raw EMBER JSONL files are NOT included in Git** due to:
- Size constraints (~30GB dataset)
- Licensing considerations
- Repository hygiene best practices

**What IS included**:
- Small EMBER risk registers (`register/risk_register_small_ember.csv`)
- Derived artifacts (risk scores, metrics, diagnostics)
- Complete code and configuration

**To obtain raw data**:
1. See `docs/DATA.md` for data availability instructions
2. Run `scripts/fetch_data.sh` or `scripts/fetch_data.ps1` for setup guidance
3. Place EMBER-2024 JSONL files in `data/ember2024_real/` or set `AICRA_EMBER2024_DIR`

---

## Common Reviewer Questions (Answered)

### "Why no raw EMBER data?"

**Answer**: Raw EMBER-2024 JSONL files (~30GB) are excluded from Git for:
1. **Size**: Exceeds GitHub's practical limits and would make the repository unusable
2. **Licensing**: EMBER dataset has specific licensing terms requiring separate distribution
3. **Best Practices**: Large datasets should be stored externally (S3, Google Drive) rather than in version control

**What IS included**:
- Small EMBER risk registers demonstrating end-to-end pipeline correctness
- Derived artifacts (risk scores, metrics) enabling H1–H3 evaluation
- Complete code and configuration for full reproducibility

**To verify results**: Run experiments locally with EMBER-2024 data (see `docs/EXPERIMENTS.md` for instructions).

**Reference**: See `docs/DATA.md` for complete data policy and `docs/RESULTS_SUMMARY.md` for results interpretation.

---

### "How is alert fatigue measured?"

**Answer**: Alert fatigue is measured through multiple proxy measures:

1. **False Negative Reduction**: Fewer missed threats = less retroactive investigation
   - Computed as: `100 * (baseline_fn - aicra_fn) / baseline_fn`
   - Stored in: `metrics.alert_fatigue_reduction.fn_reduction_pct`

2. **Estimated Analyst Fatigue Reduction**: Correlates FN reduction with fatigue
   - Computed as: `fn_reduction_pct * 0.8` (assumes 80% correlation)
   - Stored in: `metrics.alert_fatigue_reduction.estimated_analyst_fatigue_reduction_pct`

3. **Variance Reduction**: More consistent scores = less cognitive load
   - Computed as: `100 * (learned_variance - deterministic_variance) / learned_variance`
   - Stored in: `aggregated_metrics.improvements.variance_reduction_pct`

4. **Expected Loss Reduction**: Fewer unnecessary investigations = less analyst time
   - Computed as: `100 * (baseline_loss - aicra_loss) / baseline_loss`
   - Stored in: `metrics.cost_optimized.expected_loss`

**Limitation**: True alert fatigue requires longitudinal studies with real SOC analysts. Proxy measures are operationally meaningful and defensible for praxis validation.

**Reference**: See `docs/THREATS_TO_VALIDITY.md` (Section 3.1) for construct validity discussion.

---

### "Why deterministic mapping?"

**Answer**: Deterministic mapping is used because:

1. **Operational Reliability**: 100% Defense-Attack Consistency (DAC) ensures that recommended countermeasures are appropriate for detected attack techniques, which is critical for banking SOCs where analyst trust and regulatory compliance are paramount.

2. **Expert Knowledge**: Based on MITRE ATT&CK and D3FEND, industry-standard, expert-curated frameworks validated across cybersecurity domains.

3. **Experimental Control**: Provides ground truth for H3 evaluation, enabling comparison with learned mappings to assess whether embedding-based heuristics can match expert knowledge.

4. **H3 Results**: Deterministic mapping achieves 100% DAC and 0.75 actionable precision, while learned mapping achieves 0% DAC and 0.0 actionable precision, validating that deterministic mapping is essential for operational deployment.

**Why learned mapping still matters**: Learned mapping achieves 100% coverage and uses a broader set of controls (79 vs 9), suggesting it may discover alternative recommendations. However, 0% actionable precision makes it unsuitable for operational use in banking contexts.

**Reference**: See `docs/RESULTS_SUMMARY.md` (H3 Interpretation) and `docs/THREATS_TO_VALIDITY.md` (Section 3.3).

---

### "How is robustness addressed?"

**Answer**: Robustness is addressed through multiple mechanisms:

1. **Time-Ordered Splits**: Test data is chronologically later than training data, simulating real-world deployment where models must generalize to future threats.

2. **Out-of-Family Evaluation**: Tests generalization to malware families not seen during training, simulating zero-day threats.

3. **Temporal Calibration Check**: Validates that calibration parameters transfer across time periods, ensuring score reliability as threat landscape evolves.

4. **Ensemble Methods**: Bagged LightGBM (multiple models with different seeds) provides robustness to feature manipulation and distribution shift.

5. **Baseline Comparisons**: Logistic regression and majority classifier baselines provide context for interpreting absolute performance and detecting overfitting.

6. **Statistical Tests**: Paired t-tests, Wilcoxon tests, and bootstrap confidence intervals validate that improvements are statistically significant.

7. **Imbalanced Data Handling**: Class weights (`class_weight="balanced"`) and positive class weighting (`scale_pos_weight`) ensure robust performance on imbalanced ransomware datasets.

**Limitations**: Advanced adversarial techniques may still evade detection, but this is a limitation of static analysis in general, not specific to AICRA. AICRA is designed as part of a defense-in-depth strategy.

**Reference**: See `docs/THREATS_TO_VALIDITY.md` (Section 4) for temporal validity discussion and `README.md` (Imbalanced Data Handling) for technical details.

---

## Verification Checklist

To verify experimental results:

1. ✅ **Data Availability**: Check that EMBER-2024 data is available (run `scripts/fetch_data.sh`)
2. ✅ **Code Execution**: Run each experiment (`experiments/h1_train_eval.py`, etc.)
3. ✅ **Output Verification**: Check that JSON and markdown files are generated
4. ✅ **Metric Consistency**: Compare results in JSON files with `docs/RESULTS_SUMMARY.md`
5. ✅ **Reproducibility**: Verify that results are consistent across runs (fixed seeds)
6. ✅ **Baseline Comparisons**: Check that baseline metrics are computed and % improvements are reported
7. ✅ **Temporal Integrity**: Verify that time-ordered splits are used (check timestamps in data loading)
8. ✅ **Mapping Integrity**: Check `h3_mapping_integrity.json` to verify learned ≠ deterministic

---

## Additional Resources

- **Results Interpretation**: `docs/RESULTS_SUMMARY.md`
- **Validity Discussion**: `docs/THREATS_TO_VALIDITY.md`
- **Reproduction Guide**: `docs/EXPERIMENTS.md`
- **Data Policy**: `docs/DATA.md`
- **Audit Checklist**: `docs/FINAL_AUDIT_CHECKLIST.md`

---

## Contact

For questions or clarifications:
- Review repository documentation (`docs/`)
- Check experiment outputs (`results/`, `artifacts/`)
- See `README.md` for general information


