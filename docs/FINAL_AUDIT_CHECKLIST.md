# Final Praxis Audit Checklist

**AICRA: Machine Learning-Based Cyber Risk Advisor with Analytics for Endpoint Ransomware Defense in U.S. Banking Organizations**

This checklist supports final submission review and examiner audit by verifying that all required components, validations, and documentation are present and correct.

---

## Data and Reproducibility

- ✅ **Data availability documented  - `docs/DATA.md` explains EMBER-2024 exclusion and setup instructions
  - `scripts/fetch_data.sh` and `scripts/fetch_data.ps1` provide data validation
  - `aicra/utils/data_paths.py` implements environment variable support

- ✅ **No raw datasets in Git  - `.gitignore` excludes `data/`, `*.jsonl`, and large binary files
  - `docs/DATA.md` documents exclusion rationale
  - `scripts/ci_guardrails.py` enforces exclusion in CI

- ✅ **Time-ordered evaluation implemented  - `aicra/core/data.py::load_ember_2024(time_ordered=True)` implements temporal splits
  - H1 experiment uses time-ordered splits (verified in `results/H1_classification/metrics.json`)
  - Temporal integrity verified: train max timestamp < test min timestamp

- ✅ **Out-of-family testing implemented  - H1 experiment evaluates on held-out malware families
  - Out-of-family metrics computed per-family and aggregated
  - Results stored in `results/H1_classification/metrics.json` (`oof_auroc`, `oof_pr_auc`)

---

## Experimental Design

- ✅ **Benchmarks defined  - `aicra/core/benchmarks.py` implements baseline computation for H1, H2, H3
  - Baseline values computed on canonical splits
  - Baseline methodology explained in `aicra/core/benchmarks.py` docstrings

- ✅ **% improvements reported  - H1: `results/H1_classification/metrics.json` includes `improvement` section (auroc_pct, precision_pct, etc.)
  - H2: `results/H2_calibration_thresholds/metrics.json` includes `improvement` section (brier_improvement_pct, ece_improvement_pct)
  - H3: `results/H3_full_evaluation/H3_full_results.json` includes `aggregated_metrics.improvements`
  - Consolidated report: `artifacts/benchmark_improvements.csv` and `.md`

- ✅ **Calibration validated temporally  - H2 experiment includes temporal calibration check
  - Calibration performed on earlier window (validation set), tested on later window (test set)
  - Results stored in `results/H2_calibration_thresholds/metrics.json` (`calibration.temporal_calibration_check`)

- ✅ **Mapping integrity checked  - H3 experiment includes learned == deterministic bug check
  - Fails with clear error if mappings are identical
  - Diagnostic report: `results/H3_full_evaluation/h3_mapping_integrity.json`

---

## Code Quality and CI

- ✅ **CI guardrails active  - `.github/workflows/ci.yml` includes `guardrails` job
  - `scripts/ci_guardrails.py` checks for large files, forbidden paths, unsafe patterns
  - CI runs on push and pull_request

- ✅ **Standardized entrypoints  - `experiments/h1_train_eval.py` — H1 standardized entrypoint
  - `experiments/h2_calibration_eval.py` — H2 standardized entrypoint
  - `experiments/h3_mapping_compare.py` — H3 standardized entrypoint
  - All use `get_ember2024_dir()` for data path resolution

---

## Results and Artifacts

- ✅ **Risk registers visible  - Small EMBER risk registers: `register/risk_register_small_ember.csv` and `.json`
  - Full EMBER derived outputs: `results/*/risk_scores.csv`
  - Documentation: `README.md` (Risk Register Outputs section) and `docs/EXPERIMENTS.md`

- ✅ **Results tables generated  - `docs/RESULTS_SUMMARY.md` includes H1, H2, H3 results tables
  - Tables include all required metrics (AUROC, Precision, Recall, F1, Brier, ECE, Coverage, DAC, etc.)
  - Interpretation text provided for each hypothesis

- ✅ **Benchmark improvements documented  - `artifacts/benchmark_improvements.csv` — Machine-readable table
  - `artifacts/benchmark_improvements.md` — Human-readable summary
  - Generated automatically after each experiment or manually via `python -m aicra.utils.benchmark_reporter`

---

## Documentation

- ✅ **Reproducibility documented  - `docs/EXPERIMENTS.md` provides step-by-step reproduction guide
  - `README.md` includes experiment commands and data availability section
  - `docs/DATA.md` explains data setup and exclusion rationale

- ✅ **Results interpretation provided  - `docs/RESULTS_SUMMARY.md` includes interpretation text for each hypothesis
  - Interpretation explains operational significance (banking SOC context)
  - Academic tone, precise, defensible

- ✅ **Threats to validity addressed  - `docs/THREATS_TO_VALIDITY.md` identifies internal, external, construct, and temporal validity threats
  - Each threat includes risk description and mitigation implemented in AICRA
  - Remaining risks explicitly acknowledged

- ✅ **Reviewer guide available  - `docs/REVIEWER_GUIDE.md` explains repository navigation, reproduction, and common questions
  - Answers "Why no raw EMBER data?", "How is alert fatigue measured?", etc.
  - Provides verification checklist

---

## Imbalanced Data Handling

- ✅ **Class imbalance strategies documented  - `README.md` (Imbalanced Data Handling section) explains strategies per experiment
  - H1: `class_weight="balanced"`, `scale_pos_weight` computed
  - H2: Calibration applied to handle distribution shift
  - H3: Variance reduction through deterministic mapping
  - Strategies logged in `experiment_metadata.json` and `metrics.json`

---

## Statistical Validation

- ✅ **Statistical tests performed  - H3: Paired t-tests, Wilcoxon tests, bootstrap confidence intervals
  - Results stored in `results/H3_full_evaluation/H3_full_results.json` (`aggregated_metrics.statistical_tests`)
  - P-values and confidence intervals reported

---

## File Integrity

- ✅ **File hashes stored  - H3 results include SHA256 hashes of mapping files
  - Stored in `results/H3_full_evaluation/H3_full_results.json` (`file_hashes`)
  - Enables verification of input file integrity

- ✅ **Experiment metadata logged  - Each experiment generates `experiment_metadata.json` with:
    - Timestamp, data directory, output directory
    - Model type, configuration parameters
    - Random seed (default: 42)

---

## Summary

This checklist verifies that the AICRA praxis includes:

1. ✅ Complete experimental design (time-ordered splits, out-of-family tests, baselines, % improvements)
2. ✅ Robust validation (temporal calibration, mapping integrity, statistical tests)
3. ✅ Comprehensive documentation (results, interpretation, validity, reproduction)
4. ✅ Code quality (CI guardrails, standardized entrypoints, imbalanced data handling)
5. ✅ Research artifacts (results tables, benchmark reports, reviewer guide)

**Status**: All checklist items verified and documented.

---

## Notes for Examiners

- **Raw Data**: EMBER-2024 JSONL files are excluded by design (see `docs/DATA.md`). Experiments can be reproduced locally with EMBER-2024 data.
- **Results**: All experimental results are stored in `results/` and `artifacts/` directories. See `docs/RESULTS_SUMMARY.md` for research-ready tables.
- **Reproducibility**: See `docs/EXPERIMENTS.md` for step-by-step reproduction instructions.
- **Validity**: See `docs/THREATS_TO_VALIDITY.md` for threats and mitigations.
- **Review**: See `docs/REVIEWER_GUIDE.md` for navigation and common questions.

--**This checklist is intended to support final submission review and examiner audit.