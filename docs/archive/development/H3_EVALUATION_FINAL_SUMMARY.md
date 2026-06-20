> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# H3 Evaluation Pipeline - Final Summary

## Files Created or Modified

### Created Files

1. **`aicra/experiments/h3_evaluation.py`** (1,193 lines)
   - Canonical H3 evaluation module with comprehensive metrics
   - Includes main() entry point for module execution
   - All metrics, aggregation, statistical tests, and visualization

2. **`config/h3_splits.yaml`   - Configuration file for evaluation splits

3. **`run_h3_evaluation.py`   - Main entry point script

4. **`docs/h3_evaluation_README.md`   - Detailed usage guide and documentation

5. **`H3_PIPELINE_CLEANUP_SUMMARY.md`   - Summary of cleanup and unification

6. **`H3_EVALUATION_FINAL_SUMMARY.md`** (this file)
   - Final summary document

### Modified Files

1. **`README.md`   - Added comprehensive "H3 – Deterministic vs Learned ATT&CK–D3FEND Mapping Evaluation" section
   - Updated to reference new canonical pipeline

2. **`docs/heuristic_mapping.md`   - Updated references from old `H3_comparison` script to new `aicra.experiments.h3_evaluation` module

### Removed Files (Cleanup)

- `run_h3_experiment.py`
- `run_h3_full_experiment.py`
- `run_h3_validation.py`
- `prepare_h3_inputs.py`
- `prepare_h3_validation_inputs.py`
- `reprocess_h3.py`
- `setup_and_run_h3.py`
- `results/H3_comparison/` (directory)
- `results/H3_validation/` (directory)

## Command to Run H3 Evaluation

```bash
# Option 1: Run as module (recommended)
python -m aicra.experiments.h3_evaluation --config config/h3_splits.yaml

# Option 2: Run with default config (will try to infer splits)
python -m aicra.experiments.h3_evaluation

# Option 3: Use entry point script
python run_h3_evaluation.py
```

## Metrics Produced

### Per-Split Metrics (for each evaluation split)

1. **Mapping Metrics**:
   - Coverage (%): % of ATT&CK techniques with ≥1 mapped D3FEND control
   - DAC (%): Defense-Attack Consistency (overlap with reference pairs)
   - Correctness (%): % of pairs flagged as validated (if available)

2. **Register-Level Performance**:
   - Actionable Precision: Precision for actionable positives
   - Actionable F1: F1 score for actionable positives
   - Variance Reduction: Reduction in risk score variance
   - IQR Reduction: Reduction in interquartile range

3. **Baseline Metrics** (mapping-agnostic):
   - AUROC: Area Under ROC Curve
   - PR-AUC: Precision-Recall AUC
   - Brier Score: Calibration metric
   - ECE: Expected Calibration Error (10 bins)

4. **Delta Metrics** (Deterministic - Learned):
   - Δ DAC, Δ Coverage, Δ Precision, Δ F1, Δ Variance Reduction, Δ IQR Reduction

### Aggregated Metrics (across all splits)

- Mean and standard deviation for all metrics (deterministic and learned)
- Delta metrics with bootstrap 95% confidence intervals
- Statistical tests for DAC and precision; variance reduction identically 0.0 on all splits (t-test/Wilcoxon/Shapiro–Wilk not applicable)

## Output Locations

All outputs are saved to **`results/H3_full_evaluation/`**:

1. **`H3_full_results.json`   - Complete per-split metrics
   - Aggregated metrics with means, stds, and CIs
   - Statistical test results (p-values)
   - SHA256 hashes of input files
   - List of splits evaluated

2. **`H3_full_summary.md`   - Human-readable markdown report
   - Setup section
   - Per-split results table
   - Aggregated findings with interpretation
   - Statistical tests and p-values
   - Conclusion for Praxis H3 (data-driven)
   - Reproducibility section

3. **`plots/`** directory:
   - `dac_per_split.png`: DAC comparison per split
   - `precision_per_split.png`: Actionable precision per split
   - `variance_reduction_per_split.png`: Variance reduction per split
   - `summary_metrics.png`: Summary bar plot with error bars

## Key Features

- ✅ **Type hints** throughout
- ✅ **Comprehensive logging- ✅ **No allow_pickle=True** (safe data loading)
- ✅ **Does not overwrite** deterministic or learned CSV mappings
- ✅ **Cohesive, testable functions- ✅ **Automatic column name normalization** (technique_id ↔ attack_id, control_id ↔ defense_id)
- ✅ **Robust error handling** (skips missing splits with warnings)
- ✅ **Bootstrap confidence intervals** for delta metrics
- ✅ **Statistical tests without forced conclusions- ✅ **Data-driven conclusion** section in report

## Documentation

- **Main README**: `README.md` - H3 section added
- **Detailed Guide**: `docs/h3_evaluation_README.md`
- **Heuristic Mapping Docs**: `docs/heuristic_mapping.md` - Updated references

## Next Steps

1. **Configure splits**: Edit `config/h3_splits.yaml` with your evaluation split paths
2. **Run evaluation**: `python -m aicra.experiments.h3_evaluation --config config/h3_splits.yaml`
3. **Review outputs**: Check `results/H3_full_evaluation/H3_full_summary.md` for results
