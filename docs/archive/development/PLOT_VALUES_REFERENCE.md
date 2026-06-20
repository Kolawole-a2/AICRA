> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# H3 Plot Values Reference

This document shows the exact values that should appear in each plot.

## Data Source
- JSON: `results/H3_full_evaluation/H3_full_results.json`
- Generated: Latest evaluation run

## Expected Plot Values

### 1. DAC Per Split Plot (`dac_per_split.png`)

| Split | Deterministic DAC | Learned DAC |
|-------|-------------------|------------|
| main | 100.00% | 0.00% |
| small_ember | 100.00% | 0.00% |
| full_ember | 100.00% | 0.00% |
| smoke_test | 100.00% | 0.00% |

### 2. Precision Per Split Plot (`precision_per_split.png`)

| Split | Deterministic Precision | Learned Precision |
|-------|-------------------------|-------------------|
| main | 1.0000 | 0.0000 |
| small_ember | 1.0000 | 0.0000 |
| full_ember | 1.0000 | 0.0000 |
| smoke_test | 0.0000 | 0.0000 |

### 3. Variance Reduction Per Split Plot (`variance_reduction_per_split.png`)

| Split | Deterministic Var Red | Learned Var Red |
|-------|----------------------|-----------------|
| main | 0.000000 | 0.000000 |
| small_ember | 0.000000 | 0.000000 |
| full_ember | 0.000000 | 0.000000 |
| smoke_test | 0.000000 | 0.000000 |

### 4. Summary Metrics Plot (`summary_metrics.png`)

**DAC:- Deterministic: 100.00% ± 0.00%
- Learned: 0.00% ± 0.00%

**Actionable Precision:- Deterministic: 0.7500 ± 0.5000
- Learned: 0.0000 ± 0.0000

**Variance Reduction:- Deterministic: 0.000000 ± 0.000000
- Learned: 0.000000 ± 0.000000

## Verification

To verify your plots match these values, run:

```powershell
python scripts/verify_plot_values.py
```

## Regenerating Plots

If plots show incorrect values:

1. **Close all plot files** in your viewer/IDE (they may be cached)
2. **Regenerate plots:   ```powershell
   python scripts/regenerate_and_verify_plots.py
   ```
3. **Reopen the plot files** from `results/H3_full_evaluation/plots/`

## Notes

- All plots use the single `dac_%` metric (not `dac_internal_%` or `dac_external_%`)
- Deterministic precision should be **higher** than learned precision (as expected)
- The old `dac_internal_per_split.png` file has been removed

