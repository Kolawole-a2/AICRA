# H3 Evaluation Pipeline

## Overview

The H3 evaluation pipeline is the canonical experiment for comparing deterministic vs learned ATT&CK–D3FEND mappings across all evaluation splits.

**Hypothesis (H3):** "Deterministic ATT&CK–D3FEND lookup yields higher risk-score precision and consistency than learned mapping."

## Quick Start

```bash
# Run H3 evaluation with default configuration
python run_h3_evaluation.py

# Or with custom paths
python run_h3_evaluation.py \
    --splits-config config/h3_splits.yaml \
    --deterministic data/mappings/deterministic_lookup.csv \
    --learned data/mappings/learned_mapping.csv \
    --reference d3fend_reference_pairs.csv \
    --output results/H3_full_evaluation
```

## Configuration

### Splits Configuration (`config/h3_splits.yaml`)

Define all evaluation splits in `config/h3_splits.yaml`:

```yaml
splits:
  time_test: "results/time_test/risk_scores.csv"
  oof_test: "results/oof_test/risk_scores.csv"
  seed1_time_test: "results/seed1/time_test/risk_scores.csv"
```

Each split must have a risk scores CSV file with at least:
- `asset_id`
- `risk_score` (calibrated p(ransomware) ∈ [0,1])
- `predicted_label` (1/0)
- `true_label` (1/0)
- `technique_id` (ATT&CK id for the sample)

### Input Files

1. **Deterministic Mapping** (`data/mappings/deterministic_lookup.csv`)
   - Columns: `technique_id` (or `attack_id`), `control_id` (or `defense_id`)
   - This is the authoritative, curated mapping

2. **Learned Mapping** (`data/mappings/learned_mapping.csv`)
   - Columns: `technique_id` (or `attack_id`), `control_id` (or `defense_id`)
   - Optionally: `similarity_score`

3. **Reference Pairs** (`d3fend_reference_pairs.csv`)
   - Columns: `technique_id`, `control_id`
   - Canonical MITRE D3FEND reference pairs

## Metrics Computed

For each evaluation split, the pipeline computes:

1. **Defense-Attack Consistency (DAC)   - Proportion of correctly aligned ATT&CK→D3FEND pairs
   - Computed separately for deterministic and learned mappings

2. **Coverage   - Percentage of techniques with at least one mapped control

3. **Actionable Precision   - Precision for actionable positives (predicted_label==1 AND mapping exists AND is canonical-consistent)

4. **Variance Reduction   - Reduction in risk score variance from applying mapping
   - Unmapped positives are demoted by a factor (default 0.90)

5. **Deltas   - Δ DAC = DAC_deterministic - DAC_learned
   - Δ Precision = Precision_deterministic - Precision_learned
   - Δ Variance Reduction = VarRed_deterministic - VarRed_learned

## Outputs

All outputs are saved to `results/H3_full_evaluation/`:

1. **`h3_results_by_split.csv`   - Detailed metrics for each split

2. **`h3_summary.json`   - Summary statistics across all splits

3. **`h3_report.md`   - Human-readable markdown report with tables and interpretation

## Module Structure

- **`aicra/experiments/h3_evaluation.py`**: Canonical H3 evaluation module
- **`run_h3_evaluation.py`**: Main entry point script
- **`config/h3_splits.yaml`**: Evaluation splits configuration

## Notes

- The pipeline automatically handles column name variations (`technique_id` vs `attack_id`, `control_id` vs `defense_id`)
- Missing splits are logged as warnings and skipped
- All metrics are computed per-split and then aggregated

## Integration with Other Modules

The H3 evaluation module uses:
- `aicra.metrics.dac`: DAC computation functions
- Standard sklearn metrics for precision calculation

Other H3-related modules (e.g., `h3_stat_tests.py`, `h3_prepare_metrics.py`) may be used for additional analysis but are not required for the core evaluation.
