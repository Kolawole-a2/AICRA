# Hypothesis Experiments Guide

Canonical commands for H1, H2, and H3 in the AICRA praxis repository.

## Prerequisites

1. **EMBER-2024 data** in `data/ember2024_real/` (see `scripts/fetch_data.ps1` or `scripts/fetch_data.sh`)
2. **Dependencies:** `pip install -r requirements-dev.txt`
3. **Mappings:**
   - Deterministic: `data/mappings/deterministic_attack_defense_lookup.csv`
   - Learned: `data/mappings/learned_mapping.csv`

## Run all hypotheses

```bash
python scripts/run_all_hypotheses.py
```

## H1 — Static PE classification

```bash
python -m aicra.experiments.h1_classification \
  --output results/H1_classification \
  --model-type lgbm \
  --splits-config config/h1_splits.yaml
```

**Outputs:** `results/H1_classification/H1_full_results.json`, `H1_summary.md`

**Primary metric:** AUROC (multi-split + time-ordered train/test)

**Supplementary OOF evaluation (does not overwrite canonical H1):**

```bash
python scripts/evaluate_h1_oof_robust.py
```

Outputs: `results/H1_oof_robust_eval/`

## H2 — Calibration & cost-aware thresholding

Requires H1 probabilities. Run after H1.

```bash
python -m aicra.experiments.h2_calibration_thresholds \
  --output results/H2_calibration_thresholds \
  --splits-config config/h2_splits.yaml
```

**Outputs:** `results/H2_calibration_thresholds/H2_full_results.json`, `H2_summary.md`

**Primary metric:** Expected loss (cost-optimal vs F1-optimal threshold)

## H3 — Deterministic vs learned mapping

```bash
python -m aicra.experiments.h3_evaluation --config config/h3_splits.yaml
# or
python run_h3_evaluation.py
```

**Outputs:** `results/H3_full_evaluation/H3_full_results.json`, `H3_full_summary.md`

**Primary metric:** DAC_internal (deterministic vs learned)

**Secondary benchmark:** DAC_external vs `d3fend_reference_pairs.csv` (exported from `data/lookups/attack_to_d3fend.yaml`). External reference is a supplementary sanity check, not primary H3 ground truth.

**Three mappings compared in the H3 report:**
| Mapping | File | Role |
|---------|------|------|
| Deterministic | `data/mappings/deterministic_attack_defense_lookup.csv` | Primary ground truth (DAC_internal) |
| Learned | `data/mappings/learned_mapping.csv` | Alternative mapping under test |
| External reference | `d3fend_reference_pairs.csv` | Secondary benchmark (DAC_external) |

**Mapping comparison table (for praxis):**

```bash
python scripts/generate_h3_mapping_comparison.py
```

## Configuration files

| File | Purpose |
|------|---------|
| `config/h1_splits.yaml` | H1 evaluation slices (test-set nested splits) |
| `config/h2_splits.yaml` | H2 splits (aligned with H1) |
| `config/h3_splits.yaml` | H3 risk-score split paths |

## Validation report

```bash
python scripts/generate_praxis_validation_report.py
```

Or read: `results/praxis_validation_report.md`

## What not to use for canonical results

- Root-level `run_h3_*.py` scripts in `scripts/legacy/` — archived one-offs
- `scripts/h1h2_rebuild/` — optional operational rebuild, not primary H1/H2 validation
- Literature-based baseline percentages — use empirical baselines in saved JSON artifacts
