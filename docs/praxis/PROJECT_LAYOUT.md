# Repository Layout (Praxis)

```
AICRA/
├── README.md                 # Project overview & quick start
├── docs/
│   ├── praxis/               # ← Praxis documentation hub (start here)
│   ├── EXPERIMENTS.md        # Reproduction steps
│   ├── BENCHMARK_NOTES.md    # Metric snapshot
│   └── archive/              # Historical dev notes (not for defense narrative)
├── aicra/                    # Installable Python package
│   ├── experiments/          # H1, H2, H3 canonical experiment modules
│   ├── core/                 # Benchmarks, calibration helpers
│   ├── mappings/             # Mapping generation utilities
│   └── pipelines/            # Training, calibration, evaluation pipelines
├── config/                   # Split YAML (h1_splits, h2_splits, h3_splits)
├── data/
│   ├── ember2024_real/       # EMBER-2024 JSONL (not in git — fetch locally)
│   ├── mappings/             # Deterministic & learned mapping CSVs
│   └── lookups/              # ATT&CK / D3FEND lookup tables
├── results/                  # Canonical experiment outputs
│   ├── H1_classification/
│   ├── H2_calibration_thresholds/
│   ├── H3_full_evaluation/
│   ├── H1_oof_robust_eval/   # Supplementary OOF evaluation
│   └── praxis_validation_report.md
├── scripts/                  # Utilities & one-shot runners
│   ├── run_all_hypotheses.py
│   ├── evaluate_h1_oof_robust.py
│   ├── generate_h3_mapping_comparison.py
│   ├── h1h2_rebuild/         # Optional rebuild pipeline (not canonical H1/H2)
│   └── legacy/               # Archived one-off scripts from repo root
├── tests/                    # Unit tests
├── register/                 # Risk registers (operational artifacts)
└── run_h3_evaluation.py      # Thin wrapper for H3 (optional convenience)
```

## What is canonical?

- **Experiments:** `aicra/experiments/h1_classification.py`, `h2_calibration_thresholds.py`, `h3_evaluation.py`
- **Results:** `results/H1_classification/`, `results/H2_calibration_thresholds/`, `results/H3_full_evaluation/`
- **Mappings:** `data/mappings/deterministic_attack_defense_lookup.csv` (ground truth), `data/mappings/learned_mapping.csv`
- **Baselines:** Empirical (logistic regression / majority classifier on same EMBER split) — see `docs/BASELINE_METHODOLOGY_TEMP.md`

## Optional / supplementary

- `scripts/h1h2_rebuild/` — post-hoc rebuild and ransomware-only registers
- `results/H1_oof_robust_eval/` — isolated OOF robustness evaluation
- `mappings/`, `mappings_project/` — legacy mapping experiments (prefer `data/mappings/` + `aicra/mappings/`)

## Ignored at runtime

- `.venv/`, `htmlcov/`, `*.log`, `mlruns/` — local/dev artifacts (see `.gitignore`)
