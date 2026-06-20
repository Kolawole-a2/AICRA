# AICRA

**Machine-learning cyber risk advisor for endpoint ransomware defense in U.S. banking organizations.**

[![CI](https://github.com/Kolawole-a2/AICRA/actions/workflows/ci.yml/badge.svg)](https://github.com/Kolawole-a2/AICRA/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Software artifact and reproducible evidence for the Doctor of Engineering **praxis (production)** by **[Kolawole Afolabi](https://github.com/Kolawole-a2)** — *Machine Learning-Based Cyber Risk Advisor with Analytics for Endpoint Ransomware Defense in U.S. Banking Organizations*.

**Contact:** [kolawole.afolabi@gwmail.gwu.edu](mailto:kolawole.afolabi@gwmail.gwu.edu) · [ako.afolabi@gmail.com](mailto:ako.afolabi@gmail.com)

---

## What this repository is

AICRA tests three hypotheses on EMBER-2024 static PE malware data with **empirical baselines on the same splits** (not literature percentages):

| Step | Hypothesis | Question (short) | Primary metric |
|------|------------|------------------|----------------|
| **H1** | Detect | Can static PE features classify ransomware reliably? | **AUROC** |
| **H2** | Decide | Do cost-aware thresholds beat F1-optimal under banking costs? | **Expected loss** |
| **H3** | Defend | Does expert ATT&CK→D3FEND mapping beat a learned mapping? | **DAC_internal** |

Optional **risk registers** under `register/` are demonstration outputs; they do not change canonical H1–H3 results.

**Full praxis documentation (defense / review):** **[docs/praxis/README.md](docs/praxis/README.md)**

---

## Results at a glance (saved artifacts)

Values below come from canonical JSON/MD in this repo — see [docs/BENCHMARK_NOTES.md](docs/BENCHMARK_NOTES.md) and [results/praxis_validation_report.md](results/praxis_validation_report.md).

### H1 — Classification

Validated on **three modes:** time-ordered train/test, **multi-split** (`config/h1_splits.yaml`), and supplementary **out-of-family** ([`results/H1_oof_robust_eval/`](results/H1_oof_robust_eval/)).

| Metric | Benchmark / baseline | AICRA (canonical) |
|--------|----------------------|-------------------|
| AUROC | Reliability **> 0.88**; empirical logistic ≈ **0.778** (same split) | **0.9796** (full_ember); mean **0.9605** multi-split; **0.9615** OOF |
| Lift vs logistic | — | **+25.9%** AUROC on full_ember |

### H2 — Thresholds & calibration test

**Primary finding:** cost-optimal thresholding cuts expected loss vs F1-optimal (~**−50.6%** on aggregated splits).

Platt/isotonic regression is applied **post hoc to test whether calibration helps** (Brier, ECE, expected loss). The H1 model is already well-calibrated; post-hoc calibration **does not** improve expected loss here.

### H3 — Deterministic vs learned mapping

| Mapping | DAC_internal (all splits) | Variance reduction |
|---------|---------------------------|-------------------|
| Deterministic | **100%** (always correct) | **0.0** |
| Learned | **0%** (always extraneous) | **0.0** |

With zero variance on every split, t-test / Wilcoxon / Shapiro–Wilk on variance reduction are **not applicable**. H3 is supported by **perfect separation** and consistent superiority on DAC_internal and actionable precision — not variance-reduction p-values.

External reference pairs (`d3fend_reference_pairs.csv`) provide a secondary **DAC_external** check only.

---

## Canonical result files

| Hypothesis | Results | Summary |
|------------|---------|---------|
| H1 | `results/H1_classification/H1_full_results.json` | `results/H1_classification/H1_summary.md` |
| H1 OOF | `results/H1_oof_robust_eval/oof_robust_metrics.json` | `results/H1_oof_robust_eval/oof_robust_summary.md` |
| H2 | `results/H2_calibration_thresholds/H2_full_results.json` | `results/H2_calibration_thresholds/H2_summary.md` |
| H3 | `results/H3_full_evaluation/H3_full_results.json` | `results/H3_full_evaluation/H3_full_summary.md` |

Do not overwrite these casually when re-running experiments.

---

## Quick start

```bash
git clone https://github.com/Kolawole-a2/AICRA.git && cd AICRA
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements-dev.txt
# EMBER-2024: see docs/DATA.md and scripts/fetch_data.ps1

# H1 — multi-split classification
python -m aicra.experiments.h1_classification --splits-config config/h1_splits.yaml

# H1 OOF supplementary (separate folder)
python scripts/evaluate_h1_oof_robust.py

# H2 — after H1
python -m aicra.experiments.h2_calibration_thresholds --splits-config config/h2_splits.yaml

# H3 — mapping evaluation
python -m aicra.experiments.h3_evaluation --config config/h3_splits.yaml

# All three
python scripts/run_all_hypotheses.py
```

**Tests:** `pytest tests/`

Command details, layouts, and reviewer paths: **[docs/praxis/EXPERIMENTS_GUIDE.md](docs/praxis/EXPERIMENTS_GUIDE.md)** · **[docs/REVIEWER_GUIDE.md](docs/REVIEWER_GUIDE.md)**

---

## Documentation map

| For… | Start here |
|------|------------|
| Defense / examiner review | [docs/praxis/README.md](docs/praxis/README.md) |
| Step-by-step reproduction | [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) |
| Results tables & interpretation | [docs/RESULTS_SUMMARY.md](docs/RESULTS_SUMMARY.md) |
| P-values & H3 variance note | [docs/HYPOTHESIS_TESTING_PVALUES.md](docs/HYPOTHESIS_TESTING_PVALUES.md) |
| Baseline methodology | [docs/BASELINE_METHODOLOGY_TEMP.md](docs/BASELINE_METHODOLOGY_TEMP.md) |
| Historical dev notes | [docs/archive/development/](docs/archive/development/) (traceability only) |

The previous long README (~1,300 lines) is archived at [docs/archive/README_FULL_PRE_2026.md](docs/archive/README_FULL_PRE_2026.md).

---

## Citation

```bibtex
@software{aicra2024,
  title={AICRA: Machine Learning-Based Cyber Risk Advisor with Analytics for Endpoint Ransomware Defense in U.S. Banking Organizations},
  author={Afolabi, Kolawole},
  note={Doctor of Engineering praxis (production) software artifact},
  year={2024},
  url={https://github.com/Kolawole-a2/AICRA}
}
```

## License

MIT — see [LICENSE](LICENSE).
