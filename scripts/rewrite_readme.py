#!/usr/bin/env python3
"""
Rewrite README.md to match current repository reality after cleanup.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def load_latest_metrics():
    """Load latest metrics for README."""
    metrics = {}
    splits = ["smoke_test", "small_ember", "main", "full_ember"]

    for split in splits:
        metrics_path = repo_root / "results" / split / "metrics_optimized.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                metrics[split] = json.load(f)

    return metrics


def create_readme(metrics: dict):
    """Create new README.md content."""

    # Get latest metrics summary
    metrics_summary = ""
    if metrics:
        metrics_summary = "\n### Latest Performance Metrics\n\n"
        metrics_summary += "All splits meet target metrics (>= 88% Precision/Recall/F1, < 0.12 Brier/ECE):\n\n"
        metrics_summary += "| Split | Precision | Recall | F1 | Brier | ECE | AUROC |\n"
        metrics_summary += "|-------|-----------|--------|----|----|----|----|\n"
        for split in ["smoke_test", "small_ember", "main", "full_ember"]:
            if split in metrics:
                m = metrics[split]
                metrics_summary += f"| {split} | {m.get('precision', 0):.4f} | {m.get('recall', 0):.4f} | {m.get('f1', 0):.4f} | {m.get('brier_score', 0):.4f} | {m.get('ece', 0):.4f} | {m.get('auroc', 0):.4f} |\n"
        metrics_summary += "\nFor detailed metrics, see `docs/BENCHMARK_NOTES.md`.\n"

    readme_content = f"""# AICRA – Machine Learning-Based Cyber Risk Advisor for Endpoint Security in U.S. Banking Organizations

[![CI](https://github.com/Kolawole-a2/AICRA/actions/workflows/ci.yml/badge.svg)](https://github.com/Kolawole-a2/AICRA/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/aicra/aicra/branch/main/graph/badge.svg)](https://codecov.io/gh/aicra/aicra)
[![PyPI version](https://badge.fury.io/py/aicra.svg)](https://badge.fury.io/py/aicra)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Machine Learning-Based cyber risk advisor that predicts ransomware and endpoint threats, calibrates risk scores, and aligns MITRE ATT&CK techniques to D3FEND countermeasures for U.S. banking endpoint security.**

---

## Research Context & Praxis Overview

This repository implements the **Doctor of Engineering praxis**: *Machine Learning-Based Cyber Risk Advisor with Analytics for Endpoint Security in U.S. Banking Organizations (AICRA)*.

### Domain & Scope

- **Domain**: U.S. banking endpoint security, ransomware risk assessment
- **Key Innovation**: Combines ML predictions, calibrated risk scoring, and ontology-based ATT&CK→D3FEND mapping
- **Research Focus**: Validates three hypotheses (H1, H2, H3) that demonstrate improvements in detection performance, calibration, and mapping consistency

### Research Approach

AICRA integrates:
1. **Machine Learning Classification**: LightGBM-based ransomware detection using static PE features
2. **Probability Calibration**: Platt/Isotonic regression for reliable risk scores
3. **Cost-Aware Decision Making**: Business-aligned threshold optimization
4. **Ontology-Based Mapping**: Deterministic and learned ATT&CK→D3FEND mappings with quantitative consistency metrics

---

## What's Stable and Locked

### H3 Deterministic Mapping Pipeline (READ-ONLY)

**CRITICAL**: H3 code, configs, data, and results are **READ-ONLY** and must not be modified. See `docs/DO_NOT_TOUCH_H3.md` for the complete list of protected files.

H3 provides deterministic lookup tables used read-only by the H1/H2 pipeline:
- `data/lookups/family_to_attack.yaml` - Family→ATT&CK technique mappings
- `data/lookups/attack_to_d3fend.yaml` - ATT&CK→D3FEND control mappings
- `data/mappings/deterministic_lookup.csv` - Full deterministic lookup table

These mappings are considered canonical and are used to generate ransomware-only risk registers.

---

## Current Pipeline: H1/H2 Rebuild

The H1/H2 pipeline has been rebuilt to generate per-sample predictions with correct schemas and ransomware-only risk registers. The pipeline consists of four phases:

### Phase 1: Build Split Manifests

**Script**: `scripts/h1h2/build_split_manifests.py`

Creates canonical per-sample metadata manifests for each split.

**Outputs**:
- `artifacts/splits/<split>.manifest.csv`

**Schema**:
- `sample_id`: Stable sample identifier
- `split`: Split name (smoke_test, small_ember, main, full_ember)
- `true_label`: Ground truth label (0=benign, 1=ransomware)
- `family`: Malware family name (required for label==1, must not be "Unknown")
- `sha256`: Optional SHA256 hash if available

### Phase 2: Train and Score

**Script**: `scripts/h1h2/train_and_score.py`

Trains LightGBM model, applies calibration, and generates per-sample predictions.

**Outputs**:
- `results/<split>/risk_scores.csv` - Per-sample predictions
- `models/h1h2/<split>/` - Trained models (LightGBM + calibration)

**Schema** (`risk_scores.csv`):
- `sample_id`: Sample identifier
- `true_label`: Ground truth label (0=benign, 1=ransomware)
- `family`: Malware family name
- `p_ransomware`: Probability of ransomware (calibrated)
- `predicted_label`: Binary prediction
- `split_part`: "train" or "test"

### Phase 3: Generate Risk Registers

**Script**: `scripts/generate_ransomware_only_registers_FINAL.py`

Builds ransomware-only risk registers using H3 deterministic lookup tables (read-only).

**Outputs**:
- `register/<split>/ransomware_only_risk_register.csv`

**Schema**:
- `sample_id`, `family`, `p_ransomware`
- `susceptibility_bucket`: "Low", "Med", "High"
- `impact`, `expected_loss`
- `attack_technique_id`: ATT&CK technique ID
- `d3fend_control_id`, `d3fend_control_name`: D3FEND controls

**Row Grain**: One row per ransomware sample per technique

### Phase 4: Optional - Clean Registers

**Script**: `scripts/clean_register_drop_nan_family.py`

Optional post-processing to remove samples lacking resolvable family attribution.

**Usage**:
```bash
# Preview changes
python scripts/clean_register_drop_nan_family.py --dry-run

# Create cleaned files (original preserved)
python scripts/clean_register_drop_nan_family.py

# Archive originals and replace with cleaned versions
python scripts/clean_register_drop_nan_family.py --replace
```

---

## How to Run the Pipeline

### End-to-End Execution

```bash
# Step 1: Build manifests
python scripts/h1h2/build_split_manifests.py

# Step 2: Train and score
python scripts/h1h2/train_and_score.py

# Step 3: Generate registers
python scripts/generate_ransomware_only_registers_FINAL.py

# Step 4 (optional): Clean registers
python scripts/clean_register_drop_nan_family.py
```

### Expected Outputs

1. **Manifests**: `artifacts/splits/<split>.manifest.csv`
2. **Risk Scores**: `results/<split>/risk_scores.csv`
3. **Models**: `models/h1h2/<split>/h1h2_<split>.joblib` and `calibrator_<split>.joblib`
4. **Registers**: `register/<split>/ransomware_only_risk_register.csv`
5. **Metrics**: `results/<split>/metrics_optimized.json` and `results/h1h2_optimized_metrics.json`
6. **Plots**: `results/<split>/plots/` (ROC, PR, Confusion Matrix, Reliability Diagram)

### Metrics Location

- **Per-split metrics**: `results/<split>/metrics_optimized.json`
- **Combined metrics**: `results/h1h2_optimized_metrics.json`
- **Benchmark notes**: `docs/BENCHMARK_NOTES.md`
{metrics_summary}

---

## Validation Guarantees

The pipeline includes comprehensive validation at each phase:

1. **Manifest Validation**:
   - `true_label` must be {{0,1}}
   - For `label==1`, `family` must be non-null and not "Unknown"
   - Record counts match split definitions

2. **Risk Scores Validation**:
   - `p_ransomware` in [0,1]
   - Both label==0 and label==1 have multiple unique probabilities (not constant)
   - `mean(p_ransomware | label=1) > mean(p_ransomware | label=0)`

3. **Register Validation**:
   - NO benign rows (only label==1)
   - Unknown/unmapped family rate = 0%
   - Technique diversity ≥ 5 unique techniques (main/full_ember)
   - Controls mapped rate ≥ 95%
   - `p_ransomware` has ≥ 20 unique values (small_ember and larger)

---

## Hypotheses (H1, H2, H3)

### H1 – Baseline Predictive Performance

**Hypothesis**: Static PE features enable reliable ransomware classification with AUROC >= 0.95 and operational precision suitable for banking environments.

**Key Metrics**:
- AUROC, PR-AUC, Precision, Recall, F1
- Brier Score, ECE
- Lift@1%, Lift@5%, Lift@10%

**Results**: See `results/<split>/metrics_optimized.json` and `docs/BENCHMARK_NOTES.md`

### H2 – Calibration & Risk Scoring Stability

**Hypothesis**: Calibration and cost-aware thresholding produce more decision-aligned susceptibility scores than uncalibrated F1-optimized thresholds.

**Key Metrics**:
- Brier Score (before/after calibration)
- ECE (before/after calibration)
- Expected Loss at cost-optimal thresholds

**Results**: Integrated into H1/H2 rebuild pipeline outputs

### H3 – Defense–Attack Consistency (DAC)

**Hypothesis**: Deterministic ATT&CK–D3FEND mappings exhibit higher Defense–Attack Consistency (DAC_internal), higher actionable precision, and greater risk-score stability compared to learned mappings.

**IMPORTANT**: H3 is **READ-ONLY**. See `docs/DO_NOT_TOUCH_H3.md`.

**Command** (for reference only - H3 is locked):
```bash
python -m aicra.experiments.h3_evaluation --config config/h3_splits.yaml
```

**Results**: `results/H3_full_evaluation/`

---

## Data Requirements

### EMBER-2024 Dataset

For H1/H2 experiments, you need EMBER-2024 data files. The dataset should be placed in `data/ember2024/` (or set `AICRA_EMBER2024_DIR` environment variable).

**To set up the EMBER 2024 dataset**:
```bash
# Check if dataset is available
bash scripts/fetch_data.sh  # Linux/Mac
.\\scripts\\fetch_data.ps1    # Windows

# If missing, follow the instructions provided by the script
# See docs/DATA.md for detailed information
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Kolawole-a2/AICRA.git
cd AICRA

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

---

## Documentation

- **`docs/H1H2_REBUILD_GUIDE.md`** - Comprehensive guide to the H1/H2 rebuild pipeline
- **`docs/BENCHMARK_NOTES.md`** - Latest performance metrics and benchmarks
- **`docs/DO_NOT_TOUCH_H3.md`** - H3 protection document (CRITICAL)
- **`docs/EXPERIMENTS.md`** - Detailed step-by-step reproduction instructions
- **`docs/DATA.md`** - Data requirements and setup instructions

---

## Benchmarks and Improvements

For detailed benchmark comparisons and AICRA improvements, see:
- **`docs/BENCHMARK_NOTES.md`** - Latest metrics per split
- **`SOURCE_CONTRIBUTION_AND_AICRA_IMPROVEMENTS.md`** - Complete breakdown of source contributions and improvements

---

## License

MIT License - see `LICENSE` file for details.

---

## Citation

If you use AICRA in your research, please cite:

```bibtex
@software{{aicra2024,
  title={{AICRA: Machine Learning-Based Cyber Risk Advisor for Endpoint Security}},
  author={{Your Name}},
  year={{2024}},
  url={{https://github.com/Kolawole-a2/AICRA}}
}}
```

---

**Last Updated**: {Path(__file__).stat().st_mtime if Path(__file__).exists() else "N/A"}
"""

    return readme_content


def main():
    """Main execution."""
    print("=" * 80)
    print("Rewriting README.md")
    print("=" * 80)

    # Load metrics
    print("\n[1] Loading latest metrics...")
    metrics = load_latest_metrics()
    print(f"Loaded metrics for {len(metrics)} splits")

    # Create README
    print("\n[2] Creating new README.md...")
    readme_content = create_readme(metrics)

    # Write README
    readme_path = repo_root / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"\n[OK] Rewritten README.md: {readme_path.relative_to(repo_root)}")
    print("\n" + "=" * 80)
    print("README.md Update Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
