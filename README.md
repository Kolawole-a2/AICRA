# AI Cyber Risk Advisor (AICRA)

[![CI](https://github.com/aicra/aicra/workflows/CI/badge.svg)](https://github.com/aicra/aicra/actions)
[![codecov](https://codecov.io/gh/aicra/aicra/branch/main/graph/badge.svg)](https://codecov.io/gh/aicra/aicra)
[![PyPI version](https://badge.fury.io/py/aicra.svg)](https://badge.fury.io/py/aicra)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Overview

AICRA (AI Cyber Risk Advisor) is a machine learning-based cyber risk assessment system focused on endpoint ransomware detection in banking environments. It provides reproducible, auditable risk assessments with cost-sensitive decision making.

### Key Features

- **Ransomware Detection**: LightGBM-based binary classification with calibration
- **Cost-Sensitive Thresholding**: Optimal decision thresholds based on business impact
- **Risk Register**: CSV/JSON output with MITRE ATT&CK to D3FEND mappings
- **Drift Detection**: Evidently-based data and performance drift monitoring
- **Reproducible Pipeline**: Deterministic builds with MLflow tracking
- **Quality Gates**: Pre-commit hooks, type checking, security auditing

### Artifacts

- Console metrics (AUROC, PR-AUC, Brier, ECE, Lift@10%, confusion matrix)
- Cost-sensitive threshold selection with configurable impact costs
- Cyber Risk Advisor Register with prescriptive controls
- Policy JSON for auditable decision thresholds
- Drift reports and reliability diagrams

### Automatic Result Archiving

AICRA automatically archives all key artifacts from each run to timestamped folders for easy tracking and reporting:

**Directory Structure:**
```
results/
├── run_2025-10-17_2030/          # Timestamped run folder
│   ├── metrics.json               # Performance metrics
│   ├── policy.json               # Decision thresholds
│   ├── risk_register.csv         # Risk assessment results
│   ├── roc.png                   # ROC curve plot
│   ├── pr.png                    # Precision-Recall curve
│   ├── reliability.png            # Reliability diagram
│   ├── confusion_at_ops.png      # Confusion matrix
│   ├── lift_curve.png            # Lift curve (if available)
│   ├── bagged_lightgbm.joblib    # Trained model
│   ├── calibrator.joblib         # Calibration model
│   └── impact.csv                # Impact table context
├── run_2025-10-17_2045_smoke/    # Smoke test run
└── versions_log.csv              # Summary of all runs
```

**Features:**
- **Automatic Archiving**: Every run (`aicra register`, `aicra smoke`, `aicra run-all`) automatically creates a timestamped folder
- **Version Tracking**: `versions_log.csv` maintains a summary of all experiments with metrics
- **Git-Ready**: Archived results can be safely committed to GitHub for collaboration
- **Manual Control**: Use `aicra archive-results` to manually archive current artifacts

**Example Usage:**
```bash
# Run pipeline (automatically archives results)
aicra run-all

# Manually archive current results
aicra archive-results --run-name "experiment_v2"

# List recent runs
aicra list-runs --limit 5

# View versions log
cat results/versions_log.csv
```

**Sample Metrics from versions_log.csv:**
```csv
timestamp,run_name,dataset,model,AUROC,PR-AUC,ECE,Lift@5,folder_path
2025-10-17 20:30:15,run_2025-10-17_2030,ember2024,bagged_lightgbm,0.8234,0.1567,0.0892,1.45,results/run_2025-10-17_2030
2025-10-17 20:45:22,run_2025-10-17_2045_smoke,synthetic,mock_model,0.6789,0.1234,0.1456,1.23,results/run_2025-10-17_2045_smoke
```

## Quickstart

### Installation

```bash
# Clone repository
git clone https://github.com/aicra/aicra.git
cd aicra

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

### Smoke Test (Pre-Expansion)

Before scaling to full datasets, run the smoke test to validate the entire pipeline:

```bash
# Run end-to-end smoke test
make smoke
# or
aicra smoke

# Dry run (no training, just validation)
aicra smoke --dry-run
```

**Smoke Test Purpose**: Validates the complete AICRA pipeline on tiny synthetic data to ensure all components work together before scaling to production datasets.

**PASS Criteria**:
- AUROC ≥ 0.70
- PR-AUC > 0.05 (above prevalence)
- Brier Score ≤ 0.25
- Expected Calibration Error (ECE) ≤ 0.15
- Lift@5% > 1.0 (or Lift@10% > 1.0)
- All required artifacts generated (metrics.json, plots, policy.json, register.csv)
- Register contains ≥10 rows with required columns (susceptibility, bucket, techniques, controls)

**Output**: PASS/FAIL summary with metrics, artifact paths, and exit code (0=PASS, 1=FAIL).

### Basic Usage

```bash
# Run complete pipeline
aicra run-all

# Individual commands
aicra train                    # Train model
aicra evaluate                # Evaluate performance
aicra calibrate               # Calibrate probabilities
aicra thresholds              # Compute optimal thresholds
aicra drift-check --new-data sample.csv  # Check for drift
aicra register                # Generate risk register
```

### Makefile Commands

```bash
# Development
make setup                    # Set up development environment
make lint                     # Run linting
make typecheck               # Run type checking
make test                    # Run tests
make coverage                # Run tests with coverage
make audit                   # Security audit

# Pipeline
make train                   # Train model
make evaluate                # Evaluate model
make calibrate               # Calibrate model
make thresholds              # Compute thresholds
make drift                   # Check drift
make register                # Generate register

# Complete pipeline
make all                     # Run everything
```

## Configuration

AICRA uses pydantic-settings for configuration management:

```bash
# Environment variables
export RANDOM_SEED=42
export COST_FP=5.0
export COST_FN=100.0
export IMPACT_DEFAULT=5000000.0
export DRIFT_THRESHOLD=0.05
```

Or create a `.env` file:

```env
RANDOM_SEED=42
COST_FP=5.0
COST_FN=100.0
IMPACT_DEFAULT=5000000.0
DRIFT_THRESHOLD=0.05
MLFLOW_TRACKING_URI=file:./mlruns
```

## Data Requirements

- **EMBER-2024 Dataset**: Static analysis features from PE executables
- **Format**: JSONL (JSON Lines) with feature vectors and metadata
- **Schema**: Validated against `schemas/input_schema.json`

## Model Architecture

- **Base Model**: LightGBM Classifier
- **Ensemble**: Bagged ensemble with 3 models (seeds: 17, 42, 73)
- **Calibration**: Isotonic regression for probability calibration
- **Validation**: Time-ordered split with out-of-family evaluation

## Quality Assurance

### Code Quality
- **Linting**: Ruff, Pylint
- **Formatting**: Black, isort
- **Type Checking**: mypy (strict mode)
- **Security**: pip-audit, detect-secrets

### Testing
- **Unit Tests**: pytest with coverage
- **Integration Tests**: End-to-end pipeline testing
- **Schema Validation**: Input data validation

### CI/CD
- **GitHub Actions**: Automated testing and quality gates
- **SonarQube**: Code quality analysis
- **Docker**: Containerized deployment

## Governance

- **Data License**: See `DATA_LICENSE.md`
- **Model Card**: See `model_card.md`
- **Schema**: `schemas/input_schema.json`
- **Security**: Regular security audits and dependency updates

## Development

### Project Structure

```
aicra/
├── cli.py              # Typer CLI interface
├── config.py           # Pydantic settings
├── core/               # Core functionality
│   ├── data.py         # Data handling
│   ├── evaluation.py   # Metrics and evaluation
│   └── calibration.py  # Probability calibration
├── models/             # ML models
│   └── lightgbm.py    # LightGBM implementation
├── pipelines/          # ML pipelines
│   ├── training.py     # Training pipeline
│   ├── evaluation.py   # Evaluation pipeline
│   ├── calibration.py  # Calibration pipeline
│   └── drift.py        # Drift detection
├── utils/              # Utilities
└── register.py         # Risk register generation
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality gates: `make ci`
5. Submit a pull request

### License

MIT License - see `LICENSE` file for details.

## Citation

```bibtex
@software{aicra2024,
  title={AICRA: AI Cyber Risk Advisor},
  author={AICRA Team},
  year={2024},
  url={https://github.com/aicra/aicra}
}
```

## Support

- **Documentation**: [docs.aicra.org](https://docs.aicra.org)
- **Issues**: [GitHub Issues](https://github.com/aicra/aicra/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aicra/aicra/discussions)
- **Email**: [support@aicra.org](mailto:support@aicra.org)

