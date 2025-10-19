# AICRA Usage Guide

This guide provides comprehensive instructions for using the AICRA CLI and its various components.

## Installation

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd AICRA

# Install dependencies
make setup

# Or manually:
pip install -e .
pip install -r requirements/dev.txt
pre-commit install
```

### Docker

```bash
# Build the image
make docker-build

# Or manually:
docker build -t aicra:latest .

# Run with docker-compose
docker-compose up
```

## CLI Commands

### Data Management

#### Fetch Data
```bash
# Fetch EMBER 2024 dataset
aicra data fetch --data-dir data/ember2024

# Fetch with sample size for testing
aicra data fetch --data-dir data/ember2024 --sample-size 1000
```

#### Validate Data
```bash
# Validate data schema
aicra data validate --data-dir data/ember2024

# Validate with custom schema
aicra data validate --data-dir data/ember2024 --schema data_governance/schema.json
```

#### Create Snapshot
```bash
# Create data snapshot
aicra data snapshot --data-dir data/ember2024
```

### Model Training

#### Basic Training
```bash
# Train model with default settings
aicra train --data-dir data/ember2024 --model-dir models

# Train with custom parameters
aicra train \
  --data-dir data/ember2024 \
  --model-dir models \
  --sample-size 10000 \
  --seed 42
```

#### Resume Training
```bash
# Resume training from checkpoint
aicra train \
  --data-dir data/ember2024 \
  --model-dir models \
  --resume
```

### Model Evaluation

#### Basic Evaluation
```bash
# Evaluate model performance
aicra eval \
  --model-path models/model.pkl \
  --data-dir data/ember2024 \
  --output-dir artifacts

# Evaluate with sample size
aicra eval \
  --model-path models/model.pkl \
  --data-dir data/ember2024 \
  --output-dir artifacts \
  --sample-size 5000
```

### Predictions

#### Single File Prediction
```bash
# Predict on single file
aicra predict \
  --model-path models/model.pkl \
  --input-file sample.exe \
  --output-file prediction.json

# Predict with custom threshold
aicra predict \
  --model-path models/model.pkl \
  --input-file sample.exe \
  --threshold 0.7
```

### Drift Monitoring

#### Check Data Drift
```bash
# Check for drift between datasets
aicra drift \
  --reference-data data/ember2024/train \
  --current-data data/ember2024/test \
  --output-dir artifacts

# Check with custom threshold
aicra drift \
  --reference-data data/ember2024/train \
  --current-data data/ember2024/test \
  --output-dir artifacts \
  --drift-threshold 0.05
```

### Model Calibration

#### Calibrate Probabilities
```bash
# Calibrate model probabilities
aicra calibrate \
  --model-path models/model.pkl \
  --data-dir data/ember2024 \
  --output-dir artifacts \
  --method isotonic

# Use Platt scaling
aicra calibrate \
  --model-path models/model.pkl \
  --data-dir data/ember2024 \
  --output-dir artifacts \
  --method platt
```

### Threshold Optimization

#### Optimize Threshold
```bash
# Optimize threshold based on costs
aicra optimize-threshold \
  --model-path models/model.pkl \
  --data-dir data/ember2024 \
  --output-dir artifacts \
  --fn-cost 1000.0 \
  --fp-cost 100.0
```

### Model Card Generation

#### Generate Model Card
```bash
# Generate comprehensive model card
aicra generate-model-card \
  --model-path models/model.pkl \
  --output-dir artifacts
```

## Configuration

### Environment Variables

Set the following environment variables or use the `.env` file:

```bash
# Data paths
export DATA_DIR=data/ember2024
export MODEL_DIR=models
export ARTIFACTS_DIR=artifacts

# MLflow configuration
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=aicra_experiment

# Model configuration
export MODEL_TYPE=lightgbm
export RANDOM_SEED=42
export TEST_SIZE=0.2
export VALIDATION_SIZE=0.2

# Training configuration
export MAX_ITER=1000
export EARLY_STOPPING_ROUNDS=50
export LEARNING_RATE=0.1

# Evaluation configuration
export COVERAGE_THRESHOLD=0.85
export DRIFT_THRESHOLD=0.1

# Cost optimization
export FALSE_NEGATIVE_COST=1000.0
export FALSE_POSITIVE_COST=100.0
```

### Configuration Files

The system uses Pydantic settings for configuration management:

- `aicra/config/settings.py` - Main settings class
- `aicra/config/schemas.py` - Configuration schemas
- `aicra/config/profiles/` - Environment-specific profiles

## Output Artifacts

### Evaluation Results

The evaluation process generates several artifacts:

- `artifacts/plots/roc.png` - ROC curve
- `artifacts/plots/pr.png` - Precision-Recall curve
- `artifacts/plots/confusion.png` - Confusion matrix
- `artifacts/metrics.json` - Evaluation metrics

### Calibration Results

Calibration generates:

- `artifacts/plots/reliability.png` - Reliability diagram
- `artifacts/calibration_metrics.json` - Calibration metrics

### Drift Analysis

Drift monitoring creates:

- `artifacts/DriftReport.md` - Comprehensive drift report
- `artifacts/drift_metrics.json` - Drift statistics

### Threshold Optimization

Threshold optimization produces:

- `artifacts/threshold_table.csv` - Metrics by threshold
- `artifacts/threshold_analysis.json` - Optimization results
- `artifacts/plots/cost_curve.png` - Cost curve visualization

### Model Card

Model card generation creates:

- `artifacts/ModelCard.md` - Comprehensive model documentation

## Examples

### Complete Workflow

```bash
# 1. Fetch and validate data
aicra data fetch --data-dir data/ember2024
aicra data validate --data-dir data/ember2024

# 2. Train model
aicra train --data-dir data/ember2024 --model-dir models

# 3. Evaluate model
aicra eval --model-path models/model.pkl --data-dir data/ember2024 --output-dir artifacts

# 4. Calibrate probabilities
aicra calibrate --model-path models/model.pkl --data-dir data/ember2024 --output-dir artifacts

# 5. Optimize threshold
aicra optimize-threshold --model-path models/model.pkl --data-dir data/ember2024 --output-dir artifacts

# 6. Generate model card
aicra generate-model-card --model-path models/model.pkl --output-dir artifacts

# 7. Check for drift
aicra drift --reference-data data/ember2024/train --current-data data/ember2024/test --output-dir artifacts
```

### Batch Processing

```bash
# Process multiple files
for file in samples/*.exe; do
  aicra predict --model-path models/model.pkl --input-file "$file" --output-file "predictions/$(basename "$file").json"
done
```

### Continuous Monitoring

```bash
# Set up drift monitoring
aicra drift \
  --reference-data data/ember2024/train \
  --current-data data/ember2024/latest \
  --output-dir artifacts

# Check results
if grep -q "Drift Detected: ✅ YES" artifacts/DriftReport.md; then
  echo "Drift detected! Consider retraining."
fi
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `make setup`
2. **Data Not Found**: Check that data directory exists and contains valid files
3. **Model Not Found**: Verify model path is correct and model file exists
4. **Permission Errors**: Ensure output directories are writable

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
aicra train --data-dir data/ember2024 --model-dir models
```

### Performance Issues

For large datasets:

```bash
# Use smaller sample sizes for testing
aicra train --data-dir data/ember2024 --sample-size 1000

# Enable parallel processing
export OMP_NUM_THREADS=4
aicra train --data-dir data/ember2024 --model-dir models
```

## Support

For additional help:

1. Check the logs in `logs/aicra.log`
2. Review the generated artifacts in `artifacts/`
3. Consult the model card in `artifacts/ModelCard.md`
4. Contact the AICRA team for support
