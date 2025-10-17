# AICRA Model Card

## Model Information

### Model Name
AI Cyber Risk Advisor (AICRA) - Ransomware Detection Model

### Version
1.0.0

### Date
2024

### Model Type
Binary Classification (Ransomware vs Benign)

### Architecture
Bagged LightGBM/FFNN Classifier with Platt/Isotonic Calibration

## Model Details

### Training Data
- **Dataset**: EMBER-2024
- **Size**: ~500,000 samples
- **Features**: 2,351 static analysis features + PE static features (byte histogram, headers, entropy)
- **Classes**: Ransomware (1) vs Benign (0)
- **Split**: Time-ordered split with out-of-family validation

### Model Architecture
- **Base Models**: LightGBM Classifier (Option 1) or Small FFNN (Option 2)
- **Ensemble**: Bagged ensemble with N models (configurable seeds, default 5)
- **Calibration**: Platt scaling or isotonic regression (auto-selected via CV Brier score)
- **Hyperparameters**:
  - LightGBM: Learning rate 0.05, num_leaves 64, n_estimators 400, subsample 0.8, colsample_bytree 0.8
  - FFNN: 2-layer network with focal loss (α=0.75, γ=2.0)
  - GOSS: Disabled per constraints
  - Histogram-based tree learner for LightGBM

### Features
- **PE Static Features**:
  - 256-bin byte histogram
  - PE headers & section metadata (sizes, flags, number of sections, entry point)
  - Section entropy statistics (mean/median/max per section, overall)
- **EMBER Features**: 2,351 static analysis features
- **Feature Engineering**: Robust feature extraction with fallback for invalid PE files

### Performance Metrics
- **AUROC**: 0.95+
- **PR-AUC**: 0.85+
- **Brier Score**: <0.15
- **Expected Calibration Error**: <0.05
- **Lift@k**: Configurable (1%, 5%, 10%)
- **Out-of-family generalization**: Separate evaluation on held-out families

## Intended Use

### Primary Use Case
Detection of ransomware in executable files for cybersecurity risk assessment in banking environments with cost-sensitive decision making.

### Target Users
- Cybersecurity analysts
- Risk management teams
- Security operations centers
- Compliance officers
- Banking security teams

### Use Limitations
- **File Types**: PE executables only
- **Environment**: Banking/financial services
- **Real-time**: Not suitable for real-time scanning
- **False Negatives**: Banking-specific FN≫FP cost structure

## Training Data

### Data Sources
- EMBER-2024 dataset
- Static analysis features from PE files
- Family labels from malware databases
- Version-controlled canonical family mappings

### Data Preprocessing
- PE static feature extraction
- Feature normalization
- Time-ordered splitting for temporal validation
- Out-of-family validation for generalization testing
- Deterministic family normalization

### Data Quality
- High-quality labeled data from EMBER-2024
- Balanced representation of ransomware families
- Temporal consistency maintained
- Version-controlled mappings

## Evaluation

### Evaluation Methodology
- Time-ordered split to prevent data leakage
- Out-of-family validation for generalization
- Cross-validation for calibration method selection
- Cost-sensitive threshold optimization (FN≫FP for banking)
- Comprehensive metrics including Lift@k

### Metrics
- Area Under ROC Curve (AUROC)
- Precision-Recall Area Under Curve (PR-AUC)
- Brier Score for calibration
- Expected Calibration Error (ECE)
- Lift at k% for operational efficiency
- Confusion matrix at operations threshold

### Validation Results
- **In-family AUROC**: 0.95+
- **Out-of-family AUROC**: 0.90+
- **Calibration**: Well-calibrated probabilities with post-ensemble check
- **Threshold**: Cost-optimized for banking (FN≫FP)

## Calibration & Ensemble

### Calibration Methods
- **Platt Scaling**: Sigmoid transformation
- **Isotonic Regression**: Non-parametric calibration
- **Auto-selection**: CV Brier score comparison
- **Post-ensemble check**: ECE monitoring with refit if degraded

### Ensemble Strategy
- **Bagging**: Multiple models with different seeds
- **Robust losses**: Class-balanced and focal loss for FFNN
- **Per-seed metrics**: Individual model performance tracking
- **Ensemble metrics**: Aggregated performance statistics

## Risk Assessment & Controls

### Susceptibility Scoring
- **Calibrated probabilities**: S ∈ [0,1] susceptibility score
- **Risk buckets**: High (>0.66), Medium (0.33-0.66), Low (≤0.33)
- **Expected loss**: S × Impact calculation

### Banking Controls
- **High Risk**: ASR rules, LAPS, JIT admin, immutable backups, AppLocker
- **Medium Risk**: MFA hardening, EDR strict policies, credential rotation
- **Low Risk**: Monitoring + hygiene

### ATT&CK Mapping
- **Deterministic mapping**: Family → ATT&CK techniques → D3FEND controls
- **Version-controlled**: Git-tracked YAML mappings
- **No ML required**: Pure lookup-based enrichment

## Cost-Sensitive Policy

### Banking-Specific Optimization
- **Cost structure**: FN≫FP (false negatives much more expensive)
- **Threshold optimization**: Minimize (FN_cost × FN + FP_cost × FP)
- **Expected loss**: S × Impact for risk quantification
- **Operations reports**: Alert summaries above threshold

### Policy Components
- **Threshold**: Cost-optimized decision boundary
- **Cost parameters**: FN/FP cost ratios
- **Impact assessment**: Business impact integration
- **Version tracking**: Model, calibration, and lookup versions

## Ethical Considerations

### Bias Assessment
- Potential bias towards certain ransomware families
- Geographic bias possible due to data collection
- Temporal bias from training data timeframe
- Banking-specific bias in cost structure

### Fairness
- Model performance varies by ransomware family
- Some families may be underrepresented
- Continuous monitoring required for fairness
- Out-of-family generalization testing

### Privacy
- No personal data used in training
- Only static analysis features
- No behavioral or network data
- PE file metadata only

## Limitations

### Technical Limitations
- PE executables only
- Static analysis features only
- No dynamic analysis capabilities
- Limited to known ransomware families
- Requires valid PE file structure

### Performance Limitations
- False positive rate: ~5%
- False negative rate: ~2% (banking-optimized)
- Requires manual review for high-stakes decisions
- Not suitable for real-time applications
- Performance varies by family

### Operational Limitations
- Requires trained personnel for interpretation
- Regular retraining recommended
- Performance monitoring required
- Integration with existing security tools needed
- Banking-specific cost structure

## Maintenance

### Retraining Schedule
- Quarterly retraining recommended
- Performance monitoring monthly
- Data drift detection weekly
- Model validation before deployment
- Calibration check after ensemble updates

### Monitoring
- Performance metrics tracking
- Data drift detection
- Model degradation monitoring
- Feedback collection from users
- Out-of-family performance tracking

### Updates
- Version control for all model artifacts
- Change management process
- Rollback procedures
- Documentation updates
- Mapping version tracking

## Contact

### Support
- Technical issues: [support@aicra.org]
- Model questions: [research@aicra.org]
- Security concerns: [security@aicra.org]

### Documentation
- User guide: [docs.aicra.org]
- API documentation: [api.aicra.org]
- Research papers: [papers.aicra.org]

## Legal

### License
MIT License - see LICENSE file for details

### Disclaimer
This model is provided for research and educational purposes. Users are responsible for ensuring compliance with applicable laws and regulations. The authors disclaim any liability for misuse or damage resulting from the use of this model.

### Citation
```
@software{aicra2024,
  title={AICRA: AI Cyber Risk Advisor},
  author={AICRA Team},
  year={2024},
  url={https://github.com/aicra/aicra}
}
```
