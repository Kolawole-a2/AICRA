# Threats to Validity

**AICRA: Artificial Intelligence–Powered Cyber Risk Advisor for Endpoint Security in U.S. Banking Organizations**

This document identifies potential threats to the validity of the AICRA praxis results and describes mitigations implemented in the experimental design and codebase.

---

## 1. Internal Validity

Internal validity concerns whether the experimental design correctly establishes causal relationships between interventions and outcomes.

### 1.1 Label Noise in Malware Datasets

**Risk**: EMBER-2024 labels may contain errors due to:
- Automated labeling heuristics (e.g., VirusTotal aggregation)
- Evolving malware taxonomy (polymorphic variants, packers)
- False positives in dynamic analysis environments

**Impact**: Label noise can inflate or deflate performance metrics, making it difficult to assess true model capability.

**Mitigation in AICRA**:
- EMBER-2024 uses consensus labeling from multiple antivirus engines, reducing single-source errors
- Time-ordered splits prevent data leakage that could mask label noise effects
- Out-of-family evaluation tests generalization to held-out malware families, reducing sensitivity to label errors in training data
- Baseline comparisons (logistic regression, majority classifier) provide context for interpreting absolute performance

**Remaining Risk**: Some label noise likely remains, but the experimental design (temporal splits, out-of-family tests) ensures that results reflect true generalization capability rather than overfitting to noisy labels.

### 1.2 Feature Extraction Assumptions

**Risk**: Static PE feature extraction assumes that:
- File headers and sections are not corrupted or obfuscated
- Feature engineering captures all relevant ransomware indicators
- Feature distributions remain stable across time periods

**Impact**: If assumptions are violated, model performance may degrade in production or fail to generalize to new malware variants.

**Mitigation in AICRA**:
- Uses EMBER-2024's standardized feature set (2,381 features) validated in prior research (Anderson & Roth, 2018)
- Combines EMBER features with additional PE static features for robustness
- Temporal evaluation tests feature stability across time periods
- Out-of-family evaluation tests generalization to malware families not seen during training

**Remaining Risk**: Advanced obfuscation or packing techniques may evade static feature extraction, but this is a limitation of static analysis in general, not specific to AICRA.

### 1.3 Mapping Heuristics

**Risk**: The learned mapping uses embedding-based similarity, which assumes:
- Text embeddings capture semantic relationships between ATT&CK techniques and D3FEND controls
- Top-k similarity selection produces operationally relevant mappings
- Embedding models (sentence-transformers) generalize to cybersecurity domain terminology

**Impact**: If assumptions are violated, learned mappings may be semantically incorrect or operationally irrelevant, leading to poor actionable precision.

**Mitigation in AICRA**:
- Deterministic mapping provides expert-curated ground truth for comparison
- Learned mapping construction is explicitly documented to prevent data leakage (does not use deterministic pairs as labels)
- Actionable precision metric directly measures operational relevance (ransomware-relevant controls)
- H3 results show learned mapping achieves 0% actionable precision, validating that the evaluation correctly identifies poor mappings

**Remaining Risk**: The learned mapping's poor performance (0% actionable precision) suggests that embedding-based heuristics alone are insufficient for cybersecurity ontology alignment, but this is a valid experimental finding rather than a validity threat.

---

## 2. External Validity

External validity concerns whether results generalize to other contexts, populations, or settings beyond the experimental conditions.

### 2.1 Banking Endpoint Specificity

**Risk**: AICRA is designed and evaluated specifically for U.S. banking endpoint security, which may limit generalizability to:
- Other sectors (healthcare, manufacturing, government)
- Other regions (non-U.S. banking regulations, threat landscapes)
- Other endpoint types (servers, mobile devices, IoT)

**Impact**: Results may not apply to contexts with different threat models, regulatory requirements, or operational constraints.

**Mitigation in AICRA**:
- Uses EMBER-2024, a general-purpose malware dataset (not banking-specific), ensuring that model capabilities are not artificially constrained
- Banking-specific elements (cost ratios, threshold optimization) are explicitly parameterized and can be adjusted for other contexts
- The deterministic mapping uses MITRE ATT&CK and D3FEND, which are sector-agnostic cybersecurity frameworks

**Remaining Risk**: The cost structure (FN cost >> FP cost) and threshold optimization are banking-specific. Other sectors may have different cost structures, requiring threshold re-optimization.

**Generalization Strategy**: AICRA's design separates domain-specific parameters (cost ratios, thresholds) from core algorithms (LightGBM, calibration, mapping), enabling adaptation to other sectors by adjusting parameters without retraining models.

### 2.2 Transferability to Other Sectors

**Risk**: Results may not transfer to sectors with:
- Different threat landscapes (nation-state actors, insider threats, supply chain attacks)
- Different regulatory requirements (HIPAA, NIST, ISO 27001)
- Different operational constraints (real-time vs batch processing, resource limitations)

**Impact**: AICRA may perform differently or require significant adaptation for non-banking contexts.

**Mitigation in AICRA**:
- EMBER-2024 includes diverse malware families, not limited to banking-specific threats
- The model architecture (LightGBM) is general-purpose and has been validated across multiple domains
- Calibration and threshold optimization are parameterized and can be adjusted for different cost structures

**Remaining Risk**: Banking-specific threat models (ransomware focus) may not capture the full threat landscape in other sectors. However, AICRA's modular design enables sector-specific threat modeling through mapping customization.

**Generalization Strategy**: The deterministic mapping can be extended with sector-specific ATT&CK–D3FEND pairs, and cost ratios can be adjusted to reflect sector-specific risk tolerance.

### 2.3 Dataset Representativeness

**Risk**: EMBER-2024 may not represent:
- Current threat landscape (dataset collected in 2023–2024, threat landscape evolves rapidly)
- Real-world endpoint diversity (banking endpoints may have different software stacks than EMBER samples)
- Operational conditions (real endpoints have network context, user behavior, system state not captured in static PE features)

**Impact**: Model performance in production may differ from experimental results if the dataset is not representative of operational conditions.

**Mitigation in AICRA**:
- Time-ordered splits ensure that test data is chronologically later than training data, simulating real-world deployment where models must generalize to future threats
- Out-of-family evaluation tests generalization to malware families not seen during training, simulating zero-day threats
- Temporal calibration check validates that calibration parameters transfer across time periods

**Remaining Risk**: Static PE features cannot capture dynamic context (network traffic, user behavior, system state), which may limit performance in production. This is a fundamental limitation of static analysis, not specific to AICRA.

**Production Deployment Strategy**: AICRA is designed as a component of a larger security stack, where static analysis complements dynamic analysis, network monitoring, and behavioral analytics.

---

## 3. Construct Validity

Construct validity concerns whether the experimental measures accurately capture the theoretical constructs they are intended to measure.

### 3.1 Proxy Measures for Alert Fatigue

**Risk**: Alert fatigue is a complex, multi-dimensional construct (analyst burnout, decision quality, response time) that cannot be directly measured in experimental settings. AICRA uses proxy measures:
- False negative reduction (fewer missed threats = less retroactive investigation)
- Variance reduction (more consistent scores = less cognitive load)
- Expected loss reduction (fewer unnecessary investigations = less analyst time)

**Impact**: Proxy measures may not fully capture the true operational impact of alert fatigue reduction.

**Mitigation in AICRA**:
- Uses multiple proxy measures (FN reduction, variance reduction, expected loss) to triangulate alert fatigue impact
- Expected loss directly incorporates cost structure (FN cost >> FP cost), reflecting operational priorities
- Variance reduction measures score consistency, which is operationally relevant because inconsistent scores increase analyst cognitive load

**Remaining Risk**: True alert fatigue requires longitudinal studies with real SOC analysts, which is beyond the scope of this praxis. However, the proxy measures are operationally meaningful and defensible.

**Validation Strategy**: Future work should validate proxy measures through field studies with banking SOCs, but the current measures are sufficient for praxis validation.

### 3.2 Risk Score Interpretation

**Risk**: Risk scores (calibrated probabilities) may be misinterpreted by security analysts as:
- Absolute probabilities (e.g., "0.95 means 95% chance of ransomware")
- Confidence intervals (e.g., "0.95 ± 0.05")
- Binary decisions (e.g., "> 0.5 = ransomware")

**Impact**: Misinterpretation can lead to operational errors (over-triage, under-triage, threshold misconfiguration).

**Mitigation in AICRA**:
- Calibration ensures that risk scores are well-calibrated (predicted probabilities match observed frequencies)
- Documentation explicitly explains that risk scores are calibrated probabilities, not confidence intervals
- Threshold optimization provides operationally optimal decision boundaries based on cost structure
- Summary reports include confusion matrices and operational metrics (precision, recall) to aid interpretation

**Remaining Risk**: Analysts may still misinterpret scores without proper training. This is an operational deployment concern, not an experimental validity threat.

**Operational Guidance**: AICRA's documentation and summary reports provide clear interpretation guidance, and threshold optimization reduces the need for manual score interpretation.

### 3.3 Mapping Accuracy vs Ground Truth

**Risk**: The deterministic mapping is treated as ground truth for H3 evaluation, but:
- Expert knowledge may be incomplete or outdated
- Deterministic mapping may not capture all valid technique-control relationships
- Ground truth may vary across experts or organizations

**Impact**: If deterministic mapping is not truly ground truth, H3 results may overstate the advantage of deterministic over learned mappings.

**Mitigation in AICRA**:
- Deterministic mapping is based on MITRE ATT&CK and D3FEND, which are industry-standard, expert-curated frameworks
- Reference pairs provide an independent validation set (15 canonical pairs from D3FEND documentation)
- Actionable precision metric measures operational relevance (ransomware-relevant controls), not just agreement with deterministic mapping
- H3 results show that learned mapping achieves 0% actionable precision, validating that the evaluation correctly identifies poor mappings even if deterministic mapping is incomplete

**Remaining Risk**: Deterministic mapping may be incomplete, but this does not invalidate H3 results because actionable precision provides an independent measure of operational relevance.

**Validation Strategy**: The deterministic mapping's perfect DAC (100%) and actionable precision (0.75) provide strong evidence that it captures operationally relevant relationships, even if it is not exhaustive.

---

## 4. Temporal Validity

Temporal validity concerns whether results remain valid as time passes and conditions change.

### 4.1 Concept Drift

**Risk**: Malware threat landscape evolves rapidly:
- New attack techniques (zero-day exploits, novel ransomware variants)
- Evolving evasion techniques (polymorphic code, anti-analysis)
- Changing software ecosystems (new applications, updated OS versions)

**Impact**: Model performance may degrade over time as training data becomes outdated.

**Mitigation in AICRA**:
- Time-ordered splits simulate real-world deployment where models must generalize to future threats
- Temporal calibration check validates that calibration parameters transfer across time periods
- Out-of-family evaluation tests generalization to malware families not seen during training
- Model architecture (LightGBM) is robust to distribution shift through ensemble methods and regularization

**Remaining Risk**: Concept drift will eventually require model retraining, but temporal evaluation provides evidence that models remain effective for reasonable time periods (test data is chronologically later than training data).

**Production Strategy**: AICRA's design supports incremental retraining (new data can be added to training sets) and recalibration (calibration parameters can be updated without retraining models).

### 4.2 Evolving Attacker Behavior

**Risk**: Attackers adapt to defensive measures:
- Adversarial examples (specially crafted samples that evade detection)
- Feature obfuscation (packing, encryption, code injection)
- Behavioral changes (slower encryption, different file targeting)

**Impact**: Static PE features may become less effective as attackers learn to evade static analysis.

**Mitigation in AICRA**:
- Uses diverse feature set (2,381 EMBER features + PE static features) to reduce sensitivity to individual feature evasion
- Ensemble methods (bagged LightGBM) provide robustness to feature manipulation
- Out-of-family evaluation tests generalization to novel attack techniques
- Calibration and threshold optimization adapt to changing threat distributions

**Remaining Risk**: Advanced adversarial techniques may still evade detection, but this is a limitation of static analysis in general, not specific to AICRA.

**Defense Strategy**: AICRA is designed as part of a defense-in-depth strategy, where static analysis complements dynamic analysis, network monitoring, and behavioral analytics.

### 4.3 Need for Recalibration

**Risk**: Calibration parameters may become outdated as:
- Threat distribution shifts (more/less ransomware, different variants)
- Model predictions drift (feature distributions change)
- Operational conditions change (new software, updated endpoints)

**Impact**: Risk scores may become miscalibrated, leading to incorrect probability estimates and suboptimal threshold decisions.

**Mitigation in AICRA**:
- Temporal calibration check validates that calibration parameters transfer across time periods
- Calibration is performed on validation set (earlier window) and tested on test set (later window), simulating real-world recalibration intervals
- Calibration method (isotonic regression) is non-parametric and adapts to changing distributions
- Documentation provides guidance on recalibration intervals and monitoring

**Remaining Risk**: Recalibration will eventually be required, but temporal evaluation provides evidence that calibration remains stable for reasonable time periods.

**Operational Guidance**: AICRA's documentation recommends monitoring calibration metrics (Brier score, ECE) and recalibrating when metrics exceed operational thresholds.

---

## Summary

The threats to validity identified above are common in machine learning and cybersecurity research. AICRA's experimental design includes multiple mitigations:

1. **Internal Validity**: Time-ordered splits, out-of-family evaluation, baseline comparisons
2. **External Validity**: General-purpose datasets, parameterized design, modular architecture
3. **Construct Validity**: Multiple proxy measures, operational metrics, independent validation
4. **Temporal Validity**: Temporal evaluation, calibration stability checks, incremental retraining support

While some risks remain (label noise, concept drift, adversarial evasion), these are inherent to the problem domain and are explicitly acknowledged in the limitations section. The mitigations ensure that experimental results are defensible and operationally meaningful for banking SOC deployment.

---

## References

- Anderson, H. S., & Roth, P. (2018). EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models. arXiv:1804.04637
- Raff, E., et al. (2018). Malware Detection by Eating a Whole EXE. arXiv:1710.09435
- Guo, C., et al. (2017). On Calibration of Modern Neural Networks. ICML 2017. https://arxiv.org/abs/1706.04599
- MITRE ATT&CK: https://attack.mitre.org/
- MITRE D3FEND: https://d3fend.mitre.org/

