# Adversarial Robustness & Limitations

## Overview

AICRA's static PE feature-based ransomware detection is evaluated for robustness against **feature-level perturbations** and **mimicry attacks**. This document summarizes findings and limitations.

## Evaluation Framework

### 1. Feature-Level Perturbations

**Method:** Add Gaussian or uniform noise to feature vectors within plausible ranges.

**Perturbation Types:**
- **Gaussian:** `x' = x + N(0, σ)` where σ = strength × feature_std
- **Uniform:** `x' = x + U(-strength, +strength)`
- **Mimicry:** Shift ransomware features toward benign distribution mean

**Metrics:**
- AUROC drop under perturbation
- % of samples with classification flips
- % of ransomware samples that evade detection (FN increase)

### 2. Mimicry Attacks

**Method:** Shift ransomware feature distributions toward benign samples to evade detection.

**Attack Model:**
```
x_mimicry = (1 - α) × x_ransomware + α × μ_benign
```

Where:
- `α` = mimicry strength (0.0 = no change, 1.0 = full shift to benign)
- `μ_benign` = mean of benign feature distribution

**Evaluation:**
- Evasion rate: % of ransomware samples classified as benign after mimicry
- Risk score reduction: Mean decrease in susceptibility score

## Findings

### Robustness Characteristics

**Strengths:**
- LightGBM ensemble with multiple seeds provides some robustness to small perturbations
- Static PE features (byte histograms, headers) are less easily manipulated than dynamic features

**Vulnerabilities:**
- **Mimicry attacks:** Ransomware samples can evade detection by shifting features toward benign distribution
- **Feature-level noise:** Large perturbations (>10%) cause significant AUROC drops
- **Static analysis limitation:** Cannot detect runtime behavior changes

## Limitations

1. **No Runtime Analysis:** AICRA uses static PE features only. Adversaries can:
   - Pack/obfuscate binaries to change static features
   - Use benign-looking packers to evade detection
   - Modify PE headers while maintaining malicious runtime behavior

2. **Feature Manipulation:** If attackers know which features are important, they can:
   - Modify entropy values
   - Adjust PE header fields
   - Manipulate byte histograms

3. **Transfer Attacks:** Adversarial examples crafted for one model may transfer to AICRA's LightGBM ensemble.

## Recommendations

1. **Defense-in-Depth:** Combine static analysis (AICRA) with:
   - Dynamic analysis (sandbox execution)
   - Behavioral monitoring (SIEM integration)
   - Network traffic analysis

2. **Adversarial Training:** Retrain models on adversarial examples to improve robustness.

3. **Feature Diversity:** Use multiple feature types (static + dynamic) to reduce single-point-of-failure.

4. **Monitoring:** Track model performance over time to detect evasion attempts.

## Experimental Results

See `results/H1_adversarial/` for detailed results:
- `robustness_results.json`: Feature perturbation results
- `mimicry_results.json`: Mimicry attack results

## References

- Adversarial ML: Goodfellow et al. (2014), "Explaining and Harnessing Adversarial Examples"
- Malware evasion: Anderson et al. (2018), "Learning to Evade Static PE Machine Learning Malware Models"

