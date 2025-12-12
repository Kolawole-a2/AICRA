# Threshold Optimization & Calibration: Novelty Beyond Standard Cost-Optimization

## Overview

AICRA's threshold optimization goes beyond generic ROC/PR-based threshold selection by encoding **banking-specific operational constraints** and integrating with **risk-based decision theory** aligned with MITRE ATT&CK / D3FEND frameworks.

## Standard Approach (Baseline)

Generic cost-optimization minimizes:

```
Expected Cost = C_FN × P(FN) + C_FP × P(FP)
```

Where:
- `C_FN` = cost of false negative
- `C_FP` = cost of false positive
- `P(FN)` = probability of false negative at threshold `t`
- `P(FP)` = probability of false positive at threshold `t`

This is standard and well-established in ML literature.

## AICRA's Novel Contributions

### 1. Banking-Specific Cost Asymmetry

AICRA encodes **regulatory and operational constraints** specific to banking:

- **False Negative Cost (C_FN):** $5,000,000 (ransomware breach impact in banking)
- **False Positive Cost (C_FP):** $1 (analyst review time)

**Ratio:** C_FN / C_FP = 5,000,000:1

This asymmetry is **not generic** - it reflects:
- Regulatory penalties for missed ransomware detections
- Operational impact of ransomware on critical banking infrastructure
- Cost of SOC analyst time for false positives

### 2. Expected Loss Integration

AICRA's threshold optimization operates on **Expected Loss**, not just classification cost:

```
Expected Loss = p(ransomware) × Impact
```

Where:
- `p(ransomware)` = calibrated susceptibility score S ∈ [0,1]
- `Impact` = asset-specific or scenario-specific impact (default: $5M for banking)

The optimal threshold `t*` minimizes:

```
E[Loss] = Σ_i [p_i × Impact_i × I(p_i ≥ t*) × (1 - y_i)] + C_FP × I(p_i ≥ t*) × y_i
```

Where:
- `I(·)` = indicator function
- `y_i` = true label (1 = ransomware, 0 = benign)
- `p_i` = calibrated probability for sample `i`
- `Impact_i` = impact for sample `i` (can vary by asset class)

### 3. Risk Register Alignment

Thresholds map directly to **action tiers** in ATT&CK-D3FEND risk registers:

- **High Risk (S ≥ 0.8):** Immediate containment, full D3FEND control suite
- **Medium Risk (0.5 ≤ S < 0.8):** Enhanced monitoring, selective controls
- **Low Risk (S < 0.5):** Standard monitoring, baseline controls

This alignment ensures:
- **Auditability:** Threshold decisions are traceable to risk policy
- **Actionability:** Risk scores map to prescriptive controls
- **Regulatory compliance:** Decisions align with banking risk frameworks

### 4. Calibration for SIEM Transferability

AICRA uses **Isotonic calibration** to produce **SIEM-ready susceptibility scores**:

- Calibrated scores `S ∈ [0,1]` are **well-calibrated probabilities**
- Low ECE (Expected Calibration Error) ensures scores are **reliable for operational use**
- Temporal calibration evaluation detects **drift** over time

This is **novel** in the context of:
- **Transferability:** Scores work in SIEM pipelines without re-calibration
- **Temporal robustness:** Calibration stability over time windows

## Formula Summary

**Optimal Threshold Selection:**

```
t* = argmin_t [Σ_i (p_i × Impact_i × I(p_i ≥ t) × (1 - y_i)) + C_FP × I(p_i ≥ t) × y_i]
```

**Expected Loss at Threshold `t`:**

```
E[Loss | t] = (1/N) × [Σ_i (p_i × Impact_i × I(p_i ≥ t) × (1 - y_i)) + C_FP × Σ_i (I(p_i ≥ t) × y_i)]
```

Where:
- `N` = total number of samples
- `Impact_i` = $5,000,000 for banking ransomware (default)
- `C_FP` = $1 (analyst review cost)

## Code Locations

- **Threshold Optimization:** `aicra/pipelines/cost_optimization.py`, `aicra/experiments/h2_calibration_thresholds.py`
- **Expected Loss:** `aicra/experiments/h2_calibration_thresholds.py:93-116`
- **Risk Register Mapping:** `aicra/register.py`, `aicra/pipelines/policy.py`
- **Calibration:** `aicra/pipelines/calibration.py`

## References

- Cost-sensitive learning: Elkan (2001), "The Foundations of Cost-Sensitive Learning"
- Calibration: Guo et al. (2017), "On Calibration of Modern Neural Networks"
- Risk-based decision theory: Raiffa & Schlaffer (1961), "Applied Statistical Decision Theory"

