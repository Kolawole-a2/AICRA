# Impact Value Update Summary

**Date:** 2025-01-27  
**Change:** Updated default ransomware breach impact to $5,000,000 for banking

---

## Changes Made

### 1. Configuration (`aicra/config.py`)
- ✅ Already set: `impact_default: float = 5000000.0` (line 47)

### 2. Policy Pipeline (`aicra/pipelines/policy.py`)
- ✅ Updated: `scenario_impacts["ransomware_encryption"]` from 10,000,000 to **5,000,000- Location: Line 180 in default banking configuration

### 3. Policy Writer (`aicra/utils/policy_writer.py`)
- ✅ Updated: `scenario_impacts["ransomware_encryption"]` from 10,000,000 to **5,000,000- Location: Line 102 in scenario-based impact selection
- ✅ Already set: `--impact` default argument is `5_000_000` (line 29)

---

## Expected Loss Calculation

The Expected Loss formula is consistently implemented as:
```
Expected Loss = p(ransomware) × Impact
```

Where:
- `p(ransomware)` = susceptibility score (calibrated probability) ∈ [0, 1]
- `Impact` = $5,000,000 for banking ransomware breaches

### Implementation Locations:

1. **`aicra/register.py` (line 65):   ```python
   df["expected_loss"] = df["susceptibility"] * float(impact)
   ```
   Uses `policy.impact_default` which is 5,000,000

2. **`aicra/utils/policy_writer.py` (line 106):   ```python
   df["expected_loss"] = df["susceptibility"] * impact
   ```
   Uses scenario-based impact (5M for ransomware_encryption) or default (5M)

3. **`aicra/pipelines/policy.py` (line 82):   ```python
   return susceptibility_scores * impact_values
   ```
   Uses `self.settings.impact_default` which is 5,000,000

---

## Verification

All Expected Loss calculations now use **$5,000,000** as the default impact for banking ransomware breaches:

- ✅ Config default: 5,000,000
- ✅ Policy default: 5,000,000  
- ✅ Scenario impact (ransomware_encryption): 5,000,000
- ✅ Policy writer default: 5,000,000

--**Update Complete** ✅

