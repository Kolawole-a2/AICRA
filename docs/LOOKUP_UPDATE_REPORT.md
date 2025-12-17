# Deterministic Lookup Update Report


## 1. Lookup Discovery

- **Lookup file**: `C:\Users\KLAMOS\Desktop\My Cursor Projects\AICRA\data\lookups\attack_to_d3fend.yaml`  (techniques=7, controls=14, pairs=21)
- **T1055 present**: YES
- **T1027 present**: YES

## 2. Mapping Changes (summary)

- **New techniques added**: `T1055` (Process Injection), `T1027` (Obfuscated Files or Information) mapped to `D3-FA`, `D3-PM`, `D3-UBA`.
- Existing technique→control mappings were not changed; only new technique entries were added.

## 3. Register Regeneration Summary (by split)

### smoke_test

- **Path**: `C:\Users\KLAMOS\Desktop\My Cursor Projects\AICRA\register\smoke_test\ransomware_only_risk_register.csv`
- **Rows**: 558
- **Unique techniques**: 4
- **Technique IDs**: T1027, T1055, T1059, T1486

### small_ember

- **Path**: `C:\Users\KLAMOS\Desktop\My Cursor Projects\AICRA\register\small_ember\ransomware_only_risk_register.csv`
- **Rows**: 5232
- **Unique techniques**: 6
- **Technique IDs**: T1021, T1027, T1055, T1059, T1486, T1490

### main

- **Path**: `C:\Users\KLAMOS\Desktop\My Cursor Projects\AICRA\register\main\ransomware_only_risk_register.csv`
- **Rows**: 21051
- **Unique techniques**: 6
- **Technique IDs**: T1021, T1027, T1055, T1059, T1486, T1490

### full_ember

- **Path**: `C:\Users\KLAMOS\Desktop\My Cursor Projects\AICRA\register\full_ember\ransomware_only_risk_register.csv`
- **Rows**: 128727
- **Unique techniques**: 6
- **Technique IDs**: T1021, T1027, T1055, T1059, T1486, T1490

## 4. Safety Confirmation

- **H1/H2 training/testing**: NOT rerun (no models retrained; no training scripts modified).
- **risk_scores.csv**: NOT regenerated or modified (all results/*/risk_scores*.csv left untouched).
- **H3 evaluation scripts & results**: NOT changed (results in `results/H3_*` remain as originally computed).