### H3 Validation — Deterministic ATT&CK–D3FEND Mapping

**Mapping Coverage:** 100.0%  |  Target ≥ 85.0%  |  Pass: True
**Defense–Attack Consistency:** 100.0%  |  Target ≥ 90.0%  |  Pass: True
**Precision Δ (Mapped–Baseline):** 0.0286  |  Target > 0.0  |  Pass: True
**F1 Δ (Mapped–Baseline):** 0.0562
**Variance Reduction:** -0.03872  |  Target > 0.0  |  Pass: False
**IQR Reduction:** -0.14265

#### Interpretation

- If Coverage and Consistency pass AND Precision/Variance improve, H3 is supported.
- Consistency < target suggests mapping drift or mismatched controls; review lookup vs. MITRE reference pairs.
- Low Δ Precision with good consistency may indicate scoring thresholds need tuning.

#### Reproducibility Notes

- Record the exact versions of: attack_d3fend_lookup.csv (include hash), d3fend_reference_pairs.csv, and model checkpoint used to produce scores.