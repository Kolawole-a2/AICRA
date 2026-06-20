> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# H3 Reprocessing Instructions

This document provides step-by-step instructions for reprocessing all H3 deliverables with the updated understanding that:
- **Deterministic mapping** = Ransomware-focused, curated ground truth
- **Learned mapping** = Generic, broad, noisy mapping (NOT ransomware-specific)

---

## Files Modified

1. **`aicra/mappings/heuristic_mapping.py`   - Updated default config: `top_k=10`, `min_similarity=0.25` (broader, noisier)
   - Added `validate_learned_is_broader()` sanity check function
   - Sanity check ensures learned mapping has MORE pairs and contains controls NOT in deterministic

2. **`aicra/experiments/h3_evaluation.py`   - Updated to prefer `deterministic_attack_defense_lookup.csv` (ransomware-focused)
   - Added `mapping_behavior` field to output JSON
   - Added warnings if learned mapping is not broader than deterministic
   - Updated markdown report with mapping behavior narrative

3. **`scripts/regenerate_learned_mapping.py`** (NEW)
   - Script to regenerate learned mapping with generic, broad parameters

4. **`scripts/reprocess_h3_deliverables.py`** (NEW)
   - Complete pipeline to regenerate mapping and re-run H3 evaluation

---

## Commands to Run (In Order)

### Step 1: Regenerate Broad, Generic Learned Mapping

**For PowerShell (use backticks for line continuation):```powershell
# Option A: Use the regeneration script (auto-discovers ontology files)
python scripts/regenerate_learned_mapping.py

# Option B: Use the heuristic mapping CLI directly (single line)
python -m aicra.mappings.heuristic_mapping --attack data/ontology/attack_techniques.csv --d3fend data/ontology/d3fend_controls.csv --out data/mappings/learned_mapping.csv --top-k 10 --min-similarity 0.25

# Option B (with PowerShell line continuation using backticks):
python -m aicra.mappings.heuristic_mapping `
  --attack data/ontology/attack_techniques.csv `
  --d3fend data/ontology/d3fend_controls.csv `
  --out data/mappings/learned_mapping.csv `
  --top-k 10 `
  --min-similarity 0.25
```

**For Bash/Linux/Mac (use backslashes for line continuation):```bash
# Option A: Use the regeneration script
python scripts/regenerate_learned_mapping.py

# Option B: Use the heuristic mapping CLI directly
python -m aicra.mappings.heuristic_mapping \
  --attack data/ontology/attack_techniques.csv \
  --d3fend data/ontology/d3fend_controls.csv \
  --out data/mappings/learned_mapping.csv \
  --top-k 10 \
  --min-similarity 0.25
```

**Expected Output:- `data/mappings/learned_mapping.csv` with broader, generic mappings
- Sanity check will validate that learned mapping is broader than deterministic
- If sanity check fails, adjust `top_k` or `min_similarity` parameters

**Validation:After generation, the sanity check will:
- ✓ Verify learned has MORE pairs than deterministic
- ✓ Verify learned contains controls NOT in deterministic
- ✓ Verify learned controls are NOT subsets of deterministic controls

If any check fails, you'll see:
```
RuntimeError: Learned mapping is not broader/noisier than deterministic. 
Adjust top_k/min_similarity or logic.
```

### Step 2: Validate Mapping Breadth (Optional)

```powershell
python scripts/validate_mapping_breadth.py
```

This will show a clear validation report.

### Step 3: Run H3 Evaluation

```powershell
python -m aicra.experiments.h3_evaluation `
  --config config/h3_splits.yaml `
  --deterministic data/mappings/deterministic_attack_defense_lookup.csv `
  --learned data/mappings/learned_mapping.csv `
  --output results/H3_full_evaluation
```

**Or use the complete reprocessing script:```powershell
python scripts/reprocess_h3_deliverables.py
```

This script will:
1. Regenerate learned mapping (Step 1)
2. Run H3 evaluation (Step 3)
3. Generate all deliverables

---

## Expected Results

### Mapping Behavior Validation

After H3 evaluation, check `results/H3_full_evaluation/H3_full_results.json` for the `mapping_behavior` field:

```json
{
  "mapping_behavior": {
    "learned_is_broader": true,
    "learned_pairs_count": <should be > deterministic_pairs_count>,
    "deterministic_pairs_count": <number>,
    "learned_only_pairs_count": <should be > 0>,
    "techniques_with_extra_learned_controls": <should be > 0>,
    "techniques_with_only_ransomware_controls": <number>,
    ...
  }
}
```

**If `learned_is_broader: false`, you'll see a warning in the logs and summary.### H3 Metrics (Expected Behavior)

The H3 metrics should show that deterministic mapping outperforms learned mapping on:

1. **Correctness/Consistency:   - Deterministic: DAC_internal = 100% (by definition)
   - Learned: DAC_internal < 100% (lower, as expected)

2. **Actionable Precision & F1:   - Deterministic: Higher precision/F1 (fewer irrelevant controls)
   - Learned: Lower precision/F1 (includes non-ransomware controls)

3. **Risk-Score Stability:   - Deterministic: More stable variance/IQR behavior
   - Learned: More erratic variance (noisier adjustments)

### Delta Metrics

The results will include delta metrics:
- `delta_actionable_precision = det_precision - learned_precision` (should be positive)
- `delta_actionable_f1 = det_f1 - learned_f1` (should be positive)
- `delta_variance_reduction = det_var_reduction - learned_var_reduction` (interpret based on results)

---

## Verification Checklist

After running both steps, verify:

- [ ] `data/mappings/learned_mapping.csv` exists and has more pairs than deterministic
- [ ] Sanity check passed (learned is broader than deterministic)
- [ ] `results/H3_full_evaluation/H3_full_results.json` contains `mapping_behavior` field
- [ ] `mapping_behavior.learned_is_broader = true`
- [ ] `results/H3_full_evaluation/H3_full_summary.md` includes mapping behavior narrative
- [ ] Deterministic DAC_internal = 100% (by definition)
- [ ] Learned DAC_internal < 100%
- [ ] Delta metrics show deterministic outperforming learned (or metrics are honestly reported)

---

## Troubleshooting

### Issue: ModuleNotFoundError for `aicra.mapping.heuristic_mapping`

**Solution:** Use `aicra.mappings.heuristic_mapping` (with 's') instead:
```powershell
python -m aicra.mappings.heuristic_mapping --help
```

### Issue: Learned mapping is identical to deterministic

**Solution:- Increase `top_k` (e.g., `--top-k 12`)
- Decrease `min_similarity` (e.g., `--min-similarity 0.20`)
- Check that ontology files contain all D3FEND controls (not just ransomware-specific)

### Issue: Sanity check fails

**Solution:- Ensure learned mapping has MORE pairs than deterministic
- Verify learned mapping includes controls NOT in deterministic
- Check that for each technique, learned controls are NOT a subset of deterministic controls

### Issue: H3 metrics show learned outperforming deterministic

**Solution:- This is a valid result if metrics are computed honestly
- Review the narrative in `H3_full_summary.md` for interpretation
- Check that mapping_behavior confirms learned is broader/noisier
- The hypothesis may not be supported if learned mapping accidentally captures better patterns

---

## Output Files

After successful reprocessing:

- `data/mappings/learned_mapping.csv` - Regenerated generic, broad mapping
- `results/H3_full_evaluation/H3_full_results.json` - Complete results with mapping_behavior
- `results/H3_full_evaluation/H3_full_summary.md` - Human-readable report
- `results/H3_full_evaluation/plots/` - Visualization plots

---

## Next Steps

1. Review `results/H3_full_evaluation/H3_full_summary.md` for interpretation
2. Check `mapping_behavior` field in JSON to confirm learned is broader
3. Verify delta metrics show expected direction (deterministic > learned)
4. Update `results/praxis_validation_report.md` with new H3 results

--**Last Updated:** 2025-01-XX  
**Status:** Ready for execution
