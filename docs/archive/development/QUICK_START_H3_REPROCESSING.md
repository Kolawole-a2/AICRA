# Quick Start: H3 Reprocessing

## Correct PowerShell Commands

### Step 1: Regenerate Learned Mapping

**Single line (recommended for PowerShell):```powershell
python -m aicra.mappings.heuristic_mapping --attack data/ontology/attack_techniques.csv --d3fend data/ontology/d3fend_controls.csv --out data/mappings/learned_mapping.csv --top-k 10 --min-similarity 0.25
```

**Or use the regeneration script:```powershell
python scripts/regenerate_learned_mapping.py
```

**Or with PowerShell line continuation (backticks):```powershell
python -m aicra.mappings.heuristic_mapping `
  --attack data/ontology/attack_techniques.csv `
  --d3fend data/ontology/d3fend_controls.csv `
  --out data/mappings/learned_mapping.csv `
  --top-k 10 `
  --min-similarity 0.25
```

**Important Notes:- Use `aicra.mappings.heuristic_mapping` (with 's') - NOT `aicra.mapping.heuristic_mapping`
- In PowerShell, use backticks (`) for line continuation, NOT backslashes (\)
- Or just put everything on one line

### Step 2: Validate Mapping Breadth (Optional)

```powershell
python scripts/validate_mapping_breadth.py
```

### Step 3: Run H3 Evaluation

**Single line:```powershell
python -m aicra.experiments.h3_evaluation --config config/h3_splits.yaml --deterministic data/mappings/deterministic_attack_defense_lookup.csv --learned data/mappings/learned_mapping.csv --output results/H3_full_evaluation
```

**Or with line continuation:```powershell
python -m aicra.experiments.h3_evaluation `
  --config config/h3_splits.yaml `
  --deterministic data/mappings/deterministic_attack_defense_lookup.csv `
  --learned data/mappings/learned_mapping.csv `
  --output results/H3_full_evaluation
```

**Or use the complete pipeline script:```powershell
python scripts/reprocess_h3_deliverables.py
```

---

## What Changed

1. **Module Path:** Use `aicra.mappings.heuristic_mapping` (with 's'), not `aicra.mapping.heuristic_mapping`
2. **PowerShell Syntax:** Use backticks (`) for line continuation, not backslashes (\)
3. **Default Parameters:** Now `top_k=10`, `min_similarity=0.25` (broader, noisier)
4. **Sanity Check:** Automatically validates learned mapping is broader than deterministic

---

## Expected Output

After Step 1, you should see:
- `data/mappings/learned_mapping.csv` generated
- Sanity check passes (learned has MORE pairs than deterministic)
- Log messages confirming validation

After Step 3, you should see:
- `results/H3_full_evaluation/H3_full_results.json` with `mapping_behavior` field
- `results/H3_full_evaluation/H3_full_summary.md` with narrative
- Plots in `results/H3_full_evaluation/plots/`

---

## Troubleshooting

**Error: ModuleNotFoundError for 'aicra.mapping'→ Use `aicra.mappings` (with 's') instead

**Error: PowerShell syntax error with backslashes→ Use backticks (`) or put command on one line

**Error: Sanity check fails→ Increase `--top-k` (e.g., 12) or decrease `--min-similarity` (e.g., 0.20)
