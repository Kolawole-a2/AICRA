# H3 Identical Mapping Fix

## Problem

The H3 evaluation is producing identical metrics for deterministic and learned mappings because **the learned mapping file contains the same pairs as the deterministic mapping**.

From `H3_results.json`:
- Both have `pairs_count: 15`
- Both have `coverage_%: 100.0`
- Both have `consistency_%: 100.0`
- All precision, F1, and variance metrics are identical

## Root Cause

The learned mapping file (`learned_mapping.csv` or `data/mappings/learned_embedding_attack_defense_mapping.csv`) contains **exactly the same pairs** as the deterministic mapping file.

## Solution

The RuntimeError check at line 125-145 in `run_h3_experiment.py` should catch this and stop execution with a clear error message. The check has been enhanced to:

1. Print detailed diagnostics showing:
   - Number of pairs in each mapping
   - Intersection count
   - Only-in-deterministic count
   - Only-in-learned count
   - Sample pairs from each mapping
   - File paths being used

2. Raise RuntimeError with a clear message explaining:
   - The learned mapping is identical to deterministic
   - Which files are being compared
   - How to fix it (regenerate the embedding-based learned mapping)

## Next Steps

1. **Run the script** - It should now raise RuntimeError immediately when it detects identical mappings:
   ```bash
   python run_h3_experiment.py
   ```

2. **If RuntimeError is raised**, it means the learned mapping file is identical to deterministic. You need to:
   - Regenerate the embedding-based learned mapping:
     ```bash
     python -m aicra.mappings.embedding_learned_mapping
     ```
   - Verify the new learned mapping is different from deterministic
   - Re-run the H3 experiment

3. **If the script still produces identical metrics without raising RuntimeError**, there's a bug in the evaluation code that needs to be fixed.

## Expected Behavior

When the learned mapping is identical to deterministic:
- The script should **stop immediately** with RuntimeError
- It should **NOT** compute any metrics
- It should **NOT** produce H3_results.json with identical values
- It should print clear diagnostics showing why it stopped

If the learned mapping is different:
- The script should proceed normally
- Metrics should diverge between deterministic and learned
- If metrics are still identical despite different mappings, RuntimeError will be raised at the metric computation stage


























