# H2 Multi-Split Evaluation Proposal

## Current Issue

**H2 currently evaluates on a single test set (10,001 samples)**, while **H3 evaluates across multiple splits** (main, small_ember, full_ember, smoke_test). This inconsistency may raise questions during praxis defense.

## Why This Matters for Praxis Defense

1. **Consistency**: H3 shows multi-split evaluation for robustness - H2 should follow the same pattern
2. **Robustness**: Single-split evaluation may not capture variability across different data distributions
3. **Scientific Rigor**: Multi-split evaluation with aggregation provides stronger evidence
4. **Comparability**: H2 and H3 should use similar evaluation methodologies

## Proposed Solution

### Option 1: Multi-Split Evaluation (Recommended)

**Approach**: Evaluate H2 across multiple splits (same as H3), using a single calibrator trained on the validation set.

**Benefits**:
- Consistent with H3 methodology
- Shows robustness across different data sizes
- Provides aggregated metrics with confidence intervals
- More defensible for praxis

**Implementation**:
1. Train calibrator once on validation set (as currently done)
2. Evaluate calibrated predictions on multiple test splits:
   - `main` (10,000 samples)
   - `small_ember` (2,000 samples)  
   - `full_ember` (50,006 samples)
   - `smoke_test` (200 samples)
3. Compute metrics per split
4. Aggregate metrics across splits with bootstrap confidence intervals
5. Generate per-split and aggregated results

### Option 2: Keep Single-Split (Current)

**Approach**: Keep H2 as single-split evaluation, but document the rationale.

**Rationale** (if choosing this):
- H2 focuses on calibration methodology, not dataset robustness
- Single large test set (10,001 samples) is sufficient for calibration metrics
- H3 needs multi-split because it evaluates mapping consistency across different technique distributions
- Different hypotheses may require different evaluation strategies

**Defense Strategy**:
- Explain that H2 tests calibration methodology on a single, representative test set
- H3 tests mapping consistency across different data distributions (requires multiple splits)
- Both approaches are valid for their respective hypotheses

## Recommendation

**I recommend Option 1 (Multi-Split Evaluation)** because:

1. **Consistency**: Aligns H2 with H3's evaluation methodology
2. **Robustness**: Shows calibration works across different data sizes
3. **Defensibility**: Easier to defend when asked "Why does H3 use multiple splits but H2 doesn't?"
4. **Scientific Rigor**: Multi-split evaluation with aggregation is more rigorous

## Implementation Plan

1. Create `config/h2_splits.yaml` (similar to `config/h3_splits.yaml`)
2. Modify `aicra/experiments/h2_calibration_thresholds.py` to:
   - Load multiple test splits
   - Evaluate each split with the same calibrator
   - Aggregate metrics across splits
   - Generate per-split and aggregated results
3. Update documentation to reflect multi-split evaluation

## Expected Output Structure

```
results/H2_calibration_thresholds/
├── H2_full_results.json
│   ├── per_split_results: [
│   │   {split: "main", metrics: {...}},
│   │   {split: "small_ember", metrics: {...}},
│   │   {split: "full_ember", metrics: {...}},
│   │   {split: "smoke_test", metrics: {...}}
│   │ ]
│   └── aggregated_metrics: {
│       mean_brier_improvement: ...,
│       mean_ece_improvement: ...,
│       bootstrap_ci_95: {...}
│   }
└── H2_summary.md
```

