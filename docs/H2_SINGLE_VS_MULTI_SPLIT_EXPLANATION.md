# H2 Single-Split vs Multi-Split Evaluation: Explanation for Praxis Defense

## Current Situation

**H2 currently evaluates on a single test set (10,001 samples)**, while **H3 evaluates across multiple splits** (main, small_ember, full_ember, smoke_test).

## Is This a Problem for Praxis Defense?

### Short Answer: **It depends on how you frame it**

### Option 1: Defend Single-Split Design (Current)

**Rationale**: H2 and H3 test different things, so different evaluation strategies are appropriate.

**H2 Focus**: Tests **calibration methodology** and **threshold optimization**
- Calibration is a **methodological question**: "Does calibration improve probability estimates?"
- This can be answered on a single, representative test set
- The test set (10,001 samples) is large enough for reliable calibration metrics

**H3 Focus**: Tests **mapping consistency** across **different data distributions**
- Mapping consistency depends on **which techniques appear** in the data
- Different splits have different technique distributions
- Multi-split evaluation is **necessary** to show robustness across technique distributions

**Defense Strategy**:
- "H2 tests calibration methodology on a single, representative test set. This is appropriate because calibration is a methodological question that doesn't depend on specific data distributions."
- "H3 tests mapping consistency across different technique distributions. Multi-split evaluation is necessary because different splits contain different ATT&CK techniques, and we need to show consistency across these distributions."
- "Both evaluation strategies are valid for their respective hypotheses."

### Option 2: Implement Multi-Split Evaluation (Recommended for Consistency)

**Rationale**: For consistency and robustness, H2 should also evaluate across multiple splits.

**Benefits**:
- **Consistency**: Aligns H2 with H3's evaluation methodology
- **Robustness**: Shows calibration works across different data sizes
- **Defensibility**: Easier to defend when asked "Why does H3 use multiple splits but H2 doesn't?"
- **Scientific Rigor**: Multi-split evaluation with aggregation is more rigorous

**Implementation**:
- Train calibrator once on validation set
- Evaluate calibrated predictions on multiple test splits
- Aggregate metrics across splits with confidence intervals
- Show per-split and aggregated results

## Recommendation

**I recommend implementing multi-split evaluation for H2** because:

1. **Consistency**: Makes H1/H2/H3 evaluation methodology consistent
2. **Robustness**: Demonstrates calibration works across different data sizes
3. **Defensibility**: Easier to defend - no need to explain why H2 is different
4. **Scientific Rigor**: Multi-split evaluation with aggregation is more rigorous

## How to Implement

1. Create `config/h2_splits.yaml` (similar to `config/h3_splits.yaml`)
2. Modify H2 experiment to:
   - Load multiple test splits
   - Train calibrator once on validation set
   - Evaluate each split with the same calibrator
   - Aggregate metrics across splits
3. Update documentation

## If You Keep Single-Split

**Defense Strategy**:
- Explain that H2 and H3 test different things
- H2 tests methodology (calibration) - single split is sufficient
- H3 tests consistency across distributions - multi-split is necessary
- Both approaches are valid for their respective hypotheses

**Key Point**: The evaluation strategy should match what you're testing. If you can justify why single-split is appropriate for H2, that's valid. But multi-split is more consistent and easier to defend.

