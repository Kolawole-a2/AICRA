# H3 DAC Redefinition - Implementation Summary

## Problem Identified

The original H3 implementation computed DAC against `d3fend_reference_pairs.csv` (15 pairs), which is not aligned with the H3 research design. For H3, the deterministic mapping itself should be the ground truth.

## Solution Implemented

### 1. DAC_internal (H3 Primary Metric)

**Definition:- **DAC_internal_det = 100%** by definition (deterministic vs itself)
- **DAC_internal_learned = |P_det ∩ P_learn| / |P_det| × 100%This measures how well the learned mapping agrees with the deterministic expert ontology.

**Implementation:- New function `compute_dac_internal()` in `aicra/experiments/h3_evaluation.py`
- Computes overlap between learned mapping and deterministic mapping
- Normalized by total deterministic pairs (not learned pairs)

### 2. DAC_external (Secondary Benchmark)

**Definition:- Agreement with external reference pairs (`d3fend_reference_pairs.csv`)
- Computed for both deterministic and learned mappings
- Labeled clearly as secondary benchmark, not H3 primary metric

**Implementation:- Uses existing `compute_dac()` function
- Renamed to `dac_external_%` in output
- Clearly separated in JSON and markdown reports

### 3. Updated JSON Structure

```json
{
  "per_split_results": [
    {
      "deterministic": {
        "mapping_metrics": {
          "dac_internal_%": 100.0,
          "dac_external_%": ...,
          ...
        }
      },
      "learned": {
        "mapping_metrics": {
          "dac_internal_%": ...,
          "dac_external_%": ...,
          ...
        }
      },
      "deltas": {
        "delta_dac_internal_%": ...,
        "delta_dac_external_%": ...,
        ...
      }
    }
  ],
  "aggregated_metrics": {
    "deterministic": {
      "dac_internal_%": {"mean": 100.0, "std": 0.0},
      "dac_external_%": {"mean": ..., "std": ...},
      ...
    },
    "learned": {
      "dac_internal_%": {"mean": ..., "std": ...},
      "dac_external_%": {"mean": ..., "std": ...},
      ...
    },
    "statistical_tests": {
      "dac_internal": {
        "ttest": {...},
        "wilcoxon": {...},
        "spearman_vs_precision": {...},
        "spearman_vs_variance_reduction": {...}
      },
      "dac_external": {
        "ttest": {...},
        "wilcoxon": {...}
      }
    }
  },
  "mapping_overlap": {
    "between_det_and_learned": {...},
    "det_vs_reference": {...},
    "learned_vs_reference": {...}
  }
}
```

### 4. Updated Markdown Report

- **Section 1 (Setup)**: Clearly states deterministic mapping is the normative expert ontology for H3
- **Section 2 (Per-Split Results)**: Shows both DAC_internal and DAC_external
- **Section 3 (Aggregated Findings)**: 
  - Primary section for DAC_internal (H3 primary metric)
  - Secondary section for DAC_external (ontology benchmark)
- **Section 3 (Statistical Tests)**:
  - Primary tests for DAC_internal
  - Secondary tests for DAC_external
  - Spearman correlations use DAC_internal_learned
- **Section 5 (Conclusion)**: Uses DAC_internal as primary metric

### 5. Updated Plots

- `dac_internal_per_split.png`: Shows DAC_internal per split (replaces `dac_per_split.png`)
- Summary plots use DAC_internal as primary metric

### 6. Enhanced Mapping Overlap Metrics

- `between_det_and_learned`: Jaccard similarity, exact match techniques, etc.
- `det_vs_reference`: Overlap with external reference pairs
- `learned_vs_reference`: Overlap with external reference pairs

## Files Modified

1. **`aicra/experiments/h3_evaluation.py`**:
   - Added `compute_dac_internal()` function
   - Updated `compute_mapping_metrics()` to compute both DAC_internal and DAC_external
   - Updated `evaluate_split()` to use both metrics
   - Updated `aggregate_metrics()` to handle both metrics separately
   - Updated statistical tests to use DAC_internal as primary
   - Updated plots to show DAC_internal
   - Updated markdown report narrative

## Verification

- ✅ Deterministic mapping is used as ground truth for DAC_internal
- ✅ DAC_internal_det = 100% by definition
- ✅ DAC_internal_learned = overlap with deterministic / total deterministic pairs
- ✅ DAC_external clearly labeled as secondary benchmark
- ✅ JSON structure separates internal and external DAC
- ✅ Markdown report emphasizes DAC_internal as primary metric
- ✅ Statistical tests use DAC_internal
- ✅ Plots show DAC_internal

## Next Steps

1. Re-run H3 evaluation to generate updated results:
   ```bash
   python run_h3_audited.py
   ```

2. Review results to verify:
   - DAC_internal_det = 100% for all splits
   - DAC_internal_learned shows meaningful agreement with deterministic
   - DAC_external is clearly separated as secondary benchmark

3. The corrected implementation now aligns with the H3 research design where deterministic mapping is the normative expert ontology.
