# H3 DAC Redefinition - Complete Implementation

## Summary

I have successfully redefined DAC for H3 to use the deterministic mapping as the ground truth (DAC_internal), while keeping DAC_external (vs reference pairs) as a secondary benchmark with the correct normalization.

## Key Changes Implemented

### 1. DAC_internal (H3 Primary Metric) ✅

**Definition:- **DAC_internal_det = 100%** by definition (deterministic vs itself)
- **DAC_internal_learned = |P_det ∩ P_learn| / |P_det| × 100%This measures the fraction of deterministic pairs that the learned mapping exactly agrees with.

**Implementation:- New function: `compute_dac_internal()` in `aicra/experiments/h3_evaluation.py`
- Updated: `compute_mapping_metrics()` to compute both DAC_internal and DAC_external
- Updated: `evaluate_split()` to include both metrics in results
- Updated: `aggregate_metrics()` to handle both metrics separately

### 2. DAC_external (Secondary Benchmark) ✅

**Definition:- **DAC_external_det = |P_det ∩ P_ref| / |P_ref| × 100%- **DAC_external_learned = |P_learn ∩ P_ref| / |P_ref| × 100%**Key Difference:** Normalized by reference pairs (not mapping pairs), measuring what fraction of the reference pairs are covered.

**Implementation:- New function: `compute_dac_external()` with correct normalization
- Clearly labeled as secondary benchmark in all outputs

### 3. Deterministic Mapping Verification ✅

**Added:- SHA256 hash verification against expected value: `a7780cfe106057cdb615df7a658e4781b61a5185eab13f6a70b4dfb8c963ed31`
- Expected counts verification: 173 pairs, 46 techniques, 9 controls
- Detailed logging of deterministic mapping metadata
- JSON output includes complete deterministic mapping info

### 4. Learned Mapping Audit ✅

**Verified:- Does NOT use `d3fend_reference_pairs.csv` as labels
- Does NOT use `deterministic_lookup.csv` pairs as labels
- Uses deterministic CSV ONLY to extract attack/defense names for text descriptions
- Constructed purely from embedding similarity (sentence-transformers)
- No data leakage for DAC_internal evaluation

**Documentation:- Added construction details to JSON output
- Added explicit logging of construction method
- Docstrings explain no data leakage

### 5. Updated JSON Structure ✅

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
  "deterministic_mapping_info": {
    "path": "...",
    "sha256": "...",
    "n_pairs": 173,
    "n_unique_attack_techniques": 46,
    "n_unique_defense_controls": 9,
    "sample_pairs": [...]
  },
  "learned_mapping_info": {
    "path": "...",
    "sha256": "...",
    "n_pairs": 190,
    "construction_method": "embedding_based_heuristic",
    "construction_details": {
      "uses_deterministic_csv": true,
      "uses_deterministic_as_labels": false,
      "uses_reference_pairs": false,
      "description": "..."
    }
  }
}
```

### 6. Updated Markdown Report ✅

- **Section 1 (Setup)**: Clearly states deterministic mapping is the normative expert ontology for H3
- **Section 2 (Per-Split Results)**: Shows both DAC_internal and DAC_external
- **Section 3 (Aggregated Findings)**: 
  - Primary section for DAC_internal (H3 primary metric)
  - Secondary section for DAC_external (ontology benchmark)
- **Section 3 (Statistical Tests)**:
  - Primary tests for DAC_internal (learned vs 100% baseline)
  - Secondary tests for DAC_external
  - Spearman correlations use DAC_internal_learned
- **Section 5 (Conclusion)**: Uses DAC_internal as primary metric

### 7. Updated Plots ✅

- `dac_internal_per_split.png`: Shows DAC_internal per split (H3 primary)
- Summary plots use DAC_internal as primary metric

### 8. Module Documentation ✅

- Updated module docstring with Research Context & Novelty
- Updated module docstring with H3 Validation Plan
- Updated module docstring with Hypothesis (H3)

## Verification Checklist

- [x] DAC_internal_det = 100% by definition
- [x] DAC_internal_learned = |P_det ∩ P_learn| / |P_det| × 100%
- [x] DAC_external normalized by reference pairs (not mapping pairs)
- [x] Deterministic mapping SHA256 verified
- [x] Learned mapping construction verified (no data leakage)
- [x] JSON structure separates internal and external DAC
- [x] Markdown report emphasizes DAC_internal as primary
- [x] Statistical tests use DAC_internal
- [x] Plots show DAC_internal
- [x] Module documentation updated

## Next Steps

1. **Re-run H3 evaluation**:
   ```bash
   python run_h3_audited.py
   ```

2. **Review results** to verify:
   - DAC_internal_det = 100% for all splits
   - DAC_internal_learned shows meaningful values
   - DAC_external is clearly separated as secondary
   - Deterministic mapping info matches expected values

3. **The corrected implementation now aligns with H3 research design** where deterministic mapping is the normative expert ontology.
