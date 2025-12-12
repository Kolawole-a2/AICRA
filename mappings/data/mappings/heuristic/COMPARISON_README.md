# Heuristic vs Deterministic Mapping Comparison

## Issue Identified

The original `learned_mapping.csv` was generated using **all 835 ATT&CK techniques**, while the deterministic mapping is **ransomware-specific** (46 techniques). This resulted in **0% overlap** between the two mappings, making them incomparable.

## Solution

A ransomware-filtered heuristic mapping has been created: `learned_mapping_ransomware.csv`

### Coverage Comparison

| Metric | Deterministic | Heuristic (Ransomware) | Coverage |
|--------|--------------|------------------------|----------|
| Techniques | 46 | 10 | 21.7% |
| Total Mappings | 173 | 16 | - |
| Unique Controls | 9 | 8 | - |

### Why Low Coverage?

1. **TF-IDF Limitations**: The heuristic mapping is using TF-IDF (sentence-transformers not available), which produces lower similarity scores
2. **Threshold**: Even with `min_similarity=0.15`, only 10 techniques have matches above threshold
3. **Description Quality**: Some ransomware techniques may not have rich descriptions in the STIX data

### Recommendations for Accurate Comparison

1. **Install sentence-transformers** for better semantic similarity:
   ```bash
   pip install sentence-transformers torch
   ```

2. **Lower threshold further** (if using TF-IDF):
   - Current: `min_similarity=0.15`
   - Try: `min_similarity=0.10` or `0.05`

3. **Increase top_k** to get more controls per technique:
   - Current: `top_k=5`
   - Try: `top_k=10`

4. **Use ransomware-filtered version** for H3 comparison:
   - File: `data/mappings/heuristic/learned_mapping_ransomware.csv`
   - This ensures you're comparing the same set of techniques

### Files

- `learned_mapping.csv` - Original (all techniques, 40 mappings, 23 techniques)
- `learned_mapping_ransomware.csv` - Ransomware-filtered (16 mappings, 10 techniques)

For H3 comparison, use **`learned_mapping_ransomware.csv`** to ensure you're comparing the same ransomware techniques.

