# Evaluation Splits Across H1, H2, and H3

Short reference for examiners: **same four split names, different underlying datasets.**

## Split names (shared convention)

| Split | Intended size (H1/H2) | Actual size (H3) |
|-------|----------------------:|------------------:|
| `full_ember` | 10,001 | **20,002** |
| `main` | 10,000 | 10,000 |
| `small_ember` | 2,000 | 2,000 |
| `smoke_test` | 200 | **2** |

## H1 — Temporal holdout (classification)

- **Loader:** `load_ember_2024(time_ordered=True)`
- **Train:** earliest 80% after pooling and sorting by timestamp (**40,004**)
- **Test:** latest 20% temporal holdout (**10,001**)
- **Integrity:** `max(train) < min(test)` — see `results/H1_classification/temporal_split_verification.json`
- **Multi-splits:** nested prefixes of the temporal test holdout  
  `smoke_test ⊂ small_ember ⊂ main ⊂ full_ember`

## H2 — Native EMBER files (calibration & thresholds)

- **Loader:** `load_ember_2024()` (no `time_ordered`)
- **Train:** `train_features.jsonl` (40,004) → 90/10 calibration fit/val
- **Test:** `test_features.jsonl` (10,001)
- **Model:** loads **H1-trained** `models/h1_lgbm.joblib` (does not retrain)
- **Multi-splits:** nested prefixes of the **native test file** (same names as H1, **different rows**)

## H3 — Pre-scored risk registers (mapping evaluation)

- **Source:** `results/<split>/risk_scores.csv` (`config/h3_splits.yaml`)
- **Not** a live re-split of H1/H2 test data at evaluation time
- **`full_ember` (20,002)** is a broader scored cohort, not H1’s 10,001-row holdout

## One-line summary

**H1** = temporal holdout · **H2** = native test file + H1 model · **H3** = pre-scored CSVs — aligned by **name** for reporting, not by row identity.

See also: [CANONICAL_VS_REBUILD_EXPLANATION.md](CANONICAL_VS_REBUILD_EXPLANATION.md), [H1_H2_VALIDATION_OUTPUTS.md](H1_H2_VALIDATION_OUTPUTS.md)
