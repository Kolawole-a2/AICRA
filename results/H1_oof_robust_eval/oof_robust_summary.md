# H1 Robust Out-of-Family (OOF) Evaluation

**Supplementary generalization test** for canonical H1. Primary H1 evidence remains the **time-ordered temporal holdout** (`results/H1_classification/H1_summary.md`). OOF asks: *Can the model rank ransomware from malware families never seen in training?*

**Script:** `scripts/evaluate_h1_oof_robust.py`  
**Artifacts:** `oof_robust_metrics.json`, this summary  
**Model scored:** `models/h1_lgbm.joblib` (trained on the time-ordered train set)

---

## 1. Dataset and split (same protocol as canonical H1)

OOF uses **EMBER-2024** with the **time-ordered** split — **not** the native EMBER `train_features.jsonl` / `test_features.jsonl` file boundary (that split is used by **H2**).

| Loader | Used by | Split method |
|--------|---------|--------------|
| `load_ember_2024(time_ordered=True)` | **H1, OOF** | Pool 50,005 rows → sort by timestamp → 80% train / 20% test |
| `load_ember_2024()` (default) | **H2** | Native EMBER train/test files as provided |

**How train and test counts are obtained:**

1. Load all EMBER-2024 rows (50,005).
2. Sort by sample timestamp (earliest → latest).
3. **Train = first 80%** → **40,004** samples (model fitting only).
4. **Test = last 20%** → **10,001** samples (all H1/OOF scoring happens here).

Temporal integrity: `max(train timestamp) < min(test timestamp)` — see `results/H1_classification/temporal_split_verification.json`.

**Important:** OOF does **not** take samples from the train set. The 5,587 OOF rows are a **subset of the 10,001 test rows**.

---

## 2. How the OOF slice (5,587) is built

OOF is a **family-generalization stress test** inside the time-ordered test window.

**Step 1 — Find held-out malware families**

- Collect malware **family names** in **train ransomware**.
- Collect malware **family names** in **test ransomware**.
- **Held-out families** = families in test ransomware **not** in train ransomware → **140 families**.

**Step 2 — Build the OOF mask (test rows only)**

| Component | Rule | Count |
|-----------|------|------:|
| **OOF positives** | Test ransomware from a **held-out family** | **178** |
| **OOF negatives** | **All benign** rows in the test set | **5,409** |
| **OOF total** | Positives + negatives | **5,587** |

Check: 178 + 5,409 = **5,587** ✓  
Check: 5,409 benign + 4,592 test ransomware = **10,001** test ✓

**What is excluded from OOF (still in test, not in OOF slice):**

- **~4,414** test ransomware samples whose families **were seen** in train malware (in-family test ransomware). These are used for canonical H1 metrics but **not** for OOF AUROC.

```text
EMBER-2024 (50,005) — time-ordered split
├── Train (40,004)          → model training only
└── Test (10,001)           → scored by trained model
    ├── OOF slice (5,587)   → OOF AUROC / PR-AUC (primary here)
    │   ├── 178  held-out-family ransomware
    │   └── 5,409 all benign
    └── Excluded (~4,414)   → in-family test ransomware (canonical H1, not OOF)
```

---

## 3. Scoring and thresholds

1. Train H1 LightGBM on **40,004** time-ordered train rows.
2. Predict probabilities on **all 10,001** test rows.
3. Tune **banking threshold 0.0248** on the **full** test set (FN cost = **100**, FP cost = **1** — same as canonical H1).
4. Compute OOF metrics on the **5,587** slice only.

**Primary OOF metric:** **AUROC** (ranking quality with both classes present).  
**Supporting only:** precision / recall / F1 at the full-test threshold — OOF prevalence is **~3.2%** positive (178/5,587) vs **~46%** on the full test set, so operational P/R/F1 are **not** comparable to canonical H1 headline numbers.

---

## 4. Results summary

| Item | Value |
|------|------:|
| Train samples | 40,004 |
| Test samples | 10,001 |
| OOF samples | 5,587 |
| OOF positives (held-out families) | 178 |
| OOF negatives (all benign test) | 5,409 |
| Held-out malware families | 140 |
| **OOF AUROC** | **0.9616** |
| OOF PR-AUC | 0.5840 |
| OOF Brier | 0.0513 |
| OOF ECE | 0.0934 |

### Operational metrics (banking threshold on OOF slice — supporting reference)

| Metric | Value |
|--------|------:|
| Threshold (tuned on full test) | 0.0248 |
| Precision | 0.0663 |
| Recall | 0.9944 |
| F1 | 0.1243 |
| Confusion matrix | TN=2916, FP=2493, FN=1, TP=177 |

---

## 5. How to interpret for the praxis

| Question | Answer |
|----------|--------|
| Is OOF a separate dataset? | No — same EMBER-2024, **time-ordered test** subset. |
| Does OOF use native EMBER files? | No — uses **time-ordered** split (H1), not H2 native files. |
| Where do 5,587 come from? | **Test only**: all benign + held-out-family ransomware. |
| What to cite as OOF result? | **AUROC 0.9616** (exceeds > 0.88 reliability benchmark). |
| Relation to H1 full_ember AUROC 0.9796? | Complementary: full test includes in-family ransomware; OOF stresses **unseen families**. |

**Examiner-safe one-liner:**  
> Supplementary OOF evaluation scores the H1 model on 5,587 time-ordered **test** samples comprising all benign test rows and ransomware from 140 malware families not seen in training malware, yielding **OOF AUROC 0.9616**.

---

## 6. Reproduce

```bash
python scripts/evaluate_h1_oof_robust.py --model-path models/h1_lgbm.joblib --output-dir results/H1_oof_robust_eval
```

See also: `docs/EVALUATION_SPLITS_H1_H2_H3.md`, `results/H1_classification/H1_summary.md`
