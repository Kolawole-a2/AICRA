# H1 Robust OOF Evaluation

## Role in H1 evaluation hierarchy

H1 is judged **primarily on AUROC** under time-ordered testing, multi-split validation, and out-of-family (OOF) evaluation. PR-AUC, Brier, ECE, and operational threshold metrics are **supporting** evidence.

This run is the **OOF pillar**: it tests whether the model ranks **unseen malware families** above benign samples. It uses the same time-ordered split and model as canonical H1 (`models/h1_lgbm.joblib`).

## OOF slice definition

| Component | Rule | Count |
|-----------|------|------:|
| **Positives** | Test malware from families **not** present in training malware families | 178 |
| **Negatives** | All benign test samples | 5,409 |
| **Total** | Positives + negatives | 5,587 |
| **Held-out families** | Unique malware families in test but not in train | 140 |

Only **178 of 4,592** test malware samples fall in held-out families (~3.9%). This is the strictest family-generalization stress test.

## Results

- **Model:** `models\h1_lgbm.joblib`
- **Train samples:** 40005
- **Test samples:** 10001
- **OOF samples:** 5587
- **OOF class balance:** positives=178, negatives=5409 (~3.2% positive)
- **Held-out malware families:** 140
- **OOF AUROC:** 0.9615
- **OOF PR-AUC:** 0.5819
- **OOF Brier:** 0.0513
- **OOF ECE:** 0.0933

## Interpretation

### Primary metric: AUROC (0.9615)

Exceeds the H1 threshold (≥ 0.95). Unseen-family malware receives higher scores than benign files on average (positives ~0.77 vs negatives ~0.10), so **ranking generalizes** to new families.

### Supporting metrics on this slice

| Metric | OOF value | vs full test (`full_ember`) | Note |
|--------|----------:|----------------------------:|------|
| PR-AUC | 0.5819 | 0.9768 | Lower due to **~3% positive rate** (vs ~46% on full test) and hard unseen-family positives; still ~18× above random baseline (~0.032) |
| Brier | 0.0513 | 0.0554 | Comparable / slightly better |
| ECE | 0.0933 | 0.0081 | Higher because probabilities were shaped for full-test prevalence; calibration on this imbalanced subset is expected to degrade |

**Do not treat lower OOF PR-AUC or higher OOF ECE as contradicting H1** when AUROC remains the primary metric and exceeds 0.95 on all three validation modes.

## Canonical H1 statement (all three AUROC checks)

> H1 is supported primarily by AUROC ≥ 0.95 under time-ordered testing (0.98), multi-split validation (0.96 mean), and out-of-family evaluation (0.96). PR-AUC, Brier, ECE, and banking threshold metrics provide supporting evidence for operational deployment.

See also: `results/H1_classification/H1_summary.md`
