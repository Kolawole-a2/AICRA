# H1 Robust OOF Evaluation

- **Model:** `models\h1_lgbm.joblib`
- **Train samples:** 40004
- **Test samples:** 10001
- **OOF samples:** 5587
- **OOF class balance:** positives=178, negatives=5409
- **Held-out malware families:** 140
- **OOF AUROC:** 0.9616
- **OOF PR-AUC:** 0.5840
- **OOF Brier:** 0.0513
- **OOF ECE:** 0.0934

## Operational metrics (banking threshold on OOF slice)

- **Threshold:** 0.0248
- **Precision:** 0.0663
- **Recall:** 0.9944
- **F1:** 0.1243
- **Confusion matrix:** TN=2916, FP=2493, FN=1, TP=177

_Cost-sensitive threshold (FN cost=100, FP cost=1) tuned on the full time-ordered test set, then applied to the OOF slice._
