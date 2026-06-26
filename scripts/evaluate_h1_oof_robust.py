#!/usr/bin/env python3
"""Run robust OOF evaluation for H1 without touching existing outputs.

This script:
1) Loads the time-ordered EMBER split used by H1
2) Loads an already-trained H1 model (default: models/h1_lgbm.joblib)
3) Builds an OOF slice that includes:
   - Positives from malware families present in test but unseen in train
   - All benign test samples as negatives
4) Computes AUROC/PR-AUC/Brier/ECE on that slice
5) Computes precision/recall/F1 at the H1 banking threshold (for reference)
6) Writes outputs only to a new folder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aicra.core.data import load_ember_2024
from aicra.core.evaluation import cost_sensitive_threshold
from aicra.core.serialization import safe_joblib_load
from aicra.experiments.h1_classification import compute_ece


def _to_prob_positive(model, features_df: pd.DataFrame) -> np.ndarray:
    """Predict positive-class probabilities with compatibility handling."""
    probs = model.predict_proba(features_df)
    if getattr(probs, "ndim", 1) == 1:
        return np.asarray(probs)
    return np.asarray(probs)[:, 1]


def run_robust_oof_eval(model_path: Path, output_dir: Path) -> dict:
    train_data, test_data = load_ember_2024(time_ordered=True)

    model = safe_joblib_load(model_path)
    y_prob_test = _to_prob_positive(model, test_data.features)
    y_true_test = np.asarray(test_data.labels.values).astype(int)

    train_families = pd.Series(train_data.families).fillna("unknown").astype(str)
    test_families = pd.Series(test_data.families).fillna("unknown").astype(str)

    train_mal_families = set(
        train_families[np.asarray(train_data.labels.values).astype(int) == 1].unique()
    )
    test_mal_families = set(
        test_families[np.asarray(test_data.labels.values).astype(int) == 1].unique()
    )
    held_out_mal_families = sorted(test_mal_families - train_mal_families)

    # Robust OOF slice:
    # - positives: held-out malware families
    # - negatives: all benign test samples
    positive_oof_mask = (y_true_test == 1) & test_families.isin(held_out_mal_families)
    negative_oof_mask = y_true_test == 0
    oof_mask = positive_oof_mask | negative_oof_mask

    y_true_oof = y_true_test[oof_mask]
    y_prob_oof = y_prob_test[oof_mask]

    pos_count = int((y_true_oof == 1).sum())
    neg_count = int((y_true_oof == 0).sum())

    auroc = float("nan")
    if pos_count > 0 and neg_count > 0:
        auroc = float(roc_auc_score(y_true_oof, y_prob_oof))

    pr_auc = float("nan")
    if pos_count > 0:
        pr_auc = float(average_precision_score(y_true_oof, y_prob_oof))

    brier = float(brier_score_loss(y_true_oof, y_prob_oof))
    ece = float(compute_ece(y_true_oof, y_prob_oof))

    # Operational metrics at the same banking threshold used by canonical H1
    banking_threshold = float(
        cost_sensitive_threshold(
            y_true_test,
            y_prob_test,
            cost_fn=100.0,
            cost_fp=1.0,
        )
    )
    y_pred_oof = (y_prob_oof >= banking_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_oof, y_pred_oof).ravel()

    results = {
        "evaluation_name": "H1 robust OOF (held-out malware families + benign negatives)",
        "model_path": str(model_path),
        "n_train_samples": int(len(train_data.features)),
        "n_test_samples": int(len(test_data.features)),
        "n_oof_samples": int(oof_mask.sum()),
        "n_oof_positive_samples": pos_count,
        "n_oof_negative_samples": neg_count,
        "n_held_out_malware_families": int(len(held_out_mal_families)),
        "held_out_malware_families": held_out_mal_families,
        "oof_auroc": auroc,
        "oof_pr_auc": pr_auc,
        "oof_brier": brier,
        "oof_ece": ece,
        "operational_threshold": banking_threshold,
        "operational_threshold_note": (
            "Cost-sensitive threshold (FN cost=100, FP cost=1) tuned on the full "
            "time-ordered test set, then applied to the OOF slice."
        ),
        "oof_precision": float(
            precision_score(y_true_oof, y_pred_oof, zero_division=0)
        ),
        "oof_recall": float(recall_score(y_true_oof, y_pred_oof, zero_division=0)),
        "oof_f1": float(f1_score(y_true_oof, y_pred_oof, zero_division=0)),
        "oof_confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "notes": [
            "This robust OOF variant avoids AUROC=NaN by ensuring both classes in OOF.",
            "Positives are from malware families unseen in training.",
            "Negatives are all benign samples from the same test window.",
            "Precision/recall/F1 use the full-test banking threshold; OOF prevalence "
            "is ~3.2% positive vs ~46% on full test, so operational metrics are "
            "supporting reference only — AUROC remains the primary OOF metric.",
            "Existing results folders are untouched.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "oof_robust_metrics.json"
    out_md = output_dir / "oof_robust_summary.md"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# H1 Robust OOF Evaluation\n\n")
        f.write(f"- **Model:** `{model_path}`\n")
        f.write(f"- **Train samples:** {results['n_train_samples']}\n")
        f.write(f"- **Test samples:** {results['n_test_samples']}\n")
        f.write(f"- **OOF samples:** {results['n_oof_samples']}\n")
        f.write(
            f"- **OOF class balance:** positives={pos_count}, negatives={neg_count}\n"
        )
        f.write(
            f"- **Held-out malware families:** {results['n_held_out_malware_families']}\n"
        )
        f.write(f"- **OOF AUROC:** {results['oof_auroc']:.4f}\n")
        f.write(f"- **OOF PR-AUC:** {results['oof_pr_auc']:.4f}\n")
        f.write(f"- **OOF Brier:** {results['oof_brier']:.4f}\n")
        f.write(f"- **OOF ECE:** {results['oof_ece']:.4f}\n\n")
        f.write("## Operational metrics (banking threshold on OOF slice)\n\n")
        f.write(f"- **Threshold:** {results['operational_threshold']:.4f}\n")
        f.write(f"- **Precision:** {results['oof_precision']:.4f}\n")
        f.write(f"- **Recall:** {results['oof_recall']:.4f}\n")
        f.write(f"- **F1:** {results['oof_f1']:.4f}\n")
        cm = results["oof_confusion_matrix"]
        f.write(
            f"- **Confusion matrix:** TN={cm['tn']}, FP={cm['fp']}, "
            f"FN={cm['fn']}, TP={cm['tp']}\n"
        )
        f.write(f"\n_{results['operational_threshold_note']}_\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robust H1 OOF evaluation.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/h1_lgbm.joblib"),
        help="Path to trained H1 model joblib file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/H1_oof_robust_eval"),
        help="Output directory for robust OOF results.",
    )
    args = parser.parse_args()

    results = run_robust_oof_eval(args.model_path, args.output_dir)
    print("Robust OOF evaluation complete.")
    print(f"Output directory: {args.output_dir}")
    print(f"OOF AUROC: {results['oof_auroc']:.4f}")
    print(f"OOF PR-AUC: {results['oof_pr_auc']:.4f}")
    print(f"OOF Precision: {results['oof_precision']:.4f}")
    print(f"OOF Recall: {results['oof_recall']:.4f}")
    print(f"OOF F1: {results['oof_f1']:.4f}")


if __name__ == "__main__":
    main()
