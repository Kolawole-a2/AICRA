from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def ece_bin(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += np.abs(acc - conf) * (mask.mean())
    return float(ece)


def lift_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.1) -> float:
    n = len(y_true)
    k_n = max(1, int(n * k))
    order = np.argsort(-y_prob)
    return float(y_true[order][:k_n].mean() / (y_true.mean() + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ops-quantile", type=float, default=0.2)
    parser.add_argument("--out", required=True)
    parser.add_argument("--time-ordered-split", action="store_true")
    parser.add_argument(
        "--metrics", nargs="+", default=["auroc", "pr_auc", "brier", "ece", "lift", "confusion"]
    )
    parser.add_argument("--fn_pref_weight", type=float, default=10.0)
    args = parser.parse_args()

    ns = np.load(args.predictions, allow_pickle=True)
    y = ns["val_labels"].astype(int)
    p = ns["val_probs"].astype(float)
    # tss = ns.get("timestamps")  # Unused variable
    fam = ns.get("families")

    # Ops threshold by quantile of probs
    thr = float(np.quantile(p, 1.0 - args.ops_quantile))
    y_pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    metrics = {
        "threshold": thr,
    }
    if "auroc" in args.metrics:
        metrics["auroc"] = float(roc_auc_score(y, p))
    if "pr_auc" in args.metrics:
        metrics["pr_auc"] = float(average_precision_score(y, p))
    if "brier" in args.metrics:
        metrics["brier"] = float(brier_score_loss(y, p))
    if "ece" in args.metrics:
        metrics["ece"] = ece_bin(y, p)
    if "lift" in args.metrics:
        metrics["lift_at_10pct"] = lift_at_k(y, p, 0.1)
    if "confusion" in args.metrics:
        metrics["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    # Out-of-family split if families available
    if fam is not None:
        fam = np.array(fam).astype(str)
        unique = np.unique(fam)
        if len(unique) > 1:
            train_fams = set(unique[: len(unique) // 2])
            mask_oof = ~np.isin(fam, list(train_fams))
            if mask_oof.any():
                y2 = y[mask_oof]
                p2 = p[mask_oof]
                metrics["oof"] = {
                    "auroc": float(roc_auc_score(y2, p2)),
                    "pr_auc": float(average_precision_score(y2, p2)),
                }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
