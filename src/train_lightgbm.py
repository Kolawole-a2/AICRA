from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def focal_loss_sample_weight(y: np.ndarray, gamma: float = 2.0, alpha: float = 0.75) -> np.ndarray:
    p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    w_pos = alpha * (1 - p) ** gamma
    w_neg = (1 - alpha) * p**gamma
    return np.where(y == 1, w_pos, w_neg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--mapping", required=False)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--bag-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--calibration", choices=["platt", "isotonic"], default="isotonic")
    parser.add_argument("--robust-loss", choices=["balanced", "focal"], default="balanced")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    f = np.load(args.features, allow_pickle=True)
    X = f["X"]
    fam = f.get("families")
    tss = f.get("timestamps")

    labels_data = np.load(args.labels, allow_pickle=True)
    y = labels_data["y"]

    models = []
    probs_list = []
    for seed in args.bag_seeds:
        model = LGBMClassifier(
            objective="binary",
            learning_rate=0.05,
            num_leaves=64,
            n_estimators=400,
            subsample=0.8,
            colsample_bytree=0.8,
            boosting_type="gbdt",
            random_state=seed,
            class_weight=None if args.robust_loss == "focal" else "balanced",
        )
        sample_weight = None
        if args.robust_loss == "focal":
            sample_weight = focal_loss_sample_weight(y)
        model.fit(X, y, sample_weight=sample_weight)
        models.append(model)
        probs_list.append(model.predict_proba(X)[:, 1])

    probs = np.mean(np.vstack(probs_list), axis=0)

    if args.calibration == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip").fit(probs, y)
        probs_cal = cal.transform(probs)
    else:
        cal = LogisticRegression(max_iter=1000).fit(probs.reshape(-1, 1), y)
        probs_cal = cal.predict_proba(probs.reshape(-1, 1))[:, 1]

    joblib.dump(models, outdir / "lgbm_bag.joblib")
    joblib.dump(cal, outdir / "calibrator.joblib")
    np.savez(
        outdir / "predictions.npz", val_probs=probs_cal, val_labels=y, families=fam, timestamps=tss
    )

    print("Saved bagged LightGBM, calibrator, and predictions.npz")


if __name__ == "__main__":
    main()
