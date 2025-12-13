from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def is_trusted_path(path: Path) -> bool:
    """
    Check if file path is within trusted directories.

    Security: Prevents loading arbitrary files from untrusted locations
    that could contain malicious pickle data.
    """
    abs_path = path.resolve()
    trusted_dirs = [
        Path.cwd() / "data",
        Path.cwd() / "artifacts",
        Path.cwd() / "results",
        Path.cwd() / "models",
    ]
    return any(abs_path.is_relative_to(trusted.resolve()) for trusted in trusted_dirs)


def safe_load_npz(path: Path, required_keys: list[str] | None = None) -> dict:
    """
    Safely load .npz file without allow_pickle.

    Args:
        path: Path to .npz file
        required_keys: List of required keys in .npz file

    Returns:
        Dictionary with loaded arrays

    Raises:
        ValueError: If path is not trusted or file structure is invalid
    """
    if not is_trusted_path(path):
        raise ValueError(
            f"File path must be within trusted directories: "
            f"{[str(Path.cwd() / d) for d in ['data', 'artifacts', 'results', 'models']]}"
        )

    try:
        data = np.load(path, allow_pickle=False)
        if isinstance(data, np.ndarray):
            raise ValueError(f"Expected .npz file with keys, got .npy array: {path}")

        result = {}
        for key in data.keys():
            result[key] = data[key]

        if required_keys:
            missing = set(required_keys) - set(result.keys())
            if missing:
                raise ValueError(f"Missing required keys in {path}: {missing}")

        return result
    except (KeyError, TypeError, OSError) as e:
        raise ValueError(f"Invalid .npz file structure in {path}: {e}")


def focal_loss_sample_weight(
    y: np.ndarray, gamma: float = 2.0, alpha: float = 0.75
) -> np.ndarray:
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
    parser.add_argument(
        "--calibration", choices=["platt", "isotonic"], default="isotonic"
    )
    parser.add_argument(
        "--robust-loss", choices=["balanced", "focal"], default="balanced"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Safely load features
    feat_path = Path(args.features)
    f = safe_load_npz(feat_path, required_keys=["X"])
    X = f["X"]
    fam = f.get("families")
    tss = f.get("timestamps")

    # Safely load labels
    label_path = Path(args.labels)
    labels_data = safe_load_npz(label_path, required_keys=["y"])
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
        outdir / "predictions.npz",
        val_probs=probs_cal,
        val_labels=y,
        families=fam,
        timestamps=tss,
    )

    print("Saved bagged LightGBM, calibrator, and predictions.npz")


if __name__ == "__main__":
    main()
