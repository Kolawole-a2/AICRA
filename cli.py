from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.aicra.config import PATHS, TRAINING
from src.aicra.data import load_ember_2024
from src.aicra.model import train_bagged_lightgbm
from src.aicra.calibration import create_calibrator
from src.aicra.evaluation import evaluate_probs, cost_sensitive_threshold
from src.aicra.register import Policy, compute_register, write_register


def run_all():
    PATHS.artifacts_dir.mkdir(parents=True, exist_ok=True)
    PATHS.models_dir.mkdir(parents=True, exist_ok=True)
    PATHS.metrics_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_ember_2024()

    model = train_bagged_lightgbm(train.features, train.labels)
    joblib.dump(model, PATHS.models_dir / "bagged_lightgbm.joblib")

    raw_probs_valid = model.predict_proba(train.features)
    cal = create_calibrator("isotonic" if TRAINING.use_isotonic else "platt")
    cal.fit(raw_probs_valid, train.labels.values)
    joblib.dump(cal, PATHS.models_dir / "calibrator.joblib")

    probs_test = cal.transform(model.predict_proba(test.features))

    # Default banking costs (illustrative): high cost on FN vs FP
    cost_fn = 100.0
    cost_fp = 5.0
    threshold = cost_sensitive_threshold(test.labels.values, probs_test, cost_fn, cost_fp)

    m = evaluate_probs(test.labels.values, probs_test, threshold)
    metrics_df = pd.DataFrame({
        "auroc": [m.auroc],
        "pr_auc": [m.pr_auc],
        "brier": [m.brier],
        "ece": [m.ece],
        "lift_at_10pct": [m.lift_at_k],
        "threshold": [m.threshold],
        "tn": [m.confusion[0]],
        "fp": [m.confusion[1]],
        "fn": [m.confusion[2]],
        "tp": [m.confusion[3]],
    })
    metrics_df.to_csv(PATHS.metrics_dir / "metrics.csv", index=False)

    # Out-of-family generalization: evaluate only on families not present in training
    train_fams = set(train.families.astype(str).str.lower().unique())
    mask_oof = ~test.families.astype(str).str.lower().isin(train_fams)
    if mask_oof.any():
        m_oof = evaluate_probs(test.labels.values[mask_oof.values], probs_test[mask_oof.values], threshold)
        pd.DataFrame({
            "auroc": [m_oof.auroc],
            "pr_auc": [m_oof.pr_auc],
            "brier": [m_oof.brier],
            "ece": [m_oof.ece],
            "lift_at_10pct": [m_oof.lift_at_k],
            "threshold": [m_oof.threshold],
            "tn": [m_oof.confusion[0]],
            "fp": [m_oof.confusion[1]],
            "fn": [m_oof.confusion[2]],
            "tp": [m_oof.confusion[3]],
        }).to_csv(PATHS.metrics_dir / "metrics_out_of_family.csv", index=False)

    policy = Policy(threshold=threshold, cost_false_negative=cost_fn, cost_false_positive=cost_fp, impact_default=250000.0)
    with open(PATHS.policies_dir / "policy.json", "w", encoding="utf-8") as f:
        f.write(policy.to_json())

    out_df = pd.DataFrame({
        "family": test.families,
        "probability": probs_test,
        "label": test.labels,
    })
    register_df = compute_register(out_df, policy)
    write_register(register_df, name="cyber_risk_advisor_register")

    print("AICRA build complete → Train → Evaluate → Optimize Threshold → Generate Register")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true", help="Run end-to-end pipeline")
    args = parser.parse_args()
    if args.run_all:
        run_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

