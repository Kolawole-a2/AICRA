from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--attack_mapping", required=True)
    parser.add_argument("--d3fend_graph", required=True)
    parser.add_argument("--impact", type=float, default=5_000_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--risk_buckets", nargs="+", default=["High", "Medium", "Low"])
    parser.add_argument("--attach_controls", action="store_true")
    args = parser.parse_args()

    ns = np.load(args.predictions, allow_pickle=True)
    probs = ns["val_probs"].astype(float)
    fam = ns.get("families")
    fam = np.array(fam).astype(str) if fam is not None else np.array(["unknown"]) * len(probs)

    y = np.load(args.labels)["y"].astype(int)

    # Policy: threshold at 80th percentile by default
    thr = float(np.quantile(probs, 0.8))

    family_map = load_json(args.mapping)
    attack_map = load_json(args.attack_mapping)
    d3f = load_json(args.d3fend_graph)

    # Normalize families according to family mapping rules
    def normalize_family(name: str) -> str:
        norm = name
        norm_cfg = family_map.get("normalize", {})
        if norm_cfg.get("lowercase", True):
            norm = norm.lower()
        if norm_cfg.get("strip", True):
            norm = norm.strip()
        for k, v in norm_cfg.get("replace", {}).items():
            norm = norm.replace(k, v)
        return norm

    controls = []
    norm_fams = []
    for f in fam:
        nf = normalize_family(str(f))
        norm_fams.append(nf)
        techniques = attack_map.get(nf, [])
        ctrls = []
        for t in techniques:
            ctrls.extend(d3f.get(t, []))
        controls.append(ctrls)

    df = pd.DataFrame(
        {
            "family": norm_fams,
            "probability": probs,
            "label": y,
        }
    )
    df["susceptibility"] = df["probability"].clip(0, 1)
    df["expected_loss"] = df["susceptibility"] * args.impact
    if args.attach_controls:
        df["controls"] = controls

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    policy = {
        "threshold": thr,
        "impact_default": args.impact,
        "risk_buckets": args.risk_buckets,
    }
    with open(args.policy_json, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)
    print("Wrote register and policy JSON")


if __name__ == "__main__":
    main()
