from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import get_settings
from ..pipelines.mapping import MappingPipeline
from ..pipelines.playbook import generate_attack_playbook
from ..register import Policy, compute_register

logger = logging.getLogger(__name__)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--attack_mapping", required=True)
    parser.add_argument("--d3fend_graph", required=True)
    parser.add_argument("--impact", type=float, default=5_000_000)
    parser.add_argument(
        "--impact-table",
        type=Path,
        help="CSV with asset_id,impact columns for per-asset impact",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="ransomware_encryption",
        help="Scenario type: data_breach, ransomware_encryption, operational_disruption",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--policy-json", required=True)
    parser.add_argument("--risk_buckets", nargs="+", default=["High", "Medium", "Low"])
    parser.add_argument("--attach_controls", action="store_true")
    parser.add_argument(
        "--generate-playbook",
        action="store_true",
        help="Generate ATT&CK-informed playbook JSON",
    )
    args = parser.parse_args()

    # Safely load predictions
    pred_path = Path(args.predictions)
    ns = safe_load_npz(pred_path, required_keys=["val_probs"])
    probs = ns["val_probs"].astype(float)
    fam = ns.get("families")
    fam = (
        np.array(fam).astype(str)
        if fam is not None
        else np.array(["unknown"]) * len(probs)
    )

    # Safely load labels
    label_path = Path(args.labels)
    label_data = safe_load_npz(label_path, required_keys=["y"])
    y = label_data["y"].astype(int)

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
    df["asset_id"] = [f"asset_{i:04d}" for i in range(len(df))]
    df["susceptibility"] = df["probability"].clip(0, 1)

    # Impact parameterization: per-asset table, scenario-based, or default
    if args.impact_table and args.impact_table.exists():
        # Load impact table for per-asset impact
        impact_df = pd.read_csv(args.impact_table)
        impact_dict = dict(
            zip(impact_df["asset_id"], impact_df["impact"], strict=False)
        )
        df["expected_loss"] = df.apply(
            lambda row: row["susceptibility"]
            * impact_dict.get(row["asset_id"], args.impact),
            axis=1,
        )
    else:
        # Use scenario-based impact
        scenario_impacts = {
            "data_breach": 5_000_000,
            "ransomware_encryption": 5_000_000,  # Banking ransomware breach cost: $5M
            "operational_disruption": 2_000_000,
        }
        impact = scenario_impacts.get(args.scenario, args.impact)
        df["expected_loss"] = df["susceptibility"] * impact

    if args.attach_controls:
        df["controls"] = controls
        df["attack_techniques"] = [attack_map.get(nf, []) for nf in norm_fams]
        df["d3fend_controls"] = controls

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    # Create risk buckets
    df["susceptibility_bucket"] = pd.cut(
        df["susceptibility"],
        bins=[0.0, 0.33, 0.66, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    # Enrich with ATT&CK mappings if controls are attached
    if args.attach_controls:
        settings = get_settings()
        mapping_pipeline = MappingPipeline(settings)

        # Create a Policy object for compute_register
        policy_obj = Policy(
            threshold=thr,
            cost_false_negative=100.0,  # Banking default
            cost_false_positive=1.0,  # Banking default
            impact_default=args.impact,
        )

        # Compute register with controls
        df = compute_register(df, policy_obj)

    policy = {
        "threshold": thr,
        "impact_default": args.impact,
        "scenario": args.scenario,
        "risk_buckets": args.risk_buckets,
    }
    with open(args.policy_json, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)

    # Generate playbook if requested
    if args.generate_playbook:
        playbook_path = Path(args.out).parent / "attack_playbook.json"
        if args.attach_controls:
            settings = get_settings()
            mapping_pipeline = MappingPipeline(settings)
            generate_attack_playbook(df, mapping_pipeline, playbook_path, threshold=thr)
            print(f"Generated playbook: {playbook_path}")
        else:
            print("Warning: --attach_controls required for playbook generation")

    print("Wrote register and policy JSON")


if __name__ == "__main__":
    main()
