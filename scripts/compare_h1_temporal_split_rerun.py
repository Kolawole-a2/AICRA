#!/usr/bin/env python3
"""Compare canonical H1 metrics against a rerun with strict temporal split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_KEYS = [
    "auroc",
    "pr_auc",
    "brier_score",
    "ece",
    "precision",
    "recall",
    "f1",
    "lift_at_1pct",
    "lift_at_5pct",
    "lift_at_10pct",
]

SPLIT_KEYS = ["full_ember", "main", "small_ember", "smoke_test"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _split_metrics(results: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for entry in results.get("metrics", {}).get("per_split_results", []):
        out[entry["split"]] = entry
    return out


def compare(baseline: dict, rerun: dict) -> dict:
    baseline_splits = _split_metrics(baseline)
    rerun_splits = _split_metrics(rerun)

    split_diffs: dict[str, dict] = {}
    for split in SPLIT_KEYS:
        if split not in baseline_splits or split not in rerun_splits:
            continue
        b = baseline_splits[split]
        r = rerun_splits[split]
        diffs = {}
        for key in METRIC_KEYS:
            if key in b and key in r:
                diffs[key] = {
                    "baseline": b[key],
                    "rerun": r[key],
                    "delta": r[key] - b[key],
                }
        split_diffs[split] = diffs

    return {
        "baseline_n_train": baseline.get("n_train_samples"),
        "rerun_n_train": rerun.get("n_train_samples"),
        "baseline_n_test": baseline.get("n_test_samples"),
        "rerun_n_test": rerun.get("n_test_samples"),
        "per_split_metric_deltas": split_diffs,
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# H1 Temporal Split Fix — Metric Comparison",
        "",
        "## Sample counts",
        "",
        f"| | Baseline (canonical) | After fix (rerun) |",
        f"|---|---------------------:|------------------:|",
        f"| Train | {report['baseline_n_train']:,} | {report['rerun_n_train']:,} |",
        f"| Test | {report['baseline_n_test']:,} | {report['rerun_n_test']:,} |",
        "",
        "## Per-split metric deltas (rerun − baseline)",
        "",
    ]

    for split, diffs in report["per_split_metric_deltas"].items():
        lines.append(f"### `{split}`")
        lines.append("")
        lines.append("| Metric | Baseline | Rerun | Delta |")
        lines.append("|--------|---------:|------:|------:|")
        for key, row in diffs.items():
            lines.append(
                f"| {key} | {row['baseline']:.8f} | {row['rerun']:.8f} | "
                f"{row['delta']:+.8f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("results/H1_classification/H1_full_results.json"),
    )
    parser.add_argument(
        "--rerun",
        type=Path,
        default=Path("results/H1_classification_strict_split/H1_full_results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/H1_classification_strict_split"),
    )
    args = parser.parse_args()

    report = compare(_load_json(args.baseline), _load_json(args.rerun))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "metric_comparison.json"
    md_path = args.output_dir / "metric_comparison.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print(json.dumps(report, indent=2))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
