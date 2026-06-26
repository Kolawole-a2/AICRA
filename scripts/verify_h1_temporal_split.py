#!/usr/bin/env python3
"""Verify H1 time-ordered EMBER-2024 train/test temporal boundaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from aicra.config import get_settings
from aicra.core.data import load_ember_2024


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).isoformat()


def verify_temporal_split() -> dict:
    settings = get_settings()
    train, test = load_ember_2024(time_ordered=True)

    train_ts = pd.to_datetime(train.timestamps)
    test_ts = pd.to_datetime(test.timestamps)

    train_min = train_ts.min()
    train_max = train_ts.max()
    test_min = test_ts.min()
    test_max = test_ts.max()

    strict_gap = bool(train_max < test_min)
    non_strict_gap = bool(train_max <= test_min)
    any_train_after_test_min = bool((train_ts > test_min).any())
    any_test_before_train_max = bool((test_ts < train_max).any())

    boundary_ts = train_max
    train_at_boundary = int((train_ts == boundary_ts).sum())
    test_at_boundary = int((test_ts == boundary_ts).sum())

    duplicate_boundary_sample = False
    if train_at_boundary == 1 and test_at_boundary == 1:
        train_row = train.features.iloc[(train_ts == boundary_ts).values]
        test_row = test.features.iloc[(test_ts == boundary_ts).values]
        duplicate_boundary_sample = bool(
            np.array_equal(train_row.values, test_row.values)
        )

    # Reconstruct pool size from source JSONL (independent cross-check)
    ember_dir = settings.ember_dir
    n_pool = sum(
        sum(1 for _ in open(ember_dir / name, encoding="utf-8"))
        for name in ("train_features.jsonl", "test_features.jsonl")
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": str(ember_dir.resolve()),
        "loader": "aicra.core.data.load_ember_2024(time_ordered=True)",
        "pool_n_rows": n_pool,
        "train_n_samples": int(len(train_ts)),
        "test_n_samples": int(len(test_ts)),
        "train_timestamp_earliest": _iso(train_min),
        "train_timestamp_latest": _iso(train_max),
        "test_timestamp_earliest": _iso(test_min),
        "test_timestamp_latest": _iso(test_max),
        "max_train_lt_min_test": strict_gap,
        "max_train_lte_min_test": non_strict_gap,
        "any_train_timestamp_gt_min_test": any_train_after_test_min,
        "any_test_timestamp_lt_max_train": any_test_before_train_max,
        "boundary_timestamp": _iso(boundary_ts),
        "train_rows_at_boundary_timestamp": train_at_boundary,
        "test_rows_at_boundary_timestamp": test_at_boundary,
        "duplicate_sample_at_boundary": duplicate_boundary_sample,
        "interpretation": (
            "Strict temporal gap holds (max(train) < min(test))."
            if strict_gap
            else (
                "Strict gap fails because max(train) == min(test); "
                "no train row has a timestamp strictly after min(test). "
                + (
                    "The boundary timestamp row appears in both train and test "
                    "due to <= / >= split masks."
                    if duplicate_boundary_sample
                    else ""
                )
            )
        ),
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# H1 Time-Ordered Split Verification",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        f"Data source: `{report['data_source']}`",
        f"Loader: `{report['loader']}`",
        "",
        "## Sample counts",
        "",
        "| Set | Count |",
        "|-----|------:|",
        f"| Pool (train + test JSONL) | {report['pool_n_rows']:,} |",
        f"| Training (time-ordered) | {report['train_n_samples']:,} |",
        f"| Testing (time-ordered) | {report['test_n_samples']:,} |",
        "",
        "## Training timestamps",
        "",
        f"- **Earliest:** `{report['train_timestamp_earliest']}`",
        f"- **Latest:** `{report['train_timestamp_latest']}`",
        "",
        "## Testing timestamps",
        "",
        f"- **Earliest:** `{report['test_timestamp_earliest']}`",
        f"- **Latest:** `{report['test_timestamp_latest']}`",
        "",
        "## Temporal integrity",
        "",
        "| Check | Result |",
        "|-------|--------|",
        f"| `max(train_timestamp) < min(test_timestamp)` | **{report['max_train_lt_min_test']}** |",
        f"| `max(train_timestamp) <= min(test_timestamp)` | {report['max_train_lte_min_test']} |",
        f"| Any train row with `timestamp > min(test)` | {report['any_train_timestamp_gt_min_test']} |",
        f"| Any test row with `timestamp < max(train)` | {report['any_test_timestamp_lt_max_train']} |",
        "",
        "## Boundary detail",
        "",
        f"- Boundary timestamp: `{report['boundary_timestamp']}`",
        f"- Train rows at boundary: {report['train_rows_at_boundary_timestamp']}",
        f"- Test rows at boundary: {report['test_rows_at_boundary_timestamp']}",
        f"- Same sample duplicated in both sets at boundary: "
        f"**{report['duplicate_sample_at_boundary']}**",
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/H1_classification"),
        help="Directory for verification artifacts",
    )
    args = parser.parse_args()

    report = verify_temporal_split()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.output_dir / "temporal_split_verification.json"
    md_path = args.output_dir / "temporal_split_verification.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, md_path)

    print(json.dumps(report, indent=2))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
