from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Input EMBER jsonl features with family column"
    )
    parser.add_argument(
        "--out", required=True, help="Output .npz with X and families and timestamps"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"Exists, skipping: {out}")
        return

    # Create output directory
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(args.input, lines=True)
    if args.limit is not None:
        df = df.head(args.limit)

    feature_cols = [
        c
        for c in df.columns
        if c.startswith("feature_") or c.startswith("byte_") or c.startswith("pe_")
    ]
    X = df[feature_cols].astype(np.float32).values
    families = df.get("family", pd.Series(["unknown"] * len(df))).astype(str).values
    timestamps = (
        pd.to_datetime(df.get("timestamp", pd.Timestamp("2024-01-01")).astype(str), errors="coerce")
        .fillna(pd.Timestamp("2024-01-01"))
        .astype(np.int64)
        .values
    )

    np.savez(out, X=X, families=families, timestamps=timestamps)
    print(f"Wrote {out} with shape {X.shape}")


if __name__ == "__main__":
    main()
