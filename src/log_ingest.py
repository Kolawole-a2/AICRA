from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Placeholder normalized log tensor
    X = np.zeros((1, 10), dtype=np.float32)
    np.savez(args.out, X=X)
    print(f"Wrote normalized logs to {args.out}")


if __name__ == "__main__":
    main()
