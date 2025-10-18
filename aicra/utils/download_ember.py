from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urljoin

import requests

DEFAULT_FILES = {
    "train": "train/ember_features.jsonl",
    "test": "test/ember_features.jsonl",
}


def download_file(url: str, dest: Path, force: bool = False, chunk_size: int = 1 << 20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"Exists, skipping: {dest}")
        return
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded: {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=["train", "test"], default=["train", "test"]
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional line limit to keep")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EMBER_BASE_URL", ""),
        help="Base URL hosting EMBER-2024 files",
    )
    args = parser.parse_args()

    if not args.base_url:
        raise SystemExit(
            "Set --base-url or EMBER_BASE_URL to a server hosting EMBER-2024 jsonl files. "
            "Expected relative paths: train/ember_features.jsonl, test/ember_features.jsonl"
        )

    outdir = Path(args.outdir)
    for part in args.datasets:
        rel = DEFAULT_FILES[part]
        url = urljoin(args.base_url.rstrip("/") + "/", rel)
        dest = outdir / rel
        download_file(url, dest, force=args.force)

        # Optionally trim to first N lines
        if args.limit is not None:
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with open(dest, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
                for i, line in enumerate(fin):
                    if i >= args.limit:
                        break
                    fout.write(line)
            tmp.replace(dest)
            print(f"Trimmed {dest} to first {args.limit} lines")


if __name__ == "__main__":
    main()
