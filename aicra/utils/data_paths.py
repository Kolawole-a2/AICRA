from __future__ import annotations

import os
from pathlib import Path

DEFAULT_EMBER2024_DIR = Path("data") / "ember2024_real"
ENV_EMBER2024_DIR = "AICRA_EMBER2024_DIR"


def get_ember2024_dir() -> Path:
    """
    Returns the EMBER-2024 dataset directory.

    Resolution order:
      1) Environment variable AICRA_EMBER2024_DIR
      2) Default: data/ember2024_real

    Raises a clear error if missing, with instructions.
    """
    raw = os.getenv(ENV_EMBER2024_DIR)
    path = Path(raw).expanduser() if raw else DEFAULT_EMBER2024_DIR

    if not path.exists():
        raise FileNotFoundError(
            f"EMBER-2024 directory not found: {path}\n\n"
            f"Set {ENV_EMBER2024_DIR} to your dataset location, or place files at:\n"
            f"  {DEFAULT_EMBER2024_DIR}\n\n"
            f"Then run:\n"
            f"  - Windows: scripts\\fetch_data.ps1\n"
            f"  - Bash:    scripts/fetch_data.sh\n\n"
            f"See docs/DATA.md for details."
        )

    return path
