"""Safe deserialization helpers for locally trusted artifact paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

TRUSTED_ROOT_NAMES = ("data", "artifacts", "results", "models", "register")


def trusted_roots(cwd: Path | None = None) -> list[Path]:
    base = (cwd or Path.cwd()).resolve()
    return [(base / name).resolve() for name in TRUSTED_ROOT_NAMES]


def is_trusted_path(path: Path | str, *, cwd: Path | None = None) -> bool:
    """Return True when path resolves under an operator-trusted project directory."""
    abs_path = Path(path).expanduser().resolve()
    return any(abs_path.is_relative_to(root) for root in trusted_roots(cwd))


def safe_joblib_load(path: Path | str, *, cwd: Path | None = None) -> Any:
    """
    Load a joblib artifact only from trusted local directories.

    joblib uses pickle internally; restrict paths to reduce RCE risk from
    untrusted or user-supplied locations outside the project tree.
    """
    artifact_path = Path(path).expanduser()
    if not is_trusted_path(artifact_path, cwd=cwd):
        roots = trusted_roots(cwd)
        raise ValueError(
            f"Refusing to load untrusted joblib artifact: {artifact_path}\n"
            f"Path must resolve under: {[str(r) for r in roots]}"
        )
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    return joblib.load(artifact_path)
