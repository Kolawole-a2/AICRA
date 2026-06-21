"""Tests for safe artifact deserialization."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from aicra.core.serialization import is_trusted_path, safe_joblib_load


def test_is_trusted_path_accepts_models_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    artifact = models_dir / "model.joblib"
    artifact.write_bytes(b"placeholder")
    assert is_trusted_path(artifact)


def test_is_trusted_path_rejects_outside_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "outside.joblib"
    outside.write_bytes(b"placeholder")
    assert not is_trusted_path(outside)


def test_safe_joblib_load_rejects_untrusted_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / "evil.joblib"
    joblib.dump({"ok": True}, outside)
    with pytest.raises(ValueError, match="Refusing to load untrusted"):
        safe_joblib_load(outside)


def test_safe_joblib_load_reads_trusted_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    artifact = models_dir / "model.joblib"
    joblib.dump({"score": 1}, artifact)
    loaded = safe_joblib_load(artifact)
    assert loaded == {"score": 1}
