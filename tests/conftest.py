"""Shared pytest configuration."""

import os

import matplotlib

# Non-interactive backend for CI and headless environments (avoids Tk errors on Windows).
matplotlib.use("Agg")

# MLflow 3.x rejects file-store URIs unless explicitly allowed (tests use file:// tracking).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# CI: use TF-IDF for heuristic mapping tests instead of downloading HuggingFace models.
if os.getenv("CI") == "true" or os.getenv("AICRA_FORCE_TFIDF_MAPPING") == "1":
    import aicra.mappings.heuristic_mapping as _heuristic_mapping

    _heuristic_mapping.SENTENCE_TRANSFORMERS_AVAILABLE = False
