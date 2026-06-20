"""Shared pytest configuration."""

import matplotlib

# Non-interactive backend for CI and headless environments (avoids Tk errors on Windows).
matplotlib.use("Agg")
