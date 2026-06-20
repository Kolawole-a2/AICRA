"""
H1/H2 Rebuild Pipeline (separate from canonical experiments).

This package contains a lightweight, self-contained pipeline that:
- Rebuilds per-split manifests from EMBER-2024
- Retrains a LightGBM model and calibration (H1/H2-style)
- Emits per-sample risk scores for each split
- Generates basic plots and metrics per split
- Builds ransomware-only risk registers using H3 deterministic lookups

All outputs are written under:
- results/h1h2_rebuild/<split>/
- register/h1h2_rebuild/<split>/

The canonical H1/H2/H3 experiments under aicra.experiments/* and results/H1_*/H2_*/H3_*
are not modified by this pipeline.
"""
