AI Cyber Risk Advisor (AICRA)

Overview

This repository implements a reproducible pipeline for a Machine Learning–Based Cyber Risk Advisor focused on endpoint ransomware in U.S. banking. It uses full EMBER-2024 via Thrember, producing:
- Console metrics output (AUROC, PR-AUC, Brier, ECE, Lift@10%, confusion matrix)
- Cost-sensitive threshold selection with $5M impact
- Cyber Risk Advisor Register (CSV/JSON) with MITRE ATT&CK to D3FEND mappings and prescriptive controls
- Policy JSON for auditable decision thresholds and costs

Quickstart

1. Create a Python 3.11 environment.
2. Install requirements: pip install -r requirements.txt
3. Run the complete pipeline:

   make all

Or run individual steps:
- make dirs
- make download_full
- make extract_full LIMIT=50000
- make labels
- make train
- make eval
- make register IMPACT=5000000
- make plots
- make logs

Artifacts are written to artifacts/, models/, metrics/, register/, policies/, and reports/.

Notes

- Uses Thrember for full EMBER-2024 download
- Default model: LightGBM (hist), GOSS off, focal loss. Calibration: isotonic.
- Deterministic family→ATT&CK mapping and ATT&CK→D3FEND mapping are defined in mappings/.
- Time-ordered split with out-of-family generalization evaluation.
- Banking narrative: Expected Loss = S × Impact($), default $5M impact.

