# Archived development documentation

These files are **historical notes** from implementation, debugging, and cleanup sessions. They are preserved for traceability but are **not** part of the praxis defense narrative.

For praxis documentation, start at: [../../praxis/README.md](../../praxis/README.md) · [../../../README.md](../../../README.md)

## Narrative alignment (2026)

Many files in this folder were written during iterative H3 fixes and cleanup. Where updated, they carry an **Archive alignment (2026)** banner and/or corrected hypothesis wording consistent with the live repo:

1. **H1** — Time-ordered + multi-split + OOF; AUROC benchmark **> 0.88**; empirical baseline ≈ 0.778  
2. **H2** — Calibration as a **help test**; primary metric = expected loss (cost-opt vs F1-opt)  
3. **H3** — Perfect separation; variance reduction 0.0 on all splits; no variance-based significance tests  

Files **without** a banner may still contain outdated intermediate numbers (e.g. early variance-reduction estimates). Treat [../../RESULTS_SUMMARY.md](../../RESULTS_SUMMARY.md), [../../../results/H3_full_evaluation/H3_full_summary.md](../../../results/H3_full_evaluation/H3_full_summary.md), and [../../../README.md](../../../README.md) as authoritative.
