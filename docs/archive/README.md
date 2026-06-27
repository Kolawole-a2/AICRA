# Documentation archive

| Path | Contents |
|------|----------|
| [development/](development/) | Historical fix summaries, implementation reports, audit notes |
| [README_FULL_PRE_2026.md](README_FULL_PRE_2026.md) | Superseded verbose root README (archived June 2026) |

For active praxis documentation, use [../praxis/README.md](../praxis/README.md).

## Canonical narrative (aligned 2026)

Archived notes may predate final praxis wording. The **current** hypothesis narrative is:

| Hypothesis | Canonical framing |
|------------|-------------------|
| **H1** | Validated on **time-ordered**, **multi-split**, and **out-of-family (OOF)** evaluation. AUROC reliability benchmark **> 0.88** (not 0.85). Empirical logistic baseline ≈ **0.778** on the same split; full_ember AUROC **0.9796**; OOF AUROC **0.9615**. |
| **H2** | **Post-hoc Platt/isotonic calibration test** (does calibration help?) plus **cost-optimal vs F1-optimal expected loss** (primary). Model already well-calibrated from H1; calibration does **not** improve expected loss. |
| **H3** | **4 splits** (main 10,000; full_ember 20,002; small_ember 2,000; smoke_test 2; **32,004** total). Deterministic mapping **always correct** (100% DAC_internal); learned **always extraneous** (0%). Actionable precision **0.75** vs **0.00**. Variance reduction **0.0 on all splits** → variance tests **not applicable**. Validated via **perfect separation** and deterministic dominance. |

Individual archive files may include a top-of-file **Archive alignment (2026)** banner when updated. Superseded claims (e.g. 47% variance reduction, AUROC baseline 0.85, “calibration improves ECE for SIEM”) are historical only.
