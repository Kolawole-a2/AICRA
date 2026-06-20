# Documentation archive

| Path | Contents |
|------|----------|
| [development/](development/) | Historical fix summaries, implementation reports, audit notes |

For active praxis documentation, use [../praxis/README.md](../praxis/README.md).

## Canonical narrative (aligned 2026)

Archived notes may predate final praxis wording. The **current** hypothesis narrative is:

| Hypothesis | Canonical framing |
|------------|-------------------|
| **H1** | Validated on **time-ordered**, **multi-split**, and **out-of-family (OOF)** evaluation. AUROC reliability benchmark **> 0.88** (not 0.85). Empirical logistic baseline ≈ **0.778** on the same split; full_ember AUROC **0.9796**; OOF AUROC **0.9615**. |
| **H2** | **Post-hoc Platt/isotonic calibration test** (does calibration help?) plus **cost-optimal vs F1-optimal expected loss** (primary). Model already well-calibrated from H1; calibration does **not** improve expected loss. |
| **H3** | Deterministic mapping **always correct** (100% DAC_internal); learned **always extraneous** (0%). Variance reduction **0.0 on all splits** → t-test / Wilcoxon / Shapiro–Wilk on variance **not applicable**. Validated via **perfect separation** and deterministic dominance. |

Individual archive files may include a top-of-file **Archive alignment (2026)** banner when updated. Superseded claims (e.g. 47% variance reduction, AUROC baseline 0.85, “calibration improves ECE for SIEM”) are historical only.
