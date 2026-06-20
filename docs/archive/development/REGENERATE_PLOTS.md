# Regenerate H3 Plots

The plots have been regenerated from the updated JSON results. The plots now reflect:

- **Single DAC metric** (not DAC_internal/DAC_external)
- **Corrected actionable precision** (deterministic > learned)
- **Updated field names** throughout

## Current Plot Files

- `dac_per_split.png` - DAC metric per split (single metric)
- `precision_per_split.png` - Actionable precision per split
- `variance_reduction_per_split.png` - Variance reduction per split
- `summary_metrics.png` - Summary bar plot with error bars

## To Regenerate Plots Again

You can regenerate the plots from the existing JSON using:

```powershell
python scripts/regenerate_h3_plots.py
```

Or directly:

```powershell
python -c "import json; from pathlib import Path; from aicra.experiments.h3_evaluation import create_plots; data = json.load(open('results/H3_full_evaluation/H3_full_results.json', 'r', encoding='utf-8')); create_plots(data['per_split_results'], data['aggregated_metrics'], Path('results/H3_full_evaluation'))"
```

## Note

The old `dac_internal_per_split.png` file has been removed. All plots now use the single `dac_%` metric.

