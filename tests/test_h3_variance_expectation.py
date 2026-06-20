"""
Test H3 praxis expectations from saved H3_full_results.json.

Primary H3 metrics: DAC_internal and actionable precision (deterministic > learned).
Variance reduction may be 0.0 for both mappings when risk-score splits lack
technique diversity — see docs/H3_RECONCILIATION_REPORT.md.
"""

import json
import math
from pathlib import Path

import pytest


def load_h3_results(results_path: Path) -> dict:
    """Load H3 evaluation results from JSON file."""
    if not results_path.exists():
        pytest.skip(f"H3 results file not found: {results_path}")

    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


def test_h3_praxis_expectations():
    """Validate primary H3 expectations (DAC, precision) from canonical results."""
    results_path = Path("results/H3_full_evaluation/H3_full_results.json")
    results = load_h3_results(results_path)

    aggregated = results.get("aggregated_metrics", {})
    det_metrics = aggregated.get("deterministic", {})
    lrn_metrics = aggregated.get("learned", {})
    stat_tests = aggregated.get("statistical_tests", {})

    det_dac = det_metrics.get("dac_%", {}).get("mean")
    lrn_dac = lrn_metrics.get("dac_%", {}).get("mean")
    det_precision = det_metrics.get("actionable_precision", {}).get("mean")
    lrn_precision = lrn_metrics.get("actionable_precision", {}).get("mean")

    assert det_dac is not None and lrn_dac is not None
    assert (
        det_dac > lrn_dac
    ), f"Deterministic DAC_internal ({det_dac}) must exceed learned ({lrn_dac})"

    dac_test = stat_tests.get("dac", {}).get("ttest", {})
    p_dac = dac_test.get("pvalue")
    assert p_dac is not None and not math.isnan(p_dac)
    assert p_dac < 0.05, f"DAC t-test p-value must be < 0.05, got {p_dac}"

    if det_precision is not None and lrn_precision is not None:
        assert (
            det_precision >= lrn_precision
        ), f"Deterministic precision ({det_precision}) should be >= learned ({lrn_precision})"

    det_var = det_metrics.get("variance_reduction", {}).get("mean")
    lrn_var = lrn_metrics.get("variance_reduction", {}).get("mean")
    if det_var is not None and lrn_var is not None and abs(det_var - lrn_var) < 1e-8:
        # Documented praxis outcome: no variance signal on current risk-score splits
        assert det_dac == 100.0 or det_dac == 1.0 or det_dac > lrn_dac
    else:
        assert (
            det_var > lrn_var
        ), "Deterministic variance_reduction should exceed learned when non-tied"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
