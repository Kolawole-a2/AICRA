"""Drift detection pipeline using Evidently."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns

from ..config import Settings


class DriftPipeline:
    """Drift detection pipeline using Evidently."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_column: str = "label",
        prediction_column: str = "probability",
        drift_threshold: float = 0.05,
    ) -> dict[str, Any]:
        """Run drift detection and generate report."""

        # Define column mapping
        column_mapping = ColumnMapping(
            target=target_column,
            prediction=prediction_column,
            numerical_features=[col for col in reference_data.columns
                              if col.startswith('feature_')],
        )

        # Create drift report
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
        ])

        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )

        # Create test suite for drift severity
        test_suite = TestSuite(tests=[
            TestShareOfDriftedColumns(),
        ])

        test_suite.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )

        # Save HTML report
        html_path = self.settings.artifacts_dir / 'drift_report.html'
        report.save_html(str(html_path))

        # Extract drift metrics
        drift_metrics = self._extract_drift_metrics(report, test_suite)

        # Save JSON report
        json_path = self.settings.artifacts_dir / 'drift_report.json'
        with open(json_path, 'w') as f:
            json.dump(drift_metrics, f, indent=2)

        # Check for severe drift
        severe_drift = self._check_severe_drift(drift_metrics, drift_threshold)

        return {
            "drift_metrics": drift_metrics,
            "severe_drift": severe_drift,
            "html_path": str(html_path),
            "json_path": str(json_path),
        }

    def _extract_drift_metrics(self, report: Report, test_suite: TestSuite) -> dict[str, Any]:
        """Extract key drift metrics from Evidently report."""
        metrics: dict[str, Any] = {}

        # Extract data drift metrics
        if hasattr(report, 'as_dict'):
            report_dict = report.as_dict()
            metrics.update(report_dict)

        # Extract test results
        if hasattr(test_suite, 'as_dict'):
            test_dict = test_suite.as_dict()
            metrics['test_results'] = test_dict

        return metrics

    def _check_severe_drift(self, drift_metrics: dict[str, Any], threshold: float) -> bool:
        """Check if drift is severe based on threshold."""
        # This is a simplified check - in practice, you'd analyze the specific metrics
        # from Evidently to determine drift severity
        return False  # Placeholder implementation
