"""Drift monitoring for AICRA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..config import Settings


class DriftMonitor:
    """Monitor data and prediction drift."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.drift_threshold = getattr(settings, 'drift_threshold', 0.1)
    
    def compute_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index (PSI)."""
        # Create bins based on expected distribution
        expected_min, expected_max = expected.min(), expected.max()
        bin_edges = np.linspace(expected_min, expected_max, bins + 1)
        
        # Ensure we have at least 2 bins
        if len(bin_edges) < 3:
            bin_edges = np.array([expected_min, expected_max])
        
        # Compute histograms
        expected_hist, _ = np.histogram(expected, bins=bin_edges)
        actual_hist, _ = np.histogram(actual, bins=bin_edges)
        
        # Normalize to probabilities
        expected_prob = expected_hist / expected_hist.sum()
        actual_prob = actual_hist / actual_hist.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        expected_prob = np.maximum(expected_prob, epsilon)
        actual_prob = np.maximum(actual_prob, epsilon)
        
        # Compute PSI
        psi = np.sum((actual_prob - expected_prob) * np.log(actual_prob / expected_prob))
        
        return float(psi)
    
    def compute_js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence."""
        # Normalize to probabilities
        p = p / p.sum()
        q = q / q.sum()
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-6
        p = np.maximum(p, epsilon)
        q = np.maximum(q, epsilon)
        
        # Compute JS divergence
        m = 0.5 * (p + q)
        js_div = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
        
        return float(js_div)
    
    def compute_ks_statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Kolmogorov-Smirnov statistic."""
        ks_stat, _ = stats.ks_2samp(x, y)
        return float(ks_stat)
    
    def detect_data_drift(
        self, 
        reference_data: pd.DataFrame, 
        current_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Detect data drift between reference and current datasets."""
        drift_results = {}
        
        for column in reference_data.columns:
            if reference_data[column].dtype in ['float64', 'int64']:
                ref_values = reference_data[column].dropna().values
                curr_values = current_data[column].dropna().values
                
                if len(ref_values) > 0 and len(curr_values) > 0:
                    psi = self.compute_psi(ref_values, curr_values)
                    js_div = self.compute_js_divergence(ref_values, curr_values)
                    ks_stat = self.compute_ks_statistic(ref_values, curr_values)
                    
                    drift_results[column] = {
                        'psi': psi,
                        'js_divergence': js_div,
                        'ks_statistic': ks_stat,
                        'drift_detected': psi > self.drift_threshold
                    }
        
        return drift_results
    
    def detect_prediction_drift(
        self, 
        reference_predictions: np.ndarray, 
        current_predictions: np.ndarray
    ) -> Dict[str, float]:
        """Detect prediction drift."""
        psi = self.compute_psi(reference_predictions, current_predictions)
        js_div = self.compute_js_divergence(reference_predictions, current_predictions)
        ks_stat = self.compute_ks_statistic(reference_predictions, current_predictions)
        
        return {
            'psi': psi,
            'js_divergence': js_div,
            'ks_statistic': ks_stat,
            'drift_detected': psi > self.drift_threshold
        }
    
    def generate_drift_report(
        self, 
        data_drift: Dict[str, Dict[str, float]], 
        prediction_drift: Dict[str, float],
        output_path: Optional[Path] = None
    ) -> Path:
        """Generate comprehensive drift report."""
        if output_path is None:
            output_path = self.settings.artifacts_dir / "DriftReport.md"
        
        report_lines = [
            "# Drift Monitoring Report",
            "",
            f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Drift Threshold:** {self.drift_threshold}",
            "",
            "## Data Drift Analysis",
            "",
            "| Feature | PSI | JS Divergence | KS Statistic | Drift Detected |",
            "|---------|-----|---------------|--------------|----------------|"
        ]
        
        for feature, metrics in data_drift.items():
            drift_status = "✅ YES" if metrics['drift_detected'] else "❌ NO"
            report_lines.append(
                f"| {feature} | {metrics['psi']:.4f} | {metrics['js_divergence']:.4f} | "
                f"{metrics['ks_statistic']:.4f} | {drift_status} |"
            )
        
        report_lines.extend([
            "",
            "## Prediction Drift Analysis",
            "",
            f"**PSI:** {prediction_drift['psi']:.4f}",
            f"**JS Divergence:** {prediction_drift['js_divergence']:.4f}",
            f"**KS Statistic:** {prediction_drift['ks_statistic']:.4f}",
            f"**Drift Detected:** {'✅ YES' if prediction_drift['drift_detected'] else '❌ NO'}",
            "",
            "## Recommendations",
            ""
        ])
        
        # Add recommendations based on drift detection
        drifted_features = [f for f, m in data_drift.items() if m['drift_detected']]
        
        if drifted_features:
            report_lines.append("### Data Drift Detected")
            report_lines.append("")
            for feature in drifted_features:
                report_lines.append(f"- **{feature}**: Consider retraining model or investigating data source changes")
            report_lines.append("")
        
        if prediction_drift['drift_detected']:
            report_lines.append("### Prediction Drift Detected")
            report_lines.append("")
            report_lines.append("- Model performance may be degrading")
            report_lines.append("- Consider retraining with recent data")
            report_lines.append("- Investigate changes in data distribution")
            report_lines.append("")
        
        if not drifted_features and not prediction_drift['drift_detected']:
            report_lines.append("### No Significant Drift Detected")
            report_lines.append("")
            report_lines.append("- Model appears stable")
            report_lines.append("- Continue monitoring")
            report_lines.append("")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return output_path
