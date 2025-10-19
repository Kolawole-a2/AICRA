"""Cost-aware threshold optimization for AICRA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ..config import Settings


class CostOptimizer:
    """Cost-aware threshold optimization."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.false_negative_cost = getattr(settings, 'false_negative_cost', 1000.0)
        self.false_positive_cost = getattr(settings, 'false_positive_cost', 100.0)
    
    def compute_cost_at_threshold(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        threshold: float
    ) -> float:
        """Compute expected cost at given threshold."""
        y_pred = (y_prob >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Expected cost = FN_cost * FN_rate + FP_cost * FP_rate
        total_samples = len(y_true)
        fn_rate = fn / total_samples
        fp_rate = fp / total_samples
        
        expected_cost = self.false_negative_cost * fn_rate + self.false_positive_cost * fp_rate
        
        return expected_cost
    
    def optimize_threshold(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Find optimal threshold that minimizes expected cost."""
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 99)
        
        costs = []
        metrics_at_threshold = []
        
        for threshold in thresholds:
            cost = self.compute_cost_at_threshold(y_true, y_prob, threshold)
            costs.append(cost)
            
            y_pred = (y_prob >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            
            metrics = {
                'threshold': threshold,
                'cost': cost,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
            metrics_at_threshold.append(metrics)
        
        # Find threshold with minimum cost
        min_cost_idx = np.argmin(costs)
        optimal_threshold = thresholds[min_cost_idx]
        min_cost = costs[min_cost_idx]
        
        # Create threshold table
        threshold_table = pd.DataFrame(metrics_at_threshold)
        
        return optimal_threshold, {
            'optimal_threshold': float(optimal_threshold),
            'min_cost': float(min_cost),
            'threshold_table': threshold_table.to_dict('records'),
            'cost_curve': {
                'thresholds': thresholds.tolist(),
                'costs': costs
            }
        }
    
    def save_threshold_analysis(
        self, 
        analysis: Dict[str, float], 
        output_path: Optional[Path] = None
    ) -> Path:
        """Save threshold analysis to file."""
        if output_path is None:
            output_path = self.settings.artifacts_dir / "threshold_analysis.json"
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return output_path
    
    def save_threshold_table(
        self, 
        threshold_table: pd.DataFrame, 
        output_path: Optional[Path] = None
    ) -> Path:
        """Save threshold table to CSV."""
        if output_path is None:
            output_path = self.settings.artifacts_dir / "threshold_table.csv"
        
        threshold_table.to_csv(output_path, index=False)
        return output_path
