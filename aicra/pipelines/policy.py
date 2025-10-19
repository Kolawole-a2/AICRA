"""Banking narratives and cost-sensitive policy pipeline."""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from ..config import Settings
from .mapping import MappingPipeline


@dataclass
class Policy:
    """Cost-sensitive policy for banking operations."""
    
    threshold: float
    cost_false_negative: float
    cost_false_positive: float
    impact_default: float
    version: str = "1.0.0"
    timestamp: str = ""
    author: str = "AICRA"
    model_id: str = ""
    calibration_id: str = ""
    lookup_versions: Dict[str, str] = None
    
    def __post_init__(self):
        if self.timestamp == "":
            self.timestamp = datetime.now().isoformat()
        if self.lookup_versions is None:
            self.lookup_versions = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class PolicyPipeline:
    """Banking narratives and cost-sensitive policy pipeline."""
    
    def __init__(self, settings: Settings, skip_mlflow: bool = False):
        self.settings = settings
        self.skip_mlflow = skip_mlflow
        self.mapping_pipeline = MappingPipeline(settings, skip_mlflow=skip_mlflow)
        self.risk_bucket_controls = self._load_risk_bucket_controls()
    
    def _load_risk_bucket_controls(self) -> Dict[str, Any]:
        """Load risk bucket controls from YAML."""
        controls_path = self.settings.data_dir / "lookups" / "risk_bucket_controls.yaml"
        
        with open(controls_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Log version to MLflow
        if not self.skip_mlflow:
            mlflow.log_param("risk_bucket_controls_version", data.get("__version__", "unknown"))
        
        return data
    
    def compute_expected_loss(
        self, 
        susceptibility_scores: np.ndarray, 
        impact_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute Expected Loss = S × Impact."""
        if impact_values is None:
            impact_values = np.full_like(susceptibility_scores, self.settings.impact_default)
        
        return susceptibility_scores * impact_values
    
    def optimize_cost_sensitive_threshold(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray,
        cost_fn: Optional[float] = None,
        cost_fp: Optional[float] = None
    ) -> float:
        """Optimize decision threshold under cost-sensitive objective."""
        
        if cost_fn is None:
            cost_fn = self.settings.cost_fn
        if cost_fp is None:
            cost_fp = self.settings.cost_fp
        
        # Get precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        
        # Compute cost for each threshold
        costs = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            # Count false negatives and false positives
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            
            # Compute total cost
            total_cost = cost_fn * fn + cost_fp * fp
            costs.append(total_cost)
        
        # Find threshold with minimum cost
        min_cost_idx = np.argmin(costs)
        optimal_threshold = thresholds[min_cost_idx]
        
        # Log optimization results
        if not self.skip_mlflow:
            mlflow.log_metrics({
                "optimal_threshold": optimal_threshold,
                "min_cost": costs[min_cost_idx],
                "cost_fn": cost_fn,
                "cost_fp": cost_fp,
                "cost_ratio": cost_fn / cost_fp,
            })
        
        return optimal_threshold
    
    def create_policy(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_id: str = "",
        calibration_id: str = "",
        impact_table_path: Optional[Path] = None
    ) -> Policy:
        """Create cost-sensitive policy."""
        
        # Optimize threshold
        optimal_threshold = self.optimize_cost_sensitive_threshold(y_true, y_prob)
        
        # Get lookup versions
        lookup_versions = {
            "canonical_families": "1.0.0",
            "family_to_attack": "1.0.0", 
            "attack_to_d3fend": "1.0.0",
            "risk_bucket_controls": "1.0.0"
        }
        
        # Create policy
        policy = Policy(
            threshold=optimal_threshold,
            cost_false_negative=self.settings.cost_fn,
            cost_false_positive=self.settings.cost_fp,
            impact_default=self.settings.impact_default,
            model_id=model_id,
            calibration_id=calibration_id,
            lookup_versions=lookup_versions
        )
        
        return policy
    
    def save_policy(self, policy: Policy, output_path: Optional[Path] = None) -> Path:
        """Save policy to JSON file."""
        if output_path is None:
            output_path = self.settings.policies_dir / "policy.json"
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save policy
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(policy.to_json())
        
        # Log to MLflow
        if not self.skip_mlflow:
            mlflow.log_artifact(str(output_path))
        
        return output_path
    
    def generate_ops_report(
        self,
        df: pd.DataFrame,
        policy: Policy,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Generate operations report summarizing alerts above threshold."""
        
        # Filter alerts above threshold
        alerts = df[df["probability"] >= policy.threshold].copy()
        
        # Compute statistics
        total_samples = len(df)
        total_alerts = len(alerts)
        alert_rate = total_alerts / total_samples if total_samples > 0 else 0
        
        # Risk bucket distribution
        bucket_counts = alerts["susceptibility_bucket"].value_counts().to_dict()
        
        # Expected loss analysis
        expected_losses = self.compute_expected_loss(
            alerts["probability"].values,
            alerts.get("impact", self.settings.impact_default)
        )
        
        total_expected_loss = np.sum(expected_losses)
        avg_expected_loss = np.mean(expected_losses) if len(expected_losses) > 0 else 0
        
        # Family analysis
        family_counts = alerts["canonical_family"].value_counts().to_dict()
        
        # Generate report
        report = {
            "summary": {
                "total_samples": int(total_samples),
                "total_alerts": int(total_alerts),
                "alert_rate": float(alert_rate),
                "threshold": float(policy.threshold),
                "total_expected_loss": float(total_expected_loss),
                "avg_expected_loss": float(avg_expected_loss),
            },
            "risk_buckets": bucket_counts,
            "family_distribution": family_counts,
            "policy": policy.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save report
        if output_path is None:
            output_path = self.settings.artifacts_dir / "ops_report.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Log to MLflow
        if not self.skip_mlflow:
            mlflow.log_metrics({
                "total_samples": total_samples,
                "total_alerts": total_alerts,
                "alert_rate": alert_rate,
                "total_expected_loss": total_expected_loss,
                "avg_expected_loss": avg_expected_loss,
            })
            
            mlflow.log_artifact(str(output_path))
        
        return report
    
    def compute_lift_at_k_report(
        self,
        df: pd.DataFrame,
        k_values: List[int] = [1, 5, 10],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Compute Lift@k report for operational efficiency."""
        
        # Sort by probability descending
        sorted_df = df.sort_values("probability", ascending=False)
        
        lift_results = {}
        for k in k_values:
            # Get top k% of samples
            k_samples = int(len(df) * k / 100)
            top_k_df = sorted_df.head(k_samples)
            
            # Compute lift
            precision_at_k = top_k_df["label"].mean() if "label" in top_k_df.columns else 0
            overall_precision = df["label"].mean() if "label" in df.columns else 0
            
            if overall_precision > 0:
                lift = precision_at_k / overall_precision
            else:
                lift = 0.0
            
            lift_results[f"lift_at_{k}pct"] = {
                "lift": float(lift),
                "precision_at_k": float(precision_at_k),
                "overall_precision": float(overall_precision),
                "samples": int(k_samples),
            }
        
        # Generate report
        report = {
            "lift_analysis": lift_results,
            "k_values": k_values,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save report
        if output_path is None:
            output_path = self.settings.artifacts_dir / "lift_report.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Log to MLflow
        if not self.skip_mlflow:
            for k, result in lift_results.items():
                mlflow.log_metric(k, result["lift"])
            
            mlflow.log_artifact(str(output_path))
        
        return report
    
    def get_risk_bucket_controls(self, bucket: str) -> List[str]:
        """Get prescriptive controls for a risk bucket."""
        bucket_data = self.risk_bucket_controls.get("risk_buckets", {}).get(bucket, {})
        return bucket_data.get("controls", [])
    
    def enrich_register_with_controls(
        self, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Enrich register with prescriptive controls based on risk buckets."""
        
        df = df.copy()
        
        # Add prescriptive controls for each risk bucket
        def get_controls_for_bucket(bucket):
            if pd.isna(bucket):
                return []
            return self.get_risk_bucket_controls(bucket)
        
        df["prescriptive_controls"] = df["susceptibility_bucket"].apply(get_controls_for_bucket)
        
        return df
