"""Tests for policy pipeline."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from aicra.pipelines.policy import PolicyPipeline, Policy
from aicra.config import Settings


class TestPolicy:
    """Test Policy dataclass."""
    
    def test_policy_creation(self):
        """Test policy creation."""
        policy = Policy(
            threshold=0.5,
            cost_false_negative=100.0,
            cost_false_positive=5.0,
            impact_default=1000000.0
        )
        
        assert policy.threshold == 0.5
        assert policy.cost_false_negative == 100.0
        assert policy.cost_false_positive == 5.0
        assert policy.impact_default == 1000000.0
        assert policy.version == "1.0.0"
        assert policy.author == "AICRA"
    
    def test_policy_to_dict(self):
        """Test policy to dictionary conversion."""
        policy = Policy(
            threshold=0.5,
            cost_false_negative=100.0,
            cost_false_positive=5.0,
            impact_default=1000000.0
        )
        
        policy_dict = policy.to_dict()
        
        assert isinstance(policy_dict, dict)
        assert policy_dict["threshold"] == 0.5
        assert policy_dict["cost_false_negative"] == 100.0
        assert policy_dict["cost_false_positive"] == 5.0
        assert policy_dict["impact_default"] == 1000000.0
    
    def test_policy_to_json(self):
        """Test policy to JSON conversion."""
        policy = Policy(
            threshold=0.5,
            cost_false_negative=100.0,
            cost_false_positive=5.0,
            impact_default=1000000.0
        )
        
        json_str = policy.to_json()
        
        assert isinstance(json_str, str)
        assert "threshold" in json_str
        assert "0.5" in json_str


class TestPolicyPipeline:
    """Test policy pipeline functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = Settings()
        self.pipeline = PolicyPipeline(self.settings)
        
        # Create test data
        self.y_true = np.array([0, 1, 0, 1, 0])
        self.y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    
    def test_compute_expected_loss(self):
        """Test expected loss computation."""
        susceptibility_scores = np.array([0.1, 0.9, 0.2, 0.8])
        impact_values = np.array([1000000, 2000000, 500000, 1500000])
        
        expected_loss = self.pipeline.compute_expected_loss(
            susceptibility_scores, impact_values
        )
        
        assert len(expected_loss) == len(susceptibility_scores)
        assert np.allclose(expected_loss, susceptibility_scores * impact_values)
    
    def test_compute_expected_loss_default_impact(self):
        """Test expected loss computation with default impact."""
        susceptibility_scores = np.array([0.1, 0.9, 0.2, 0.8])
        
        expected_loss = self.pipeline.compute_expected_loss(susceptibility_scores)
        
        assert len(expected_loss) == len(susceptibility_scores)
        expected_default = susceptibility_scores * self.settings.impact_default
        assert np.allclose(expected_loss, expected_default)
    
    def test_optimize_cost_sensitive_threshold(self):
        """Test cost-sensitive threshold optimization."""
        threshold = self.pipeline.optimize_cost_sensitive_threshold(
            self.y_true, self.y_prob, cost_fn=100.0, cost_fp=5.0
        )
        
        assert isinstance(threshold, float)
        assert 0 <= threshold <= 1
    
    def test_optimize_cost_sensitive_threshold_default_costs(self):
        """Test cost-sensitive threshold optimization with default costs."""
        threshold = self.pipeline.optimize_cost_sensitive_threshold(
            self.y_true, self.y_prob
        )
        
        assert isinstance(threshold, float)
        assert 0 <= threshold <= 1
    
    @patch('aicra.pipelines.policy.mlflow.log_metrics')
    def test_create_policy(self, mock_log_metrics):
        """Test policy creation."""
        policy = self.pipeline.create_policy(
            self.y_true, self.y_prob, model_id="test_model", calibration_id="test_cal"
        )
        
        assert isinstance(policy, Policy)
        assert 0 <= policy.threshold <= 1
        assert policy.cost_false_negative == self.settings.cost_fn
        assert policy.cost_false_positive == self.settings.cost_fp
        assert policy.impact_default == self.settings.impact_default
        assert policy.model_id == "test_model"
        assert policy.calibration_id == "test_cal"
    
    @patch('aicra.pipelines.policy.mlflow.log_artifact')
    def test_save_policy(self, mock_log_artifact):
        """Test policy saving."""
        policy = Policy(
            threshold=0.5,
            cost_false_negative=100.0,
            cost_false_positive=5.0,
            impact_default=1000000.0
        )
        
        output_path = self.pipeline.save_policy(policy)
        
        assert output_path.exists()
        mock_log_artifact.assert_called()
    
    def test_generate_ops_report(self):
        """Test operations report generation."""
        df = pd.DataFrame({
            "probability": [0.1, 0.9, 0.2, 0.8],
            "label": [0, 1, 0, 1],
            "susceptibility_bucket": ["Low", "High", "Low", "High"],
            "canonical_family": ["Unknown", "Locky", "Unknown", "Ryuk"]
        })
        
        policy = Policy(
            threshold=0.5,
            cost_false_negative=100.0,
            cost_false_positive=5.0,
            impact_default=1000000.0
        )
        
        with patch('aicra.pipelines.policy.mlflow.log_metrics') as mock_log_metrics, \
             patch('aicra.pipelines.policy.mlflow.log_artifact') as mock_log_artifact:
            
            report = self.pipeline.generate_ops_report(df, policy)
        
        assert isinstance(report, dict)
        assert "summary" in report
        assert "risk_buckets" in report
        assert "family_distribution" in report
        assert "policy" in report
        
        assert report["summary"]["total_samples"] == 4
        assert report["summary"]["total_alerts"] >= 0
    
    def test_compute_lift_at_k_report(self):
        """Test Lift@k report computation."""
        df = pd.DataFrame({
            "probability": [0.1, 0.9, 0.2, 0.8],
            "label": [0, 1, 0, 1]
        })
        
        with patch('aicra.pipelines.policy.mlflow.log_metric') as mock_log_metric, \
             patch('aicra.pipelines.policy.mlflow.log_artifact') as mock_log_artifact:
            
            report = self.pipeline.compute_lift_at_k_report(df, k_values=[1, 5, 10])
        
        assert isinstance(report, dict)
        assert "lift_analysis" in report
        assert "k_values" in report
        
        assert "lift_at_1pct" in report["lift_analysis"]
        assert "lift_at_5pct" in report["lift_analysis"]
        assert "lift_at_10pct" in report["lift_analysis"]
    
    def test_get_risk_bucket_controls(self):
        """Test getting risk bucket controls."""
        high_controls = self.pipeline.get_risk_bucket_controls("High")
        medium_controls = self.pipeline.get_risk_bucket_controls("Medium")
        low_controls = self.pipeline.get_risk_bucket_controls("Low")
        
        assert isinstance(high_controls, list)
        assert isinstance(medium_controls, list)
        assert isinstance(low_controls, list)
        
        assert len(high_controls) > 0
        assert len(medium_controls) > 0
        assert len(low_controls) > 0
    
    def test_enrich_register_with_controls(self):
        """Test enriching register with prescriptive controls."""
        df = pd.DataFrame({
            "susceptibility_bucket": ["High", "Medium", "Low", "High"]
        })
        
        enriched_df = self.pipeline.enrich_register_with_controls(df)
        
        assert "prescriptive_controls" in enriched_df.columns
        assert len(enriched_df) == 4
        
        # Check that controls are lists
        for controls in enriched_df["prescriptive_controls"]:
            assert isinstance(controls, list)
    
    def test_enrich_register_with_controls_nan(self):
        """Test enriching register with NaN buckets."""
        df = pd.DataFrame({
            "susceptibility_bucket": ["High", None, "Low"]
        })
        
        enriched_df = self.pipeline.enrich_register_with_controls(df)
        
        assert "prescriptive_controls" in enriched_df.columns
        assert len(enriched_df) == 3
        
        # Check that NaN buckets get empty controls
        assert enriched_df["prescriptive_controls"].iloc[1] == []
    
    @patch('aicra.pipelines.policy.yaml.safe_load')
    @patch('builtins.open', new_callable=lambda: MagicMock())
    def test_load_risk_bucket_controls(self, mock_file, mock_yaml):
        """Test loading risk bucket controls from YAML."""
        mock_data = {
            "__version__": "1.0.0",
            "risk_buckets": {
                "High": {
                    "controls": ["Control1", "Control2"]
                }
            }
        }
        mock_yaml.return_value = mock_data
        
        controls = self.pipeline._load_risk_bucket_controls()
        
        assert "risk_buckets" in controls
        assert "High" in controls["risk_buckets"]
        assert "Control1" in controls["risk_buckets"]["High"]["controls"]
