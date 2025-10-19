"""Model card generation for AICRA."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import Settings


class ModelCardGenerator:
    """Generate model cards for AICRA models."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
    
    def compute_dataset_hash(self, dataset_path: Path) -> str:
        """Compute hash of dataset for versioning."""
        hasher = hashlib.sha256()
        
        if dataset_path.is_file():
            with open(dataset_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        else:
            # For directories, hash all files
            for file_path in sorted(dataset_path.rglob('*')):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
        
        return hasher.hexdigest()[:16]
    
    def generate_model_card(
        self,
        model_info: Dict[str, Any],
        training_info: Dict[str, Any],
        evaluation_metrics: Dict[str, float],
        calibration_info: Optional[Dict[str, Any]] = None,
        threshold_info: Optional[Dict[str, Any]] = None,
        drift_info: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """Generate comprehensive model card."""
        if output_path is None:
            output_path = self.settings.artifacts_dir / "ModelCard.md"
        
        # Compute dataset hash
        dataset_hash = self.compute_dataset_hash(Path("data/ember2024"))
        
        # Generate model card content
        card_lines = [
            "# Model Card: AICRA Malware Detection",
            "",
            f"**Version:** {model_info.get('version', '1.0.0')}",
            f"**Created:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Dataset Hash:** {dataset_hash}",
            "",
            "## Model Overview",
            "",
            f"**Purpose:** {model_info.get('purpose', 'Malware detection and family classification')}",
            f"**Algorithm:** {model_info.get('algorithm', 'LightGBM')}",
            f"**Input Features:** {model_info.get('num_features', 'N/A')}",
            f"**Output:** Binary classification (malware/benign)",
            "",
            "## Training Data",
            "",
            f"**Dataset:** EMBER 2024",
            f"**Training Samples:** {training_info.get('train_samples', 'N/A')}",
            f"**Validation Samples:** {training_info.get('val_samples', 'N/A')}",
            f"**Test Samples:** {training_info.get('test_samples', 'N/A')}",
            f"**Feature Engineering:** {training_info.get('feature_engineering', 'N/A')}",
            "",
            "## Model Performance",
            "",
            "### Classification Metrics",
            "",
            f"**AUROC:** {evaluation_metrics.get('auroc', 'N/A'):.4f}",
            f"**PR-AUC:** {evaluation_metrics.get('pr_auc', 'N/A'):.4f}",
            f"**Precision:** {evaluation_metrics.get('precision', 'N/A'):.4f}",
            f"**Recall:** {evaluation_metrics.get('recall', 'N/A'):.4f}",
            f"**F1-Score:** {evaluation_metrics.get('f1', 'N/A'):.4f}",
            "",
            "### Calibration Metrics",
            ""
        ]
        
        if calibration_info:
            card_lines.extend([
                f"**Brier Score:** {calibration_info.get('brier_score', 'N/A'):.4f}",
                f"**Expected Calibration Error:** {calibration_info.get('ece', 'N/A'):.4f}",
                f"**Calibration Method:** {calibration_info.get('method', 'N/A')}",
                ""
            ])
        else:
            card_lines.append("**Calibration:** Not available")
            card_lines.append("")
        
        if threshold_info:
            card_lines.extend([
                "### Threshold Optimization",
                "",
                f"**Optimal Threshold:** {threshold_info.get('optimal_threshold', 'N/A'):.4f}",
                f"**Expected Cost:** {threshold_info.get('min_cost', 'N/A'):.4f}",
                f"**False Negative Cost:** {threshold_info.get('fn_cost', 'N/A'):.4f}",
                f"**False Positive Cost:** {threshold_info.get('fp_cost', 'N/A'):.4f}",
                ""
            ])
        
        if drift_info:
            card_lines.extend([
                "### Drift Monitoring",
                "",
                f"**Data Drift Detected:** {'Yes' if drift_info.get('data_drift', False) else 'No'}",
                f"**Prediction Drift Detected:** {'Yes' if drift_info.get('prediction_drift', False) else 'No'}",
                ""
            ])
        
        card_lines.extend([
            "## Model Limitations",
            "",
            "- Model performance may degrade on new malware families not seen during training",
            "- Feature extraction relies on static analysis; dynamic analysis features not included",
            "- Model may be vulnerable to adversarial examples",
            "- Performance may vary across different file types and sizes",
            "",
            "## Ethical Considerations",
            "",
            "- Model is designed for legitimate security purposes only",
            "- Should not be used for malicious activities",
            "- Users should be aware of potential false positives/negatives",
            "- Model decisions should be reviewed by security professionals",
            "",
            "## Usage Guidelines",
            "",
            "### Input Requirements",
            "- Input files should be PE executables",
            "- Files should be in binary format",
            "- Minimum file size: 1KB",
            "- Maximum file size: 100MB",
            "",
            "### Output Interpretation",
            "- Probability scores range from 0.0 to 1.0",
            "- Scores > 0.5 typically indicate malware",
            "- Threshold should be adjusted based on business requirements",
            "- Always validate results with additional security tools",
            "",
            "## Model Versioning",
            "",
            f"**Current Version:** {model_info.get('version', '1.0.0')}",
            f"**Dataset Version:** {dataset_hash}",
            f"**Training Date:** {training_info.get('training_date', 'N/A')}",
            f"**Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Contact Information",
            "",
            "For questions or issues with this model, please contact the AICRA team.",
            ""
        ])
        
        # Write model card
        with open(output_path, 'w') as f:
            f.write('\n'.join(card_lines))
        
        return output_path
