"""Debug utilities for full EMBER-2024 phase diagnostics."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from ..config import Settings


class FullDebugPipeline:
    """Debug pipeline for full EMBER-2024 phase to diagnose AUROC ≈ 0.50 issues."""
    
    def __init__(self, settings: Settings, debug_mode: bool = True):
        self.settings = settings
        self.debug_mode = debug_mode
        self.debug_data = {}
        self.warnings = []
        self.errors = []
        
    def validate_data_loading(
        self, 
        features_df: pd.DataFrame, 
        labels_series: pd.Series, 
        families_series: pd.Series
    ) -> Dict[str, Any]:
        """Validate data loading and compute diagnostics."""
        print("🔍 DEBUG: Validating data loading...")
        
        # Basic counts
        total_rows = len(features_df)
        n_features = len(features_df.columns)
        
        # Label analysis
        y_unique = labels_series.unique()
        y_counts = labels_series.value_counts()
        prevalence = labels_series.sum() / total_rows if total_rows > 0 else 0
        
        # Feature analysis
        nan_counts = features_df.isnull().sum()
        inf_counts = np.isinf(features_df.select_dtypes(include=[np.number])).sum()
        
        # Per-feature missing rates and variance
        missing_rates = (nan_counts / total_rows).to_dict()
        feature_variance = features_df.var().to_dict()
        
        # Constant/near-constant features
        constant_features = []
        near_constant_features = []
        
        for col in features_df.columns:
            unique_vals = features_df[col].nunique()
            variance = features_df[col].var()
            
            if unique_vals <= 1:
                constant_features.append(col)
            elif variance < 1e-10:
                near_constant_features.append(col)
        
        data_summary = {
            "total_rows": total_rows,
            "n_features": n_features,
            "y_unique": y_unique.tolist(),
            "y_counts": y_counts.to_dict(),
            "prevalence": prevalence,
            "missing_rates": missing_rates,
            "feature_variance": feature_variance,
            "constant_features": constant_features,
            "near_constant_features": near_constant_features,
            "inf_counts": inf_counts.to_dict()
        }
        
        # Save data summary
        summary_path = self.settings.artifacts_dir / "debug_full_data_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(data_summary, f, indent=2)
        
        # FAIL FAST checks
        if len(y_unique) < 2:
            raise RuntimeError("CRITICAL: Only one class found in labels. Cannot train binary classifier.")
        
        # Allow smaller datasets in debug mode
        min_rows = 100 if hasattr(self, 'debug_mode') and self.debug_mode else 10_000
        if total_rows < min_rows:
            raise RuntimeError(f"CRITICAL: Insufficient data ({total_rows} rows). Need at least {min_rows:,} rows for {'debug' if min_rows == 100 else 'full'} phase.")
        
        # Check for high missing rates on critical features
        critical_feature_sets = [
            [col for col in features_df.columns if col.startswith('feature_')],
            [col for col in features_df.columns if col.startswith('byte_')],
            [col for col in features_df.columns if col.startswith('pe_')]
        ]
        
        for feature_set in critical_feature_sets:
            if feature_set:
                max_missing_rate = max(missing_rates.get(col, 0) for col in feature_set)
                if max_missing_rate > 0.2:
                    self.warnings.append(f"High missing rate ({max_missing_rate:.2%}) in {feature_set[0][:10]}... features")
        
        # Log constant feature removal
        if constant_features or near_constant_features:
            removed_count = len(constant_features) + len(near_constant_features)
            print(f"⚠️  DEBUG: Removing {removed_count} constant/near-constant features")
            
            # Save removed features list
            removed_features = constant_features + near_constant_features
            removed_df = pd.DataFrame({
                'feature': removed_features,
                'reason': ['constant'] * len(constant_features) + ['near_constant'] * len(near_constant_features)
            })
            removed_path = self.settings.artifacts_dir / "removed_features_full.csv"
            removed_df.to_csv(removed_path, index=False)
        
        self.debug_data['data_summary'] = data_summary
        return data_summary
    
    def validate_split_integrity(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        train_labels: pd.Series,
        test_labels: pd.Series,
        time_split: bool = False,
        families_series: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Validate train/test split integrity."""
        print("🔍 DEBUG: Validating split integrity...")
        
        split_summary = {
            "train_rows": len(train_data),
            "test_rows": len(test_data),
            "train_prevalence": train_labels.sum() / len(train_labels) if len(train_labels) > 0 else 0,
            "test_prevalence": test_labels.sum() / len(test_labels) if len(test_labels) > 0 else 0,
            "split_type": "time_ordered" if time_split else "stratified"
        }
        
        # Time split validation
        if time_split and 'timestamp' in train_data.columns:
            train_max_ts = train_data['timestamp'].max()
            test_min_ts = test_data['timestamp'].min()
            
            if train_max_ts >= test_min_ts:
                self.warnings.append("Time split integrity issue: train max timestamp >= test min timestamp")
            
            # Log counts by month
            train_monthly = train_data['timestamp'].dt.to_period('M').value_counts().sort_index()
            test_monthly = test_data['timestamp'].dt.to_period('M').value_counts().sort_index()
            
            split_summary.update({
                "train_max_timestamp": str(train_max_ts),
                "test_min_timestamp": str(test_min_ts),
                "train_monthly_counts": train_monthly.to_dict(),
                "test_monthly_counts": test_monthly.to_dict()
            })
            
            # Save time split details
            time_split_path = self.settings.artifacts_dir / "debug_full_split_time.json"
            with open(time_split_path, 'w') as f:
                json.dump(split_summary, f, indent=2)
        else:
            # Stratified split validation
            stratified_summary = {
                "train_class_balance": train_labels.value_counts().to_dict(),
                "test_class_balance": test_labels.value_counts().to_dict()
            }
            split_summary.update(stratified_summary)
            
            # Save stratified split details
            stratified_path = self.settings.artifacts_dir / "debug_full_split_stratified.json"
            with open(stratified_path, 'w') as f:
                json.dump(split_summary, f, indent=2)
        
        # Leakage check (if ID columns exist)
        leakage_detected = False
        leakage_details = []
        
        id_columns = ['id', 'sha256', 'hash', 'file_hash']
        for id_col in id_columns:
            if id_col in train_data.columns and id_col in test_data.columns:
                train_ids = set(train_data[id_col].dropna())
                test_ids = set(test_data[id_col].dropna())
                overlap = train_ids.intersection(test_ids)
                
                if overlap:
                    leakage_detected = True
                    leakage_details.append({
                        'column': id_col,
                        'overlap_count': len(overlap),
                        'overlap_ids': list(overlap)[:10]  # First 10 for brevity
                    })
        
        if leakage_detected:
            # HARD FAIL for leakage
            leakage_msg = "CRITICAL: Data leakage detected between train and test sets!"
            for detail in leakage_details:
                leakage_msg += f"\n  {detail['column']}: {detail['overlap_count']} overlapping IDs"
            raise RuntimeError(leakage_msg)
        
        # Save leakage check results
        leakage_path = self.settings.artifacts_dir / "leakage_check_full.csv"
        if leakage_details:
            leakage_df = pd.DataFrame(leakage_details)
            leakage_df.to_csv(leakage_path, index=False)
        else:
            # Create empty file if no leakage
            pd.DataFrame(columns=['column', 'overlap_count', 'overlap_ids']).to_csv(leakage_path, index=False)
        
        split_summary['leakage_check'] = {
            'leakage_detected': leakage_detected,
            'details': leakage_details
        }
        
        self.debug_data['split_summary'] = split_summary
        return split_summary
    
    def retune_lightgbm_large_data(
        self,
        train_data: pd.DataFrame,
        train_labels: pd.Series,
        test_data: pd.DataFrame,
        test_labels: pd.Series
    ) -> Tuple[Any, Dict[str, Any]]:
        """Retune LightGBM for large data with early stopping."""
        print("🔍 DEBUG: Retuning LightGBM for large data...")
        
        # Clean features (remove constant/near-constant)
        train_clean = self._clean_features(train_data)
        test_clean = self._clean_features(test_data)
        
        # Determine optimal parameters based on feature count
        n_features = len(train_clean.columns)
        num_leaves = 127 if n_features > 100 else 64
        
        # Large data optimized parameters
        lgb_params = {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'max_depth': -1,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_data_in_leaf': 200,
            'class_weight': 'balanced',
            'verbose': -1,
            'random_state': self.settings.random_seed
        }
        
        # Create validation split from training data
        X_train, X_val, y_train, y_val = train_test_split(
            train_clean, train_labels, 
            test_size=0.1, 
            random_state=self.settings.random_seed,
            stratify=train_labels
        )
        
        # Train with early stopping
        import lightgbm as lgb
        
        train_dataset = lgb.Dataset(X_train, label=y_train)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)
        
        model = lgb.train(
            lgb_params,
            train_dataset,
            valid_sets=[val_dataset],
            num_boost_round=3000,
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)]
        )
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'feature': train_clean.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        # Save feature importance
        importance_path = self.settings.artifacts_dir / "feature_importance_full.csv"
        feature_importance.to_csv(importance_path, index=False)
        
        # Training summary
        training_summary = {
            'best_iteration': model.best_iteration,
            'best_score': model.best_score,
            'params': lgb_params,
            'n_features_used': len(train_clean.columns),
            'n_features_original': len(train_data.columns),
            'top_20_features': feature_importance.head(20).to_dict('records')
        }
        
        # Log to MLflow
        mlflow.log_param("best_iteration", model.best_iteration)
        mlflow.log_param("best_val_auc", model.best_score['valid_0']['auc'])
        mlflow.log_param("num_leaves", num_leaves)
        mlflow.log_param("n_features_used", len(train_clean.columns))
        
        self.debug_data['training_summary'] = training_summary
        return model, training_summary
    
    def _clean_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove constant and near-constant features."""
        cleaned_data = data.copy()
        
        # Remove constant features
        constant_features = []
        near_constant_features = []
        
        for col in cleaned_data.columns:
            unique_vals = cleaned_data[col].nunique()
            variance = cleaned_data[col].var()
            
            if unique_vals <= 1:
                constant_features.append(col)
            elif variance < 1e-10:
                near_constant_features.append(col)
        
        # Remove identified features
        features_to_remove = constant_features + near_constant_features
        cleaned_data = cleaned_data.drop(columns=features_to_remove)
        
        return cleaned_data
    
    def generate_debug_report(
        self,
        test_metrics: Any,
        model: Any,
        training_summary: Dict[str, Any]
    ) -> str:
        """Generate comprehensive debug report."""
        print("🔍 DEBUG: Generating debug report...")
        
        # Extract metrics from Metrics object
        if hasattr(test_metrics, 'auroc'):
            metrics_dict = {
                'auroc': test_metrics.auroc,
                'pr_auc': test_metrics.pr_auc,
                'brier': test_metrics.brier,
                'ece': test_metrics.ece,
                'lift_at_5pct': getattr(test_metrics, 'lift_at_5pct', 0)
            }
        else:
            # Fallback for dictionary
            metrics_dict = test_metrics
        
        # Determine probable causes for low AUROC
        auroc = metrics_dict.get('auroc', 0.0)
        probable_causes = []
        
        if auroc <= 0.55:
            if len(self.debug_data['data_summary']['y_unique']) < 2:
                probable_causes.append("Single-class labels detected")
            
            if self.debug_data['data_summary']['prevalence'] < 0.01 or self.debug_data['data_summary']['prevalence'] > 0.99:
                probable_causes.append("Extreme class imbalance")
            
            if len(self.debug_data['training_summary']['top_20_features']) < 5:
                probable_causes.append("Insufficient informative features after cleaning")
            
            if self.debug_data['split_summary']['leakage_check']['leakage_detected']:
                probable_causes.append("Data leakage between train/test sets")
            
            if training_summary['best_iteration'] < 50:
                probable_causes.append("Model underfitting - try reducing min_data_in_leaf or increasing num_leaves")
            
            if training_summary['best_iteration'] > 2500:
                probable_causes.append("Model overfitting - try increasing min_data_in_leaf or reducing num_leaves")
        
        # Generate recommendations
        recommendations = []
        if auroc <= 0.55:
            recommendations.extend([
                "Try num_leaves=127 for datasets with >100 features",
                "Reduce min_data_in_leaf to 100 for smaller datasets",
                "Verify timestamp column exists for time-ordered splits",
                "Check label column contains binary values (0/1)",
                "Consider feature engineering for low-variance features"
            ])
        
        # Compile debug report
        debug_report = {
            'test_metrics': metrics_dict,
            'data_summary': self.debug_data['data_summary'],
            'split_summary': self.debug_data['split_summary'],
            'training_summary': training_summary,
            'warnings': self.warnings,
            'errors': self.errors,
            'probable_causes': probable_causes,
            'recommendations': recommendations,
            'debug_mode': True
        }
        
        # Save JSON report
        report_path = self.settings.artifacts_dir / "debug_full_report.json"
        with open(report_path, 'w') as f:
            json.dump(debug_report, f, indent=2)
        
        # Generate Markdown report
        md_content = self._generate_markdown_report(debug_report)
        md_path = self.settings.artifacts_dir / "debug_full_report.md"
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        # Print console warnings
        if auroc <= 0.55:
            print(f"\n⚠️  Full run AUROC low ({auroc:.3f}) — potential data/param issue.")
            print(f"📋 See artifacts/debug_full_report.json for details.")
            if probable_causes:
                print("🔍 Probable causes:")
                for cause in probable_causes:
                    print(f"  • {cause}")
        
        return str(report_path)
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate Markdown debug report."""
        md_lines = [
            "# Full EMBER-2024 Debug Report",
            "",
            f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Test Metrics",
            f"- **AUROC:** {report['test_metrics'].get('auroc', 'N/A'):.3f}",
            f"- **PR-AUC:** {report['test_metrics'].get('pr_auc', 'N/A'):.3f}",
            f"- **Brier Score:** {report['test_metrics'].get('brier', 'N/A'):.3f}",
            f"- **ECE:** {report['test_metrics'].get('ece', 'N/A'):.3f}",
            f"- **Lift@5%:** {report['test_metrics'].get('lift_at_5pct', 'N/A'):.3f}",
            "",
            "## Data Summary",
            f"- **Total Rows:** {report['data_summary']['total_rows']:,}",
            f"- **Features:** {report['data_summary']['n_features']}",
            f"- **Prevalence:** {report['data_summary']['prevalence']:.3f}",
            f"- **Classes:** {report['data_summary']['y_unique']}",
            "",
            "## Split Summary",
            f"- **Split Type:** {report['split_summary']['split_type']}",
            f"- **Train Rows:** {report['split_summary']['train_rows']:,}",
            f"- **Test Rows:** {report['split_summary']['test_rows']:,}",
            f"- **Leakage Detected:** {report['split_summary']['leakage_check']['leakage_detected']}",
            "",
            "## Training Summary",
            f"- **Best Iteration:** {report['training_summary']['best_iteration']}",
            f"- **Best Val AUC:** {report['training_summary']['best_score']['valid_0']['auc']:.3f}",
            f"- **Features Used:** {report['training_summary']['n_features_used']}",
            "",
            "## Top 20 Features",
        ]
        
        for i, feature in enumerate(report['training_summary']['top_20_features'][:20], 1):
            md_lines.append(f"{i}. {feature['feature']}: {feature['importance']:.2f}")
        
        if report['probable_causes']:
            md_lines.extend([
                "",
                "## Probable Causes for Low AUROC",
            ])
            for cause in report['probable_causes']:
                md_lines.append(f"- {cause}")
        
        if report['recommendations']:
            md_lines.extend([
                "",
                "## Recommendations",
            ])
            for rec in report['recommendations']:
                md_lines.append(f"- {rec}")
        
        if report['warnings']:
            md_lines.extend([
                "",
                "## Warnings",
            ])
            for warning in report['warnings']:
                md_lines.append(f"- {warning}")
        
        return "\n".join(md_lines)
