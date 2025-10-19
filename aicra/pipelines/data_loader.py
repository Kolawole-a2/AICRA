"""Real EMBER-2024 data loader for small_ember and full phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from ..config import Settings


class EMBERDataLoader:
    """Loader for real EMBER-2024 JSONL data."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.artifacts_dir = settings.artifacts_dir
        
    def load_ember_data(
        self,
        data_dir: str = "data/ember2024",
        sample_size: Optional[int] = None,
        seed: int = 42,
        phase: str = "full"
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        """
        Load real EMBER-2024 data from JSONL files.
        
        Args:
            data_dir: Directory containing JSONL files
            sample_size: Number of samples to load (None for all)
            seed: Random seed for deterministic sampling
            phase: Test phase name for error messages
            
        Returns:
            Tuple of (features_df, labels_series, families_series, metadata_dict)
            
        Raises:
            RuntimeError: If data directory missing or no JSONL files found
        """
        data_path = Path(data_dir)
        
        # Validate data directory exists
        if not data_path.exists():
            raise RuntimeError(
                f"EMBER-2024 data directory not found: {data_path.absolute()}\n"
                f"For {phase} phase, you must provide real EMBER-2024 data.\n"
                f"Expected structure: {data_path}/*.jsonl\n"
                f"Use: aicra run-test --phase {phase} --data-dir <path-to-ember-data>"
            )
        
        # Find JSONL files
        jsonl_files = list(data_path.glob("*.jsonl"))
        if not jsonl_files:
            raise RuntimeError(
                f"No JSONL files found in {data_path.absolute()}\n"
                f"For {phase} phase, you must provide real EMBER-2024 JSONL files.\n"
                f"Expected files: {data_path}/*.jsonl\n"
                f"Use: aicra run-test --phase {phase} --data-dir <path-to-ember-data>"
            )
        
        print(f"📁 Loading EMBER-2024 data from {len(jsonl_files)} JSONL files in {data_path}")
        
        # Load and combine JSONL files
        all_data = []
        for jsonl_file in jsonl_files:
            print(f"  📄 Loading {jsonl_file.name}")
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"  ⚠️  Skipping invalid JSON on line {line_num} in {jsonl_file.name}: {e}")
                            continue
            except Exception as e:
                print(f"  ❌ Error reading {jsonl_file.name}: {e}")
                continue
        
        if not all_data:
            raise RuntimeError(
                f"No valid data found in JSONL files in {data_path.absolute()}\n"
                f"For {phase} phase, you must provide valid EMBER-2024 JSONL data.\n"
                f"Check that files contain valid JSON lines with features and labels."
            )
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        print(f"📊 Loaded {len(df)} samples from EMBER-2024 data")
        
        # Validate schema
        self._validate_schema(df, phase)
        
        # Extract features, labels, and families
        features_df, labels_series, families_series = self._extract_features_labels_families(df)
        
        # Clean features - handle non-finite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0.0)  # Fill NaN with 0
        
        # Apply sampling if requested
        if sample_size is not None and len(features_df) > sample_size:
            print(f"🎯 Sampling {sample_size} rows deterministically (seed={seed})")
            np.random.seed(seed)
            sample_indices = np.random.choice(len(features_df), size=sample_size, replace=False)
            features_df = features_df.iloc[sample_indices].reset_index(drop=True)
            labels_series = labels_series.iloc[sample_indices].reset_index(drop=True)
            families_series = families_series.iloc[sample_indices].reset_index(drop=True)
        
        # Generate metadata
        metadata = self._generate_metadata(features_df, labels_series, phase, data_dir)
        
        # Save data summary
        self._save_data_summary(metadata, phase)
        
        return features_df, labels_series, families_series, metadata
    
    def _validate_schema(self, df: pd.DataFrame, phase: str) -> None:
        """Validate that DataFrame has required schema."""
        # Check for label column
        label_columns = ['label', 'labels', 'y', 'target']
        label_col = None
        for col in label_columns:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            raise RuntimeError(
                f"Invalid EMBER-2024 schema: No label column found.\n"
                f"For {phase} phase, JSONL files must contain a label column.\n"
                f"Expected columns: {label_columns}\n"
                f"Found columns: {list(df.columns)}"
            )
        
        # Check for feature columns
        feature_cols = [col for col in df.columns if col.startswith(('feature_', 'byte_', 'pe_'))]
        if not feature_cols:
            raise RuntimeError(
                f"Invalid EMBER-2024 schema: No feature columns found.\n"
                f"For {phase} phase, JSONL files must contain feature columns.\n"
                f"Expected columns starting with: feature_, byte_, pe_\n"
                f"Found columns: {list(df.columns)}"
            )
        
        print(f"✅ Schema validation passed: {len(feature_cols)} features, label column: {label_col}")
    
    def _extract_features_labels_families(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Extract features, labels, and families from DataFrame."""
        # Find label column
        label_columns = ['label', 'labels', 'y', 'target']
        label_col = None
        for col in label_columns:
            if col in df.columns:
                label_col = col
                break
        
        # Find family column
        family_columns = ['family', 'families', 'malware_family', 'tag']
        family_col = None
        for col in family_columns:
            if col in df.columns:
                family_col = col
                break
        
        # Extract features (all columns except label, family, and metadata)
        feature_cols = [col for col in df.columns if col.startswith(('feature_', 'byte_', 'pe_'))]
        features_df = df[feature_cols].astype(float)
        
        # Extract labels - handle potential NaN values
        labels_series = df[label_col].astype(float)  # First convert to float
        labels_series = labels_series.fillna(0)  # Fill NaN with 0
        labels_series = labels_series.astype(int)  # Then convert to int
        
        # Extract families - handle missing family column
        if family_col is not None:
            families_series = df[family_col].fillna('unknown').astype(str)
        else:
            families_series = pd.Series(['unknown'] * len(df), dtype=str)
        
        return features_df, labels_series, families_series
    
    def _generate_metadata(self, features_df: pd.DataFrame, labels_series: pd.Series, phase: str, data_dir: str) -> Dict[str, Any]:
        """Generate metadata about the loaded data."""
        total_samples = len(features_df)
        positive_samples = int(labels_series.sum())
        negative_samples = total_samples - positive_samples
        prevalence = positive_samples / total_samples if total_samples > 0 else 0
        
        metadata = {
            "phase": phase,
            "data_dir": data_dir,
            "total_samples": total_samples,
            "positive_samples": positive_samples,
            "negative_samples": negative_samples,
            "prevalence": prevalence,
            "n_features": len(features_df.columns),
            "feature_columns": list(features_df.columns),
            "data_type": "real_ember2024",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        return metadata
    
    def _save_data_summary(self, metadata: Dict[str, Any], phase: str) -> None:
        """Save data summary to artifacts directory."""
        summary_path = self.artifacts_dir / f"data_summary_{phase}.json"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📋 Data summary saved to {summary_path}")
        
        # Print summary to console
        print(f"\n📊 EMBER-2024 Data Summary ({phase.upper()}):")
        print(f"  📁 Source: {metadata['data_dir']}")
        print(f"  📈 Samples: {metadata['total_samples']:,}")
        print(f"  🎯 Features: {metadata['n_features']}")
        print(f"  ⚖️  Prevalence: {metadata['prevalence']:.3f} ({metadata['positive_samples']:,} positive, {metadata['negative_samples']:,} negative)")
        print(f"  🏷️  Type: {metadata['data_type']}")
