"""Deterministic mapping layer for malware families to ATT&CK techniques."""

from __future__ import annotations

import json
import re
import sys
import time
import yaml
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import mlflow
import pandas as pd

from ..config import Settings
from ..utils.normalize import FamilyNormalizer


class MappingPipeline:
    """Version-controlled mapping pipeline for malware families with coverage tracking."""
    
    def __init__(self, settings: Settings, skip_mlflow: bool = False):
        self.settings = settings
        self.skip_mlflow = skip_mlflow
        self.normalizer = FamilyNormalizer()
        
        # Load mappings with caching
        self.canonical_families = self._load_canonical_families()
        self.family_to_attack_dict = self._load_family_to_attack()
        self.attack_to_d3fend_dict = self._load_attack_to_d3fend()
        
        # Coverage tracking
        self.coverage_stats = {
            'alias_to_family': {'mapped': 0, 'total': 0, 'unmapped': []},
            'family_to_attack': {'mapped': 0, 'total': 0, 'unmapped': []},
            'attack_to_d3fend': {'mapped': 0, 'total': 0, 'unmapped': []}
        }
        
        # Pre-compiled patterns for performance
        self._compiled_patterns = self._compile_patterns()
    
    def _load_canonical_families(self) -> Dict[str, str]:
        """Load canonical families mapping from YAML."""
        families_path = self.settings.data_dir / "lookups" / "canonical_families.yaml"
        
        with open(families_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Log version to MLflow
        if not self.skip_mlflow:
            mlflow.log_param("canonical_families_version", data.get("__version__", "unknown"))
        
        return data.get("mappings", {})
    
    def _load_family_to_attack(self) -> Dict[str, List[str]]:
        """Load family to ATT&CK techniques mapping from YAML."""
        attack_path = self.settings.data_dir / "lookups" / "family_to_attack.yaml"
        
        with open(attack_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Log version to MLflow
        if not self.skip_mlflow:
            mlflow.log_param("family_to_attack_version", data.get("__version__", "unknown"))
        
        return data.get("mappings", {})
    
    def _load_attack_to_d3fend(self) -> Dict[str, List[str]]:
        """Load ATT&CK to D3FEND countermeasures mapping from YAML."""
        d3fend_path = self.settings.data_dir / "lookups" / "attack_to_d3fend.yaml"
        
        with open(d3fend_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Log version to MLflow
        if not self.skip_mlflow:
            mlflow.log_param("attack_to_d3fend_version", data.get("__version__", "unknown"))
        
        return data.get("mappings", {})
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Pre-compile regex patterns for performance."""
        patterns = {}
        for pattern in self.canonical_families.keys():
            if '*' in pattern:
                regex_pattern = pattern.replace('*', '.*').replace('.', r'\.')
                try:
                    patterns[pattern] = re.compile(regex_pattern)
                except re.error:
                    # Skip invalid patterns
                    continue
        return patterns
    
    def normalize_family(self, raw_tag: str) -> str:
        """Normalize raw malware family/tag string to canonical family."""
        if not raw_tag or not isinstance(raw_tag, str):
            return "Unknown"
        
        # Use normalizer for consistent processing
        cleaned_tag = self.normalizer.normalize(raw_tag)
        
        # Direct mapping first
        if cleaned_tag in self.canonical_families:
            return self.canonical_families[cleaned_tag]
        
        # Pattern matching with pre-compiled patterns
        for pattern, canonical in self.canonical_families.items():
            if pattern in self._compiled_patterns:
                if self._compiled_patterns[pattern].match(cleaned_tag):
                    return canonical
            elif self._matches_pattern(cleaned_tag, pattern):
                return canonical
        
        # Default to Unknown if no match
        return "Unknown"
    
    def _matches_pattern(self, tag: str, pattern: str) -> bool:
        """Check if tag matches a pattern (supports wildcards)."""
        # Convert pattern to regex
        regex_pattern = pattern.replace('*', '.*').replace('.', r'\.')
        
        try:
            return bool(re.match(regex_pattern, tag))
        except re.error:
            # If regex fails, fall back to simple string matching
            return tag == pattern
    
    def family_to_attack(self, canonical_family: str) -> List[str]:
        """Map canonical family to list of ATT&CK techniques."""
        return self.family_to_attack_dict.get(canonical_family, [])
    
    def attack_to_d3fend(self, techniques: List[str]) -> List[str]:
        """Map ATT&CK techniques to unique set of D3FEND countermeasures."""
        countermeasures = set()
        
        for technique in techniques:
            if technique in self.attack_to_d3fend_dict:
                countermeasures.update(self.attack_to_d3fend_dict[technique])
        
        return sorted(list(countermeasures))
    
    def get_complete_mapping(self, raw_tag: str) -> Dict[str, List[str]]:
        """Get complete mapping from raw tag to techniques and countermeasures."""
        canonical_family = self.normalize_family(raw_tag)
        techniques = self.family_to_attack(canonical_family)
        countermeasures = self.attack_to_d3fend(techniques)
        
        return {
            "canonical_family": canonical_family,
            "techniques": techniques,
            "countermeasures": countermeasures
        }
    
    def get_all_canonical_families(self) -> List[str]:
        """Get list of all canonical families."""
        families_path = self.settings.data_dir / "lookups" / "canonical_families.yaml"
        
        with open(families_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return data.get("canonical_families", [])
    
    def get_family_techniques(self, raw_tag: str) -> List[str]:
        """Get ATT&CK techniques for a raw family tag."""
        canonical_family = self.normalize_family(raw_tag)
        return self.family_to_attack(canonical_family)
    
    def validate_mappings(self) -> Dict[str, List[str]]:
        """Validate that all canonical families have ATT&CK mappings."""
        canonical_families = self.get_all_canonical_families()
        missing_mappings = []
        
        for family in canonical_families:
            if family not in self.family_to_attack:
                missing_mappings.append(family)
        
        return {
            "canonical_families": canonical_families,
            "missing_mappings": missing_mappings,
            "total_families": len(canonical_families),
            "mapped_families": len(canonical_families) - len(missing_mappings)
        }
    
    def map_families_batch(self, raw_families: List[str], phase: str = "unknown") -> pd.DataFrame:
        """
        Vectorized batch mapping of families to techniques and controls.
        
        Args:
            raw_families: List of raw family names
            phase: Test phase for logging
            
        Returns:
            DataFrame with mappings
        """
        start_time = time.time()
        
        # Create DataFrame for vectorized operations
        df = pd.DataFrame({'raw_family': raw_families})
        
        # Step 1: Normalize families
        df['normalized_family'] = df['raw_family'].apply(self.normalizer.normalize)
        
        # Step 2: Map to canonical families using vectorized operations
        df['canonical_family'] = df['normalized_family'].map(self.canonical_families).fillna('Unknown')
        
        # Step 3: Map canonical families to techniques
        df['techniques'] = df['canonical_family'].map(self.family_to_attack_dict).fillna('').apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        # Step 4: Explode techniques and map to D3FEND controls
        techniques_df = df.explode('techniques')
        techniques_df['d3fend_controls'] = techniques_df['techniques'].map(self.attack_to_d3fend_dict).fillna('').apply(
            lambda x: x if isinstance(x, list) else []
        )
        
        # Step 5: Aggregate back to original rows
        result_df = techniques_df.groupby('raw_family').agg({
            'canonical_family': 'first',
            'techniques': lambda x: list(set([t for t in x if t])),
            'd3fend_controls': lambda x: list(set([c for controls in x for c in controls if c]))
        }).reset_index()
        
        # Update coverage statistics
        self._update_coverage_stats(df, result_df, phase)
        
        # Log performance metrics
        elapsed_time = time.time() - start_time
        if not self.skip_mlflow:
            mlflow.log_metric(f"mapping_batch_time_{phase}", elapsed_time)
            mlflow.log_metric(f"mapping_samples_per_second_{phase}", len(raw_families) / elapsed_time)
        
        return result_df
    
    def _update_coverage_stats(self, input_df: pd.DataFrame, result_df: pd.DataFrame, phase: str):
        """Update coverage statistics."""
        # Alias to family coverage
        total_samples = len(input_df)
        mapped_families = len(result_df[result_df['canonical_family'] != 'Unknown'])
        unmapped_aliases = input_df[~input_df['normalized_family'].isin(self.canonical_families.keys())]['raw_family'].tolist()
        
        # Filter out NaN values
        unmapped_aliases = [alias for alias in unmapped_aliases if pd.notna(alias) and alias != '']
        
        self.coverage_stats['alias_to_family']['total'] += total_samples
        self.coverage_stats['alias_to_family']['mapped'] += mapped_families
        self.coverage_stats['alias_to_family']['unmapped'].extend(unmapped_aliases)
        
        # Family to attack coverage
        mapped_families_with_techniques = len(result_df[result_df['techniques'].apply(len) > 0])
        unmapped_families = result_df[result_df['techniques'].apply(len) == 0]['canonical_family'].tolist()
        
        # Filter out NaN values
        unmapped_families = [family for family in unmapped_families if pd.notna(family) and family != '']
        
        self.coverage_stats['family_to_attack']['total'] += mapped_families
        self.coverage_stats['family_to_attack']['mapped'] += mapped_families_with_techniques
        self.coverage_stats['family_to_attack']['unmapped'].extend(unmapped_families)
        
        # Attack to D3FEND coverage
        all_techniques = [t for techniques in result_df['techniques'] for t in techniques if pd.notna(t) and t != '']
        mapped_techniques = len([t for t in all_techniques if t in self.attack_to_d3fend_dict])
        unmapped_techniques = [t for t in all_techniques if t not in self.attack_to_d3fend_dict]
        
        self.coverage_stats['attack_to_d3fend']['total'] += len(all_techniques)
        self.coverage_stats['attack_to_d3fend']['mapped'] += mapped_techniques
        self.coverage_stats['attack_to_d3fend']['unmapped'].extend(unmapped_techniques)
    
    def compute_coverage_metrics(self) -> Dict[str, float]:
        """Compute coverage metrics."""
        metrics = {}
        
        for stage, stats in self.coverage_stats.items():
            if stats['total'] > 0:
                coverage = stats['mapped'] / stats['total']
                metrics[f'{stage}_coverage'] = coverage
                metrics[f'{stage}_unmapped_rate'] = 1 - coverage
            else:
                metrics[f'{stage}_coverage'] = 0.0
                metrics[f'{stage}_unmapped_rate'] = 1.0
        
        return metrics
    
    def log_coverage_report(self, phase: str):
        """Log coverage report to artifacts."""
        coverage_metrics = self.compute_coverage_metrics()
        
        # Save coverage metrics
        coverage_file = self.settings.artifacts_dir / f"mapping_coverage_{phase}.json"
        with open(coverage_file, 'w') as f:
            json.dump({
                'phase': phase,
                'timestamp': time.time(),
                'coverage_metrics': coverage_metrics,
                'coverage_stats': self.coverage_stats
            }, f, indent=2)
        
        # Save unmapped report
        unmapped_file = self.settings.artifacts_dir / f"unmapped_report_{phase}.csv"
        unmapped_data = []
        
        for stage, stats in self.coverage_stats.items():
            unmapped_counts = {}
            for item in stats['unmapped']:
                unmapped_counts[item] = unmapped_counts.get(item, 0) + 1
            
            for item, count in unmapped_counts.items():
                unmapped_data.append({
                    'stage': stage,
                    'unmapped_item': item,
                    'count': count
                })
        
        if unmapped_data:
            unmapped_df = pd.DataFrame(unmapped_data)
            unmapped_df.to_csv(unmapped_file, index=False)
        
        # Log to MLflow
        if not self.skip_mlflow:
            for metric, value in coverage_metrics.items():
                mlflow.log_metric(metric, value)
        
        return coverage_metrics
    
    def check_coverage_thresholds(self, phase: str) -> bool:
        """
        Check if coverage meets minimum thresholds.
        
        Args:
            phase: Test phase name
            
        Returns:
            True if coverage is acceptable, False otherwise
        """
        coverage_metrics = self.compute_coverage_metrics()
        
        # Check alias to family coverage
        alias_coverage = coverage_metrics.get('alias_to_family_coverage', 0.0)
        max_unmapped_rate = self.settings.max_unmapped_rate
        
        # For EMBER datasets, don't fail on coverage but log a warning
        if self.settings.dataset_type == "ember" and not self.settings.require_family_mapping:
            if alias_coverage < (1 - max_unmapped_rate):
                print(f"⚠️  WARN: Low alias-to-family coverage ({alias_coverage:.3f}) for EMBER dataset")
                print(f"   This is expected for EMBER datasets. Proceeding with reduced coverage.")
                return True  # Don't fail for EMBER datasets
        
        # For non-EMBER datasets or when family mapping is required
        if alias_coverage < (1 - max_unmapped_rate):
            error_msg = (
                f"Coverage threshold failed for {phase} phase:\n"
                f"Alias-to-family coverage: {alias_coverage:.3f} < {(1 - max_unmapped_rate):.3f}\n"
                f"Maximum allowed unmapped rate: {max_unmapped_rate:.3f}\n"
                f"Please update canonical_families.yaml to improve coverage."
            )
            print(f"❌ {error_msg}", file=sys.stderr)
            return False
        
        return True
