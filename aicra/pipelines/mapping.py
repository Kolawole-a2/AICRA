"""Deterministic mapping layer for malware families to ATT&CK techniques."""

from __future__ import annotations

import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional

import mlflow

from ..config import Settings


class MappingPipeline:
    """Version-controlled mapping pipeline for malware families."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.canonical_families = self._load_canonical_families()
        self.family_to_attack = self._load_family_to_attack()
        self.attack_to_d3fend = self._load_attack_to_d3fend()
    
    def _load_canonical_families(self) -> Dict[str, str]:
        """Load canonical families mapping from YAML."""
        families_path = self.settings.data_dir / "lookups" / "canonical_families.yaml"
        
        with open(families_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Log version to MLflow
        mlflow.log_param("canonical_families_version", data.get("__version__", "unknown"))
        
        return data.get("mappings", {})
    
    def _load_family_to_attack(self) -> Dict[str, List[str]]:
        """Load family to ATT&CK techniques mapping from YAML."""
        attack_path = self.settings.data_dir / "lookups" / "family_to_attack.yaml"
        
        with open(attack_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Log version to MLflow
        mlflow.log_param("family_to_attack_version", data.get("__version__", "unknown"))
        
        return data.get("mappings", {})
    
    def _load_attack_to_d3fend(self) -> Dict[str, List[str]]:
        """Load ATT&CK to D3FEND countermeasures mapping from YAML."""
        d3fend_path = self.settings.data_dir / "lookups" / "attack_to_d3fend.yaml"
        
        with open(d3fend_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Log version to MLflow
        mlflow.log_param("attack_to_d3fend_version", data.get("__version__", "unknown"))
        
        return data.get("mappings", {})
    
    def normalize_family(self, raw_tag: str) -> str:
        """Normalize raw malware family/tag string to canonical family."""
        if not raw_tag or not isinstance(raw_tag, str):
            return "Unknown"
        
        # Clean the input
        cleaned_tag = raw_tag.strip().lower()
        
        # Direct mapping first
        if cleaned_tag in self.canonical_families:
            return self.canonical_families[cleaned_tag]
        
        # Pattern matching
        for pattern, canonical in self.canonical_families.items():
            if self._matches_pattern(cleaned_tag, pattern):
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
        return self.family_to_attack.get(canonical_family, [])
    
    def attack_to_d3fend(self, techniques: List[str]) -> List[str]:
        """Map ATT&CK techniques to unique set of D3FEND countermeasures."""
        countermeasures = set()
        
        for technique in techniques:
            if technique in self.attack_to_d3fend:
                countermeasures.update(self.attack_to_d3fend[technique])
        
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
