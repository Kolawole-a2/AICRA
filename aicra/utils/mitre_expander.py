"""MITRE ATT&CK/D3FEND data expander for lookup files."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

import yaml


class MITREExpander:
    """Expands lookup files from MITRE ATT&CK/D3FEND JSON data."""
    
    def __init__(self, mitre_path: Path, output_dir: Path):
        self.mitre_path = mitre_path
        self.output_dir = output_dir
        
        # Load existing mappings
        self.existing_families = self._load_existing_families()
        self.existing_techniques = self._load_existing_techniques()
        self.existing_controls = self._load_existing_controls()
        
        # Load MITRE data
        self.mitre_data = self._load_mitre_data()
    
    def _load_existing_families(self) -> Dict[str, str]:
        """Load existing canonical families mapping."""
        families_file = self.output_dir / "canonical_families.yaml"
        if families_file.exists():
            with open(families_file, 'r') as f:
                data = yaml.safe_load(f)
                return data.get("mappings", {})
        return {}
    
    def _load_existing_techniques(self) -> Dict[str, List[str]]:
        """Load existing family to attack mapping."""
        techniques_file = self.output_dir / "family_to_attack.yaml"
        if techniques_file.exists():
            with open(techniques_file, 'r') as f:
                data = yaml.safe_load(f)
                return data.get("mappings", {})
        return {}
    
    def _load_existing_controls(self) -> Dict[str, List[str]]:
        """Load existing attack to d3fend mapping."""
        controls_file = self.output_dir / "attack_to_d3fend.yaml"
        if controls_file.exists():
            with open(controls_file, 'r') as f:
                data = yaml.safe_load(f)
                return data.get("mappings", {})
        return {}
    
    def _load_mitre_data(self) -> Dict[str, Any]:
        """Load MITRE ATT&CK/D3FEND JSON data."""
        mitre_data = {}
        
        # Load ATT&CK techniques
        attack_file = self.mitre_path / "enterprise-attack.json"
        if attack_file.exists():
            with open(attack_file, 'r') as f:
                mitre_data['techniques'] = json.load(f)
        
        # Load D3FEND controls
        d3fend_file = self.mitre_path / "d3fend.json"
        if d3fend_file.exists():
            with open(d3fend_file, 'r') as f:
                mitre_data['controls'] = json.load(f)
        
        return mitre_data
    
    def analyze_changes(self) -> Dict[str, int]:
        """Analyze what changes would be made."""
        new_families = 0
        new_techniques = 0
        new_controls = 0
        
        # Analyze techniques
        if 'techniques' in self.mitre_data:
            for technique in self.mitre_data['techniques'].get('objects', []):
                if technique.get('type') == 'attack-pattern':
                    technique_id = technique.get('external_references', [{}])[0].get('external_id', '')
                    if technique_id and technique_id not in self.existing_controls:
                        new_techniques += 1
        
        # Analyze controls
        if 'controls' in self.mitre_data:
            for control in self.mitre_data['controls'].get('objects', []):
                if control.get('type') == 'd3fend:Countermeasure':
                    control_id = control.get('external_references', [{}])[0].get('external_id', '')
                    if control_id:
                        new_controls += 1
        
        return {
            'new_families': new_families,
            'new_techniques': new_techniques,
            'new_controls': new_controls
        }
    
    def expand_lookups(self) -> Dict[str, Any]:
        """Expand lookup files with MITRE data."""
        results = {
            'new_families': 0,
            'new_techniques': 0,
            'new_controls': 0,
            'versions': {}
        }
        
        # Expand canonical families (minimal changes for now)
        families_result = self._expand_canonical_families()
        results['new_families'] = families_result['new_count']
        results['versions']['canonical_families'] = families_result['version']
        
        # Expand family to attack mappings
        techniques_result = self._expand_family_to_attack()
        results['new_techniques'] = techniques_result['new_count']
        results['versions']['family_to_attack'] = techniques_result['version']
        
        # Expand attack to d3fend mappings
        controls_result = self._expand_attack_to_d3fend()
        results['new_controls'] = controls_result['new_count']
        results['versions']['attack_to_d3fend'] = controls_result['version']
        
        return results
    
    def _expand_canonical_families(self) -> Dict[str, Any]:
        """Expand canonical families mapping."""
        # For now, just bump version - actual expansion would require malware family data
        new_version = f"1.0.{int(time.time())}"
        
        families_data = {
            '__version__': new_version,
            '__source_version__': 'mitre_expander',
            '__last_updated__': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'mappings': self.existing_families
        }
        
        families_file = self.output_dir / "canonical_families.yaml"
        with open(families_file, 'w') as f:
            yaml.dump(families_data, f, default_flow_style=False)
        
        return {'new_count': 0, 'version': new_version}
    
    def _expand_family_to_attack(self) -> Dict[str, Any]:
        """Expand family to attack mappings."""
        new_version = f"1.0.{int(time.time())}"
        new_count = 0
        
        # Extract techniques from MITRE data
        techniques = {}
        if 'techniques' in self.mitre_data:
            for technique in self.mitre_data['techniques'].get('objects', []):
                if technique.get('type') == 'attack-pattern':
                    technique_id = technique.get('external_references', [{}])[0].get('external_id', '')
                    if technique_id:
                        techniques[technique_id] = technique.get('name', technique_id)
        
        # Merge with existing mappings
        updated_mappings = self.existing_techniques.copy()
        for technique_id, technique_name in techniques.items():
            if technique_id not in updated_mappings:
                # Add to a generic family for now
                if 'Generic' not in updated_mappings:
                    updated_mappings['Generic'] = []
                updated_mappings['Generic'].append(technique_id)
                new_count += 1
        
        families_data = {
            '__version__': new_version,
            '__source_version__': 'mitre_expander',
            '__last_updated__': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'mappings': updated_mappings
        }
        
        families_file = self.output_dir / "family_to_attack.yaml"
        with open(families_file, 'w') as f:
            yaml.dump(families_data, f, default_flow_style=False)
        
        return {'new_count': new_count, 'version': new_version}
    
    def _expand_attack_to_d3fend(self) -> Dict[str, Any]:
        """Expand attack to d3fend mappings."""
        new_version = f"1.0.{int(time.time())}"
        new_count = 0
        
        # Extract controls from D3FEND data
        controls = {}
        if 'controls' in self.mitre_data:
            for control in self.mitre_data['controls'].get('objects', []):
                if control.get('type') == 'd3fend:Countermeasure':
                    control_id = control.get('external_references', [{}])[0].get('external_id', '')
                    if control_id:
                        controls[control_id] = control.get('name', control_id)
        
        # Merge with existing mappings
        updated_mappings = self.existing_controls.copy()
        for control_id, control_name in controls.items():
            if control_id not in updated_mappings:
                # Add to a generic technique for now
                if 'T0000' not in updated_mappings:
                    updated_mappings['T0000'] = []
                updated_mappings['T0000'].append(control_id)
                new_count += 1
        
        families_data = {
            '__version__': new_version,
            '__source_version__': 'mitre_expander',
            '__last_updated__': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'mappings': updated_mappings
        }
        
        families_file = self.output_dir / "attack_to_d3fend.yaml"
        with open(families_file, 'w') as f:
            yaml.dump(families_data, f, default_flow_style=False)
        
        return {'new_count': new_count, 'version': new_version}
