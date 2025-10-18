"""MITRE ATT&CK and D3FEND JSON bundle parser for auto-populating lookup tables."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import yaml

from ..config import Settings

logger = logging.getLogger(__name__)


class MitreParser:
    """Parser for MITRE ATT&CK and D3FEND JSON bundles."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lookups_dir = settings.data_dir / "lookups"
        self.lookups_dir.mkdir(parents=True, exist_ok=True)

    def parse_and_update_lookups(
        self,
        attack_bundle_path: Optional[Path] = None,
        d3fend_bundle_path: Optional[Path] = None,
        malware_families: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Parse MITRE bundles and update lookup tables."""
        
        results = {
            "canonical_families_updated": False,
            "family_to_attack_updated": False,
            "attack_to_d3fend_updated": False,
            "source_versions": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Parse ATT&CK bundle if provided
        if attack_bundle_path and attack_bundle_path.exists():
            attack_data = self._parse_attack_bundle(attack_bundle_path)
            results["source_versions"]["attack"] = attack_data.get("version", "unknown")
            
            # Update family to attack mappings
            if malware_families:
                self._update_family_to_attack(attack_data, malware_families)
                results["family_to_attack_updated"] = True

        # Parse D3FEND bundle if provided
        if d3fend_bundle_path and d3fend_bundle_path.exists():
            d3fend_data = self._parse_d3fend_bundle(d3fend_bundle_path)
            results["source_versions"]["d3fend"] = d3fend_data.get("version", "unknown")
            
            # Update attack to D3FEND mappings
            if attack_bundle_path and attack_bundle_path.exists():
                attack_data = self._parse_attack_bundle(attack_bundle_path)
                self._update_attack_to_d3fend(attack_data, d3fend_data)
                results["attack_to_d3fend_updated"] = True

        # Update canonical families if malware families provided
        if malware_families:
            self._update_canonical_families(malware_families)
            results["canonical_families_updated"] = True

        # Log to MLflow
        self._log_to_mlflow(results)

        return results

    def _parse_attack_bundle(self, bundle_path: Path) -> Dict[str, Any]:
        """Parse MITRE ATT&CK JSON bundle."""
        logger.info(f"Parsing ATT&CK bundle from {bundle_path}")
        
        with open(bundle_path, 'r', encoding='utf-8') as f:
            bundle_data = json.load(f)
        
        # Extract techniques and their metadata
        techniques = {}
        for obj in bundle_data.get("objects", []):
            if obj.get("type") == "attack-pattern":
                technique_id = obj.get("external_references", [{}])[0].get("external_id")
                if technique_id and technique_id.startswith("T"):
                    techniques[technique_id] = {
                        "name": obj.get("name", ""),
                        "description": obj.get("description", ""),
                        "kill_chain_phases": [
                            phase.get("phase_name", "") 
                            for phase in obj.get("kill_chain_phases", [])
                        ],
                        "platforms": obj.get("x_mitre_platforms", []),
                    }
        
        return {
            "version": bundle_data.get("version", "unknown"),
            "techniques": techniques,
        }

    def _parse_d3fend_bundle(self, bundle_path: Path) -> Dict[str, Any]:
        """Parse MITRE D3FEND JSON bundle."""
        logger.info(f"Parsing D3FEND bundle from {bundle_path}")
        
        with open(bundle_path, 'r', encoding='utf-8') as f:
            bundle_data = json.load(f)
        
        # Extract countermeasures and their metadata
        countermeasures = {}
        for obj in bundle_data.get("objects", []):
            if obj.get("type") == "course-of-action":
                d3fend_id = obj.get("external_references", [{}])[0].get("external_id")
                if d3fend_id and d3fend_id.startswith("D3-"):
                    countermeasures[d3fend_id] = {
                        "name": obj.get("name", ""),
                        "description": obj.get("description", ""),
                        "attack_techniques": self._extract_attack_techniques(obj),
                    }
        
        return {
            "version": bundle_data.get("version", "unknown"),
            "countermeasures": countermeasures,
        }

    def _extract_attack_techniques(self, d3fend_obj: Dict[str, Any]) -> List[str]:
        """Extract ATT&CK techniques that D3FEND countermeasure addresses."""
        techniques = []
        
        # Look for relationships or references to ATT&CK techniques
        for ref in d3fend_obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                technique_id = ref.get("external_id")
                if technique_id and technique_id.startswith("T"):
                    techniques.append(technique_id)
        
        return techniques

    def _update_canonical_families(self, malware_families: List[str]) -> None:
        """Update canonical families lookup, preserving manual entries."""
        canonical_path = self.lookups_dir / "canonical_families.yaml"
        
        # Load existing mappings
        existing_data = {}
        if canonical_path.exists():
            with open(canonical_path, 'r', encoding='utf-8') as f:
                existing_data = yaml.safe_load(f) or {}
        
        # Create new mappings for families not already present
        mappings = existing_data.get("mappings", {})
        new_families_added = 0
        
        for family in malware_families:
            family_lower = family.lower()
            if family_lower not in mappings:
                mappings[family_lower] = family
                new_families_added += 1
        
        # Update with new data
        updated_data = {
            "__version__": existing_data.get("__version__", "1.0.0"),
            "__source_version__": f"mitre_parser_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "__last_updated__": datetime.now().isoformat(),
            "mappings": mappings,
        }
        
        with open(canonical_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(updated_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated canonical families: {new_families_added} new families added")

    def _update_family_to_attack(
        self, attack_data: Dict[str, Any], malware_families: List[str]
    ) -> None:
        """Update family to attack techniques mapping."""
        family_to_attack_path = self.lookups_dir / "family_to_attack.yaml"
        
        # Load existing mappings
        existing_data = {}
        if family_to_attack_path.exists():
            with open(family_to_attack_path, 'r', encoding='utf-8') as f:
                existing_data = yaml.safe_load(f) or {}
        
        # Create mappings based on common ransomware techniques
        mappings = existing_data.get("mappings", {})
        
        # Common ransomware techniques
        common_techniques = [
            "T1486",  # Data Encrypted for Impact
            "T1490",  # Inhibit System Recovery
            "T1059",  # Command and Scripting Interpreter
            "T1021",  # Remote Services
            "T1562",  # Impair Defenses
            "T1071",  # Application Layer Protocol
            "T1041",  # Exfiltration Over C2 Channel
        ]
        
        new_mappings_added = 0
        for family in malware_families:
            if family not in mappings:
                # Assign common techniques to new families
                mappings[family] = common_techniques[:3]  # First 3 techniques
                new_mappings_added += 1
        
        # Update with new data
        updated_data = {
            "__version__": existing_data.get("__version__", "1.0.0"),
            "__source_version__": f"mitre_parser_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "__last_updated__": datetime.now().isoformat(),
            "mappings": mappings,
        }
        
        with open(family_to_attack_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(updated_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated family to attack: {new_mappings_added} new mappings added")

    def _update_attack_to_d3fend(
        self, attack_data: Dict[str, Any], d3fend_data: Dict[str, Any]
    ) -> None:
        """Update attack techniques to D3FEND countermeasures mapping."""
        attack_to_d3fend_path = self.lookups_dir / "attack_to_d3fend.yaml"
        
        # Load existing mappings
        existing_data = {}
        if attack_to_d3fend_path.exists():
            with open(attack_to_d3fend_path, 'r', encoding='utf-8') as f:
                existing_data = yaml.safe_load(f) or {}
        
        # Create mappings based on D3FEND countermeasures
        mappings = existing_data.get("mappings", {})
        
        # Map techniques to countermeasures based on D3FEND data
        new_mappings_added = 0
        for technique_id, technique_info in attack_data.get("techniques", {}).items():
            if technique_id not in mappings:
                # Find relevant D3FEND countermeasures
                countermeasures = []
                for d3fend_id, d3fend_info in d3fend_data.get("countermeasures", {}).items():
                    if technique_id in d3fend_info.get("attack_techniques", []):
                        countermeasures.append(d3fend_id)
                
                # If no specific mapping found, use general countermeasures
                if not countermeasures:
                    countermeasures = self._get_general_countermeasures(technique_id)
                
                mappings[technique_id] = countermeasures[:3]  # Limit to 3 countermeasures
                new_mappings_added += 1
        
        # Update with new data
        updated_data = {
            "__version__": existing_data.get("__version__", "1.0.0"),
            "__source_version__": f"mitre_parser_{datetime.now().strftime('%Y%m%d_%H%M')}",
            "__last_updated__": datetime.now().isoformat(),
            "mappings": mappings,
        }
        
        with open(attack_to_d3fend_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(updated_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Updated attack to D3FEND: {new_mappings_added} new mappings added")

    def _get_general_countermeasures(self, technique_id: str) -> List[str]:
        """Get general countermeasures for techniques without specific mappings."""
        # General countermeasures based on technique categories
        general_mappings = {
            "T1486": ["D3-BDR", "D3-BAC", "D3-IR"],  # Data Encrypted for Impact
            "T1490": ["D3-BDR", "D3-BAC"],  # Inhibit System Recovery
            "T1059": ["D3-SAW", "D3-AL"],  # Command and Scripting Interpreter
            "T1021": ["D3-NFP", "D3-VPM"],  # Remote Services
            "T1562": ["D3-EDR", "D3-AV"],  # Impair Defenses
        }
        
        return general_mappings.get(technique_id, ["D3-MON", "D3-DET"])  # Default monitoring/detection

    def _log_to_mlflow(self, results: Dict[str, Any]) -> None:
        """Log parsing results to MLflow."""
        try:
            with mlflow.start_run(run_name="mitre_parser_update"):
                mlflow.log_params({
                    "canonical_families_updated": results["canonical_families_updated"],
                    "family_to_attack_updated": results["family_to_attack_updated"],
                    "attack_to_d3fend_updated": results["attack_to_d3fend_updated"],
                    "timestamp": results["timestamp"],
                })
                
                # Log source versions
                for source, version in results["source_versions"].items():
                    mlflow.log_param(f"{source}_version", version)
                
                # Log lookup files as artifacts
                for lookup_file in ["canonical_families.yaml", "family_to_attack.yaml", "attack_to_d3fend.yaml"]:
                    lookup_path = self.lookups_dir / lookup_file
                    if lookup_path.exists():
                        mlflow.log_artifact(str(lookup_path))
        
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")


def update_policy_with_versions(policy_path: Path, results: Dict[str, Any]) -> None:
    """Update policy.json with source versions."""
    if not policy_path.exists():
        return
    
    try:
        with open(policy_path, 'r', encoding='utf-8') as f:
            policy_data = json.load(f)
        
        # Add source versions
        policy_data["lookup_source_versions"] = results["source_versions"]
        policy_data["lookup_last_updated"] = results["timestamp"]
        
        with open(policy_path, 'w', encoding='utf-8') as f:
            json.dump(policy_data, f, indent=2)
        
        logger.info(f"Updated policy.json with lookup versions")
    
    except Exception as e:
        logger.warning(f"Failed to update policy.json: {e}")
