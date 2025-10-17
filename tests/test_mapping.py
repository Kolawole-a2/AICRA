"""Tests for mapping pipeline."""

import pytest
import yaml
from unittest.mock import patch, mock_open

from aicra.pipelines.mapping import MappingPipeline
from aicra.config import Settings


class TestMappingPipeline:
    """Test mapping pipeline functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = Settings()
        self.pipeline = MappingPipeline(self.settings)
    
    def test_normalize_family(self):
        """Test family normalization."""
        # Test exact match
        assert self.pipeline.normalize_family("locky") == "Locky"
        assert self.pipeline.normalize_family("ryuk") == "Ryuk"
        
        # Test pattern matching
        assert self.pipeline.normalize_family("locky_variant") == "Locky"
        assert self.pipeline.normalize_family("ryuk-2.0") == "Ryuk"
        
        # Test case insensitive
        assert self.pipeline.normalize_family("LOCKY") == "Locky"
        assert self.pipeline.normalize_family("RyUk") == "Ryuk"
        
        # Test unknown family
        assert self.pipeline.normalize_family("unknown_family") == "Unknown"
        assert self.pipeline.normalize_family("") == "Unknown"
        assert self.pipeline.normalize_family(None) == "Unknown"
    
    def test_family_to_attack(self):
        """Test family to ATT&CK techniques mapping."""
        techniques = self.pipeline.family_to_attack("Locky")
        assert isinstance(techniques, list)
        assert "T1486" in techniques  # Data Encrypted for Impact
        assert "T1059" in techniques  # Command and Scripting Interpreter
        
        # Test unknown family
        techniques_unknown = self.pipeline.family_to_attack("UnknownFamily")
        assert techniques_unknown == []
    
    def test_attack_to_d3fend(self):
        """Test ATT&CK to D3FEND countermeasures mapping."""
        techniques = ["T1486", "T1059"]
        countermeasures = self.pipeline.attack_to_d3fend(techniques)
        
        assert isinstance(countermeasures, list)
        assert "D3-APAL" in countermeasures  # Application Allowlisting
        assert "D3-FCR" in countermeasures   # File-Content Rules
        
        # Test empty techniques
        empty_countermeasures = self.pipeline.attack_to_d3fend([])
        assert empty_countermeasures == []
    
    def test_get_complete_mapping(self):
        """Test complete mapping from raw tag to techniques and countermeasures."""
        mapping = self.pipeline.get_complete_mapping("locky")
        
        assert mapping["canonical_family"] == "Locky"
        assert isinstance(mapping["techniques"], list)
        assert isinstance(mapping["countermeasures"], list)
        assert "T1486" in mapping["techniques"]
        assert "D3-APAL" in mapping["countermeasures"]
    
    def test_get_all_canonical_families(self):
        """Test getting all canonical families."""
        families = self.pipeline.get_all_canonical_families()
        
        assert isinstance(families, list)
        assert "Locky" in families
        assert "Ryuk" in families
        assert "Unknown" in families
    
    def test_validate_mappings(self):
        """Test mapping validation."""
        validation = self.pipeline.validate_mappings()
        
        assert "canonical_families" in validation
        assert "missing_mappings" in validation
        assert "total_families" in validation
        assert "mapped_families" in validation
        
        assert validation["total_families"] > 0
        assert validation["mapped_families"] >= 0
    
    def test_matches_pattern(self):
        """Test pattern matching functionality."""
        # Test exact match
        assert self.pipeline._matches_pattern("locky", "locky")
        
        # Test wildcard matching
        assert self.pipeline._matches_pattern("locky_variant", "locky.*")
        assert self.pipeline._matches_pattern("locky-2.0", "locky.*")
        
        # Test no match
        assert not self.pipeline._matches_pattern("ryuk", "locky.*")
        
        # Test invalid regex
        assert not self.pipeline._matches_pattern("test", "[invalid")
    
    @patch('aicra.pipelines.mapping.yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_canonical_families(self, mock_file, mock_yaml):
        """Test loading canonical families from YAML."""
        mock_data = {
            "__version__": "1.0.0",
            "mappings": {
                "test_family": "TestFamily"
            }
        }
        mock_yaml.return_value = mock_data
        
        families = self.pipeline._load_canonical_families()
        
        assert families["test_family"] == "TestFamily"
        mock_file.assert_called()
    
    @patch('aicra.pipelines.mapping.yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_family_to_attack(self, mock_file, mock_yaml):
        """Test loading family to ATT&CK mapping from YAML."""
        mock_data = {
            "__version__": "1.0.0",
            "mappings": {
                "TestFamily": ["T1486", "T1059"]
            }
        }
        mock_yaml.return_value = mock_data
        
        mappings = self.pipeline._load_family_to_attack()
        
        assert mappings["TestFamily"] == ["T1486", "T1059"]
        mock_file.assert_called()
    
    @patch('aicra.pipelines.mapping.yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_attack_to_d3fend(self, mock_file, mock_yaml):
        """Test loading ATT&CK to D3FEND mapping from YAML."""
        mock_data = {
            "__version__": "1.0.0",
            "mappings": {
                "T1486": ["D3-APAL", "D3-FCR"]
            }
        }
        mock_yaml.return_value = mock_data
        
        mappings = self.pipeline._load_attack_to_d3fend()
        
        assert mappings["T1486"] == ["D3-APAL", "D3-FCR"]
        mock_file.assert_called()
