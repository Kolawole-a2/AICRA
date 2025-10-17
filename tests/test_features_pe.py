"""Tests for PE feature builders."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, mock_open

from aicra.pipelines.features_pe import PEFeatureBuilder, build_pe_features


class TestPEFeatureBuilder:
    """Test PE feature builder functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = PEFeatureBuilder()
    
    def test_extract_byte_histogram(self):
        """Test byte histogram extraction."""
        # Create test PE data
        pe_data = b'MZ' + b'\x00' * 60 + b'PE\x00\x00' + b'\x00' * 100
        
        features = self.builder._extract_byte_histogram(pe_data)
        
        # Check that we have 256 histogram bins
        assert len([k for k in features.keys() if k.startswith('byte_hist_')]) == 256
        
        # Check that probabilities sum to 1
        hist_values = [v for k, v in features.items() if k.startswith('byte_hist_')]
        assert abs(sum(hist_values) - 1.0) < 1e-6
    
    def test_extract_pe_headers(self):
        """Test PE header extraction."""
        # Create minimal PE data
        pe_data = b'MZ' + b'\x00' * 60 + b'PE\x00\x00' + b'\x00' * 200
        
        features = self.builder._extract_pe_headers(pe_data)
        
        # Check that PE features are present
        assert 'pe_machine' in features
        assert 'pe_num_sections' in features
        assert 'pe_timestamp' in features
        assert 'pe_entry_point' in features
    
    def test_extract_entropy_stats(self):
        """Test entropy statistics extraction."""
        # Create test data with varying entropy
        pe_data = b'MZ' + b'\x00' * 60 + b'PE\x00\x00' + b'\x00' * 200
        
        features = self.builder._extract_entropy_stats(pe_data)
        
        # Check that entropy features are present
        assert 'entropy_overall' in features
        assert 'entropy_section_mean' in features
        assert 'entropy_section_median' in features
        assert 'entropy_section_max' in features
        assert 'entropy_section_std' in features
        assert 'entropy_section_count' in features
        
        # Check that entropy is between 0 and 8 (max for bytes)
        assert 0 <= features['entropy_overall'] <= 8
    
    def test_calculate_entropy(self):
        """Test entropy calculation."""
        # Test with uniform distribution (high entropy)
        uniform_data = bytes(range(256))
        entropy_uniform = self.builder._calculate_entropy(uniform_data)
        assert entropy_uniform > 7.0  # Should be close to 8
        
        # Test with constant data (low entropy)
        constant_data = b'\x00' * 100
        entropy_constant = self.builder._calculate_entropy(constant_data)
        assert entropy_constant == 0.0
        
        # Test with empty data
        entropy_empty = self.builder._calculate_entropy(b'')
        assert entropy_empty == 0.0
    
    def test_extract_features(self):
        """Test complete feature extraction."""
        pe_data = b'MZ' + b'\x00' * 60 + b'PE\x00\x00' + b'\x00' * 200
        
        features = self.builder.extract_features(pe_data)
        
        # Check that all feature types are present
        assert any(k.startswith('byte_hist_') for k in features.keys())
        assert any(k.startswith('pe_') for k in features.keys())
        assert any(k.startswith('entropy_') for k in features.keys())
    
    def test_get_default_pe_features(self):
        """Test default PE features for invalid files."""
        features = self.builder._get_default_pe_features()
        
        # Check that all expected features are present with default values
        assert features['pe_machine'] == 0.0
        assert features['pe_num_sections'] == 0.0
        assert features['pe_timestamp'] == 0.0
        assert features['pe_entry_point'] == 0.0
        
        # Check byte histogram is all zeros
        hist_values = [v for k, v in features.items() if k.startswith('byte_hist_')]
        assert all(v == 0.0 for v in hist_values)


class TestBuildPEFeatures:
    """Test build_pe_features function."""
    
    def test_build_pe_features_success(self):
        """Test successful PE feature building."""
        # Create test dataframe
        df = pd.DataFrame({
            'file_path': ['test_file1.bin', 'test_file2.bin']
        })
        
        # Mock file reading
        test_data = b'MZ' + b'\x00' * 60 + b'PE\x00\x00' + b'\x00' * 200
        
        with patch('builtins.open', mock_open(read_data=test_data)):
            features_df = build_pe_features(df)
        
        # Check that features are extracted
        assert len(features_df) == 2
        assert len(features_df.columns) > 256  # At least byte histogram + PE + entropy features
    
    def test_build_pe_features_file_not_found(self):
        """Test handling of file not found errors."""
        df = pd.DataFrame({
            'file_path': ['nonexistent_file.bin']
        })
        
        features_df = build_pe_features(df)
        
        # Should return default features
        assert len(features_df) == 1
        assert features_df['pe_machine'].iloc[0] == 0.0
    
    def test_build_pe_features_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame(columns=['file_path'])
        
        features_df = build_pe_features(df)
        
        assert len(features_df) == 0
        assert len(features_df.columns) > 0  # Should have feature columns
