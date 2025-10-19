"""Comprehensive tests for mapping pipeline scalability and coverage."""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import List, Dict

import pytest
import pandas as pd
import yaml

from aicra.config import Settings
from aicra.pipelines.mapping import MappingPipeline
from aicra.utils.normalize import FamilyNormalizer


class TestMappingScale:
    """Test mapping pipeline scalability and performance."""
    
    @pytest.fixture
    def temp_settings(self):
        """Create temporary settings for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test lookup files
            lookups_dir = temp_path / "data" / "lookups"
            lookups_dir.mkdir(parents=True)
            
            # Create canonical families mapping
            families_data = {
                '__version__': '1.0.0-test',
                'mappings': {
                    'lockbit': 'LockBit',
                    'lock_bit': 'LockBit',
                    'conti': 'Conti',
                    'ryuk': 'Ryuk',
                    'unknown': 'Unknown',
                    'test*': 'TestFamily'
                }
            }
            
            with open(lookups_dir / "canonical_families.yaml", 'w') as f:
                yaml.dump(families_data, f)
            
            # Create family to attack mapping
            attack_data = {
                '__version__': '1.0.0-test',
                'mappings': {
                    'LockBit': ['T1486', 'T1055'],
                    'Conti': ['T1486', 'T1027'],
                    'Ryuk': ['T1486', 'T1055', 'T1027'],
                    'Unknown': [],
                    'TestFamily': ['T0000']
                }
            }
            
            with open(lookups_dir / "family_to_attack.yaml", 'w') as f:
                yaml.dump(attack_data, f)
            
            # Create attack to d3fend mapping
            d3fend_data = {
                '__version__': '1.0.0-test',
                'mappings': {
                    'T1486': ['D3-CM001', 'D3-CM002'],
                    'T1055': ['D3-CM003'],
                    'T1027': ['D3-CM004'],
                    'T0000': ['D3-CM000']
                }
            }
            
            with open(lookups_dir / "attack_to_d3fend.yaml", 'w') as f:
                yaml.dump(d3fend_data, f)
            
            # Create settings
            settings = Settings()
            settings.data_dir = temp_path / "data"
            settings.artifacts_dir = temp_path / "artifacts"
            settings.artifacts_dir.mkdir(parents=True)
            
            yield settings
    
    def test_normalization_performance(self):
        """Test normalization performance with large datasets."""
        normalizer = FamilyNormalizer()
        
        # Generate test data
        test_families = [
            'LockBit', 'lockbit', 'LOCKBIT', 'Lock-Bit', 'lock_bit',
            'Conti', 'conti', 'CONTI', 'Conti-Ransomware',
            'Ryuk', 'ryuk', 'RYUK', 'Ryuk-Ransomware',
            'Unknown', 'unknown', 'UNKNOWN', 'unmapped',
            'TestFamily', 'test_family', 'TEST-FAMILY'
        ] * 1000  # 20,000 samples
        
        start_time = time.time()
        normalized = normalizer.normalize_batch(test_families)
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed_time < 5.0, f"Normalization took too long: {elapsed_time:.2f}s"
        
        # Check results
        assert len(normalized) == len(test_families)
        assert all(isinstance(n, str) for n in normalized)
    
    def test_batch_mapping_performance(self, temp_settings):
        """Test batch mapping performance with large datasets."""
        mapping_pipeline = MappingPipeline(temp_settings, skip_mlflow=True)
        
        # Generate synthetic alias data
        known_families = ['lockbit', 'conti', 'ryuk', 'test_family']
        unknown_families = ['unknown_family_1', 'unknown_family_2', 'unmapped_family']
        
        # Create 100k samples with mixture of known/unknown
        raw_families = []
        for i in range(100000):
            if i % 10 == 0:  # 10% unknown
                raw_families.append(unknown_families[i % len(unknown_families)])
            else:
                raw_families.append(known_families[i % len(known_families)])
        
        # Test batch mapping performance
        start_time = time.time()
        result_df = mapping_pipeline.map_families_batch(raw_families, "test")
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (configurable threshold)
        max_time = 30.0  # 30 seconds max
        assert elapsed_time < max_time, f"Batch mapping took too long: {elapsed_time:.2f}s"
        
        # Check results (batch mapping aggregates by unique families)
        unique_families = len(set(raw_families))
        assert len(result_df) == unique_families
        assert 'canonical_family' in result_df.columns
        assert 'techniques' in result_df.columns
        assert 'd3fend_controls' in result_df.columns
        
        # Check coverage
        coverage_metrics = mapping_pipeline.compute_coverage_metrics()
        assert 'alias_to_family_coverage' in coverage_metrics
        assert 'family_to_attack_coverage' in coverage_metrics
        assert 'attack_to_d3fend_coverage' in coverage_metrics
    
    def test_coverage_logging(self, temp_settings):
        """Test coverage logging functionality."""
        mapping_pipeline = MappingPipeline(temp_settings, skip_mlflow=True)
        
        # Test with known and unknown families
        test_families = ['lockbit', 'conti', 'unknown_family', 'test_family']
        
        # Run batch mapping
        mapping_pipeline.map_families_batch(test_families, "test")
        
        # Log coverage report
        coverage_metrics = mapping_pipeline.log_coverage_report("test")
        
        # Check coverage file was created
        coverage_file = temp_settings.artifacts_dir / "mapping_coverage_test.json"
        assert coverage_file.exists()
        
        with open(coverage_file, 'r') as f:
            coverage_data = json.load(f)
        
        assert coverage_data['phase'] == 'test'
        assert 'coverage_metrics' in coverage_data
        assert 'coverage_stats' in coverage_data
        
        # Check unmapped report was created
        unmapped_file = temp_settings.artifacts_dir / "unmapped_report_test.csv"
        assert unmapped_file.exists()
        
        unmapped_df = pd.read_csv(unmapped_file)
        assert 'stage' in unmapped_df.columns
        assert 'unmapped_item' in unmapped_df.columns
        assert 'count' in unmapped_df.columns
    
    def test_coverage_thresholds(self, temp_settings):
        """Test coverage threshold checking."""
        mapping_pipeline = MappingPipeline(temp_settings, skip_mlflow=True)
        
        # Test with mostly unknown families (should fail threshold)
        unknown_families = ['unknown1', 'unknown2', 'unknown3'] * 1000
        
        mapping_pipeline.map_families_batch(unknown_families, "test")
        
        # Should fail coverage check
        assert not mapping_pipeline.check_coverage_thresholds("test")
        
        # Test with mostly known families (should pass threshold)
        known_families = ['lockbit', 'conti', 'ryuk'] * 1000
        
        # Reset coverage stats
        mapping_pipeline.coverage_stats = {
            'alias_to_family': {'mapped': 0, 'total': 0, 'unmapped': []},
            'family_to_attack': {'mapped': 0, 'total': 0, 'unmapped': []},
            'attack_to_d3fend': {'mapped': 0, 'total': 0, 'unmapped': []}
        }
        
        mapping_pipeline.map_families_batch(known_families, "test")
        
        # Should pass coverage check
        assert mapping_pipeline.check_coverage_thresholds("test")
    
    def test_memory_usage(self, temp_settings):
        """Test memory usage stays reasonable with large datasets."""
        import psutil
        import os
        
        mapping_pipeline = MappingPipeline(temp_settings, skip_mlflow=True)
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate large dataset
        test_families = ['lockbit', 'conti', 'ryuk', 'unknown'] * 25000  # 100k samples
        
        # Run batch mapping
        result_df = mapping_pipeline.map_families_batch(test_families, "test")
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 1GB for 100k samples)
        assert memory_increase < 1000, f"Memory usage increased too much: {memory_increase:.1f}MB"
        
        # Verify results
        assert len(result_df) == len(test_families)
    
    def test_pattern_matching_performance(self, temp_settings):
        """Test pattern matching performance with wildcards."""
        mapping_pipeline = MappingPipeline(temp_settings, skip_mlflow=True)
        
        # Test pattern matching with wildcards
        test_cases = [
            'test_family_1',
            'test_family_2', 
            'test_family_3',
            'test_something_else',
            'not_test_family'
        ] * 1000  # 5k samples
        
        start_time = time.time()
        results = [mapping_pipeline.normalize_family(family) for family in test_cases]
        elapsed_time = time.time() - start_time
        
        # Should complete quickly
        assert elapsed_time < 2.0, f"Pattern matching took too long: {elapsed_time:.2f}s"
        
        # Check results
        test_family_results = [r for r in results if r == 'TestFamily']
        assert len(test_family_results) == 4000  # 4 out of 5 patterns should match
    
    def test_caching_effectiveness(self, temp_settings):
        """Test that caching improves performance."""
        mapping_pipeline = MappingPipeline(temp_settings, skip_mlflow=True)
        
        # Test repeated lookups
        test_families = ['lockbit', 'conti', 'ryuk'] * 1000
        
        # First run
        start_time = time.time()
        mapping_pipeline.map_families_batch(test_families, "test1")
        first_run_time = time.time() - start_time
        
        # Second run (should be faster due to caching)
        start_time = time.time()
        mapping_pipeline.map_families_batch(test_families, "test2")
        second_run_time = time.time() - start_time
        
        # Second run should be faster (or at least not significantly slower)
        assert second_run_time <= first_run_time * 1.5, "Caching not effective"
    
    def test_error_handling(self, temp_settings):
        """Test error handling with malformed data."""
        mapping_pipeline = MappingPipeline(temp_settings, skip_mlflow=True)
        
        # Test with various malformed inputs
        malformed_families = [
            None,
            '',
            '   ',
            123,
            [],
            {},
            'valid_family',
            'lockbit',
            'conti'
        ]
        
        # Should handle gracefully
        result_df = mapping_pipeline.map_families_batch(malformed_families, "test")
        
        # Check results
        assert len(result_df) == len(malformed_families)
        assert all(isinstance(family, str) for family in result_df['canonical_family'])
    
    def test_concurrent_access(self, temp_settings):
        """Test thread safety of mapping pipeline."""
        import threading
        import queue
        
        mapping_pipeline = MappingPipeline(temp_settings, skip_mlflow=True)
        
        # Test concurrent access
        results_queue = queue.Queue()
        
        def worker(worker_id: int):
            test_families = ['lockbit', 'conti', 'ryuk'] * 100
            result_df = mapping_pipeline.map_families_batch(test_families, f"worker_{worker_id}")
            results_queue.put((worker_id, len(result_df)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 5
        for worker_id, count in results:
            assert count == 300  # 3 families * 100 repetitions
