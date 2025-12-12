"""Tests for heuristic ATT&CK→D3FEND mapping."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from aicra.mappings.heuristic_mapping import (
    HeuristicMappingConfig,
    build_heuristic_mapping,
    load_attack_techniques_with_descriptions,
    load_d3fend_controls_with_descriptions,
    set_seeds,
)


@pytest.fixture
def sample_attack_data():
    """Create sample ATT&CK techniques with obvious textual overlap."""
    return pd.DataFrame({
        "technique_id": ["T1486", "T1055", "T1070"],
        "name": [
            "Data Encrypted for Impact",
            "Process Injection",
            "Indicator Removal on Host",
        ],
        "description": [
            "Adversaries may encrypt data on target systems or on large numbers of systems in a network to interrupt availability to system and network resources. This technique is commonly used by ransomware.",
            "Adversaries may inject code into processes in order to evade process-based defenses as well as possibly elevate privileges.",
            "Adversaries may delete or modify artifacts generated on a system to remove evidence of their presence or hinder defenses.",
        ],
    })


@pytest.fixture
def sample_d3fend_data():
    """Create sample D3FEND controls with obvious textual overlap."""
    return pd.DataFrame({
        "control_id": ["D3-RA", "D3-PE", "D3-EDR", "D3-FA"],
        "name": [
            "Restore Access",
            "Process Eviction",
            "Endpoint Detection and Response",
            "File Analysis",
        ],
        "description": [
            "Restore access to encrypted files and systems by recovering from backups or using decryption keys.",
            "Evict malicious processes from the system to prevent unauthorized code execution.",
            "Monitor and detect suspicious activities on endpoints to identify and respond to threats.",
            "Analyze file contents and metadata to detect malicious files and prevent unauthorized access.",
        ],
    })


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


def test_build_heuristic_mapping_basic(sample_attack_data, sample_d3fend_data, temp_dir):
    """Test basic heuristic mapping functionality."""
    # Save sample data to CSV files
    attack_csv = temp_dir / "attack_techniques.csv"
    d3fend_csv = temp_dir / "d3fend_controls.csv"
    
    sample_attack_data.to_csv(attack_csv, index=False)
    sample_d3fend_data.to_csv(d3fend_csv, index=False)
    
    # Build mapping with low threshold to ensure matches
    config = HeuristicMappingConfig(
        top_k=2,
        min_similarity=0.20,  # Low threshold for test
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        seed=42,
    )
    
    # Build mapping
    result_df = build_heuristic_mapping(
        attack_path=str(attack_csv),
        d3fend_path=str(d3fend_csv),
        config=config,
    )
    
    # Assertions
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) > 0, "Should have at least one mapping"
    
    # Check required columns
    required_cols = ["technique_id", "control_id", "similarity_score"]
    assert all(col in result_df.columns for col in required_cols), \
        f"Missing required columns. Got: {list(result_df.columns)}"
    
    # Check similarity scores are in [0, 1]
    assert result_df["similarity_score"].min() >= 0.0, "Similarity scores should be >= 0"
    assert result_df["similarity_score"].max() <= 1.0, "Similarity scores should be <= 1"
    
    # Check that each technique appears at least once
    unique_techniques = result_df["technique_id"].unique()
    assert len(unique_techniques) > 0, "Should have mappings for at least one technique"
    
    # Check that we have mappings for all techniques (if suitable controls exist)
    # With low threshold, we should get mappings for all techniques
    assert len(unique_techniques) == len(sample_attack_data), \
        f"Should have mappings for all techniques. Got: {unique_techniques}"


def test_build_heuristic_mapping_min_similarity_filtering(
    sample_attack_data, sample_d3fend_data, temp_dir
):
    """Test that min_similarity threshold filters out weak matches."""
    # Save sample data
    attack_csv = temp_dir / "attack_techniques.csv"
    d3fend_csv = temp_dir / "d3fend_controls.csv"
    
    sample_attack_data.to_csv(attack_csv, index=False)
    sample_d3fend_data.to_csv(d3fend_csv, index=False)
    
    # Build mapping with high threshold
    config_high = HeuristicMappingConfig(
        top_k=10,  # High top_k to get all matches
        min_similarity=0.90,  # Very high threshold
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        seed=42,
    )
    
    result_high = build_heuristic_mapping(
        attack_path=str(attack_csv),
        d3fend_path=str(d3fend_csv),
        config=config_high,
    )
    
    # Build mapping with low threshold
    config_low = HeuristicMappingConfig(
        top_k=10,
        min_similarity=0.20,  # Low threshold
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        seed=42,
    )
    
    result_low = build_heuristic_mapping(
        attack_path=str(attack_csv),
        d3fend_path=str(d3fend_csv),
        config=config_low,
    )
    
    # High threshold should have fewer or equal mappings
    assert len(result_high) <= len(result_low), \
        "High threshold should filter out more mappings"
    
    # All similarities in high threshold result should be >= threshold
    if len(result_high) > 0:
        assert result_high["similarity_score"].min() >= config_high.min_similarity, \
            "All similarities should meet the high threshold"


def test_build_heuristic_mapping_top_k(sample_attack_data, sample_d3fend_data, temp_dir):
    """Test that top_k limits the number of controls per technique."""
    # Save sample data
    attack_csv = temp_dir / "attack_techniques.csv"
    d3fend_csv = temp_dir / "d3fend_controls.csv"
    
    sample_attack_data.to_csv(attack_csv, index=False)
    sample_d3fend_data.to_csv(d3fend_csv, index=False)
    
    # Build mapping with top_k=1
    config = HeuristicMappingConfig(
        top_k=1,
        min_similarity=0.20,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        seed=42,
    )
    
    result = build_heuristic_mapping(
        attack_path=str(attack_csv),
        d3fend_path=str(d3fend_csv),
        config=config,
    )
    
    # Each technique should have at most top_k mappings
    if len(result) > 0:
        technique_counts = result["technique_id"].value_counts()
        assert technique_counts.max() <= config.top_k, \
            f"Each technique should have at most {config.top_k} mappings"


def test_set_seeds():
    """Test that seed setting works."""
    set_seeds(42)
    # Just verify it doesn't raise an error
    assert True


def test_load_attack_techniques_with_descriptions_csv(sample_attack_data, temp_dir):
    """Test loading ATT&CK techniques from CSV."""
    csv_path = temp_dir / "attack_techniques.csv"
    sample_attack_data.to_csv(csv_path, index=False)
    
    result = load_attack_techniques_with_descriptions(csv_path=csv_path)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_attack_data)
    assert "technique_id" in result.columns
    assert "name" in result.columns
    assert "description" in result.columns


def test_load_d3fend_controls_with_descriptions_csv(sample_d3fend_data, temp_dir):
    """Test loading D3FEND controls from CSV."""
    csv_path = temp_dir / "d3fend_controls.csv"
    sample_d3fend_data.to_csv(csv_path, index=False)
    
    result = load_d3fend_controls_with_descriptions(csv_path=csv_path)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_d3fend_data)
    assert "control_id" in result.columns
    assert "name" in result.columns
    assert "description" in result.columns


@pytest.mark.slow
def test_build_heuristic_mapping_deterministic(sample_attack_data, sample_d3fend_data, temp_dir):
    """Test that mapping is deterministic with same seed."""
    # Save sample data
    attack_csv = temp_dir / "attack_techniques.csv"
    d3fend_csv = temp_dir / "d3fend_controls.csv"
    
    sample_attack_data.to_csv(attack_csv, index=False)
    sample_d3fend_data.to_csv(d3fend_csv, index=False)
    
    config = HeuristicMappingConfig(
        top_k=2,
        min_similarity=0.20,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        seed=42,
    )
    
    # Build mapping twice with same seed
    result1 = build_heuristic_mapping(
        attack_path=str(attack_csv),
        d3fend_path=str(d3fend_csv),
        config=config,
    )
    
    result2 = build_heuristic_mapping(
        attack_path=str(attack_csv),
        d3fend_path=str(d3fend_csv),
        config=config,
    )
    
    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2, check_dtype=False)

