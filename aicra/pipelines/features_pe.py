"""PE static feature builders for ransomware detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple
import struct
import math


class PEFeatureBuilder:
    """Builder for PE static features including byte histogram, headers, and entropy."""
    
    def __init__(self):
        self.byte_histogram_bins = 256
        self.entropy_window_size = 1024
    
    def extract_features(self, pe_data: bytes) -> Dict[str, Any]:
        """Extract all PE static features from binary data."""
        features = {}
        
        # Byte histogram
        features.update(self._extract_byte_histogram(pe_data))
        
        # PE headers and section metadata
        features.update(self._extract_pe_headers(pe_data))
        
        # Section entropy statistics
        features.update(self._extract_entropy_stats(pe_data))
        
        return features
    
    def _extract_byte_histogram(self, pe_data: bytes) -> Dict[str, float]:
        """Extract 256-bin byte histogram features."""
        histogram = np.zeros(self.byte_histogram_bins, dtype=np.float32)
        
        for byte_val in pe_data:
            histogram[byte_val] += 1
        
        # Normalize to probabilities
        total_bytes = len(pe_data)
        if total_bytes > 0:
            histogram = histogram / total_bytes
        
        # Return as dictionary with byte_hist_ prefix
        features = {}
        for i in range(self.byte_histogram_bins):
            features[f"byte_hist_{i:03d}"] = float(histogram[i])
        
        return features
    
    def _extract_pe_headers(self, pe_data: bytes) -> Dict[str, Any]:
        """Extract PE headers and section metadata."""
        features = {}
        
        try:
            # Check for PE signature
            if len(pe_data) < 64 or pe_data[:2] != b'MZ':
                # Not a valid PE file
                return self._get_default_pe_features()
            
            # Find PE header offset
            pe_offset = struct.unpack('<I', pe_data[60:64])[0]
            if pe_offset >= len(pe_data) - 4:
                return self._get_default_pe_features()
            
            # Check PE signature
            if pe_data[pe_offset:pe_offset+4] != b'PE\x00\x00':
                return self._get_default_pe_features()
            
            # COFF header
            coff_header_offset = pe_offset + 4
            machine = struct.unpack('<H', pe_data[coff_header_offset:coff_header_offset+2])[0]
            num_sections = struct.unpack('<H', pe_data[coff_header_offset+2:coff_header_offset+4])[0]
            timestamp = struct.unpack('<I', pe_data[coff_header_offset+4:coff_header_offset+8])[0]
            entry_point = struct.unpack('<I', pe_data[coff_header_offset+16:coff_header_offset+20])[0]
            
            features.update({
                'pe_machine': float(machine),
                'pe_num_sections': float(num_sections),
                'pe_timestamp': float(timestamp),
                'pe_entry_point': float(entry_point),
            })
            
            # Optional header
            opt_header_offset = coff_header_offset + 20
            magic = struct.unpack('<H', pe_data[opt_header_offset:opt_header_offset+2])[0]
            features['pe_magic'] = float(magic)
            
            # Section headers
            section_header_offset = opt_header_offset + 96  # Standard optional header size
            section_sizes = []
            section_flags = []
            
            for i in range(min(num_sections, 16)):  # Limit to 16 sections for safety
                section_offset = section_header_offset + (i * 40)
                if section_offset + 40 > len(pe_data):
                    break
                
                # Section characteristics (flags)
                characteristics = struct.unpack('<I', pe_data[section_offset+36:section_offset+40])[0]
                section_flags.append(characteristics)
                
                # Virtual size
                virtual_size = struct.unpack('<I', pe_data[section_offset+8:section_offset+12])[0]
                section_sizes.append(virtual_size)
            
            # Aggregate section statistics
            features.update({
                'pe_section_count': float(len(section_sizes)),
                'pe_section_size_mean': float(np.mean(section_sizes)) if section_sizes else 0.0,
                'pe_section_size_std': float(np.std(section_sizes)) if section_sizes else 0.0,
                'pe_section_size_max': float(np.max(section_sizes)) if section_sizes else 0.0,
                'pe_section_flags_mean': float(np.mean(section_flags)) if section_flags else 0.0,
            })
            
        except (struct.error, IndexError, ValueError):
            # If parsing fails, return default features
            return self._get_default_pe_features()
        
        return features
    
    def _extract_entropy_stats(self, pe_data: bytes) -> Dict[str, float]:
        """Extract entropy statistics for sections and overall file."""
        features = {}
        
        try:
            # Overall file entropy
            overall_entropy = self._calculate_entropy(pe_data)
            features['entropy_overall'] = overall_entropy
            
            # Section-wise entropy (if PE file)
            if len(pe_data) > 64 and pe_data[:2] == b'MZ':
                pe_offset = struct.unpack('<I', pe_data[60:64])[0]
                if pe_offset < len(pe_data) - 4 and pe_data[pe_offset:pe_offset+4] == b'PE\x00\x00':
                    section_entropies = self._extract_section_entropies(pe_data, pe_offset)
                    
                    if section_entropies:
                        features.update({
                            'entropy_section_mean': float(np.mean(section_entropies)),
                            'entropy_section_median': float(np.median(section_entropies)),
                            'entropy_section_max': float(np.max(section_entropies)),
                            'entropy_section_std': float(np.std(section_entropies)),
                            'entropy_section_count': float(len(section_entropies)),
                        })
                    else:
                        features.update({
                            'entropy_section_mean': 0.0,
                            'entropy_section_median': 0.0,
                            'entropy_section_max': 0.0,
                            'entropy_section_std': 0.0,
                            'entropy_section_count': 0.0,
                        })
                else:
                    features.update({
                        'entropy_section_mean': 0.0,
                        'entropy_section_median': 0.0,
                        'entropy_section_max': 0.0,
                        'entropy_section_std': 0.0,
                        'entropy_section_count': 0.0,
                    })
            else:
                features.update({
                    'entropy_section_mean': 0.0,
                    'entropy_section_median': 0.0,
                    'entropy_section_max': 0.0,
                    'entropy_section_std': 0.0,
                    'entropy_section_count': 0.0,
                })
                
        except Exception:
            features.update({
                'entropy_overall': 0.0,
                'entropy_section_mean': 0.0,
                'entropy_section_median': 0.0,
                'entropy_section_max': 0.0,
                'entropy_section_std': 0.0,
                'entropy_section_count': 0.0,
            })
        
        return features
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = np.zeros(256, dtype=np.int32)
        for byte_val in data:
            byte_counts[byte_val] += 1
        
        # Calculate entropy
        total_bytes = len(data)
        entropy = 0.0
        
        for count in byte_counts:
            if count > 0:
                probability = count / total_bytes
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _extract_section_entropies(self, pe_data: bytes, pe_offset: int) -> List[float]:
        """Extract entropy for each PE section."""
        entropies = []
        
        try:
            # Get number of sections
            coff_header_offset = pe_offset + 4
            num_sections = struct.unpack('<H', pe_data[coff_header_offset+2:coff_header_offset+4])[0]
            
            # Optional header
            opt_header_offset = coff_header_offset + 20
            section_header_offset = opt_header_offset + 96
            
            for i in range(min(num_sections, 16)):  # Limit for safety
                section_offset = section_header_offset + (i * 40)
                if section_offset + 40 > len(pe_data):
                    break
                
                # Get section data offset and size
                raw_data_offset = struct.unpack('<I', pe_data[section_offset+20:section_offset+24])[0]
                raw_data_size = struct.unpack('<I', pe_data[section_offset+16:section_offset+20])[0]
                
                if raw_data_offset > 0 and raw_data_size > 0 and raw_data_offset + raw_data_size <= len(pe_data):
                    section_data = pe_data[raw_data_offset:raw_data_offset + raw_data_size]
                    entropy = self._calculate_entropy(section_data)
                    entropies.append(entropy)
        
        except (struct.error, IndexError, ValueError):
            pass
        
        return entropies
    
    def _get_default_pe_features(self) -> Dict[str, float]:
        """Return default PE features for invalid/non-PE files."""
        features = {
            'pe_machine': 0.0,
            'pe_num_sections': 0.0,
            'pe_timestamp': 0.0,
            'pe_entry_point': 0.0,
            'pe_magic': 0.0,
            'pe_section_count': 0.0,
            'pe_section_size_mean': 0.0,
            'pe_section_size_std': 0.0,
            'pe_section_size_max': 0.0,
            'pe_section_flags_mean': 0.0,
        }
        
        # Add default byte histogram
        for i in range(self.byte_histogram_bins):
            features[f"byte_hist_{i:03d}"] = 0.0
        
        return features


def build_pe_features(dataframe: pd.DataFrame, file_path_column: str = 'file_path') -> pd.DataFrame:
    """Build PE features for a dataframe containing file paths."""
    builder = PEFeatureBuilder()
    feature_rows = []
    
    for idx, row in dataframe.iterrows():
        try:
            with open(row[file_path_column], 'rb') as f:
                pe_data = f.read()
            
            features = builder.extract_features(pe_data)
            feature_rows.append(features)
            
        except (FileNotFoundError, IOError, OSError):
            # If file can't be read, use default features
            features = builder._get_default_pe_features()
            feature_rows.append(features)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(feature_rows)
    
    # Ensure all columns are present and fill missing values
    expected_columns = set()
    for i in range(256):
        expected_columns.add(f"byte_hist_{i:03d}")
    
    expected_columns.update([
        'pe_machine', 'pe_num_sections', 'pe_timestamp', 'pe_entry_point',
        'pe_magic', 'pe_section_count', 'pe_section_size_mean', 'pe_section_size_std',
        'pe_section_size_max', 'pe_section_flags_mean', 'entropy_overall',
        'entropy_section_mean', 'entropy_section_median', 'entropy_section_max',
        'entropy_section_std', 'entropy_section_count'
    ])
    
    for col in expected_columns:
        if col not in features_df.columns:
            features_df[col] = 0.0
    
    return features_df
