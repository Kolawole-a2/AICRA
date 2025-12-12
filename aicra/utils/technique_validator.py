"""
Technique ID Validation and Normalization Module

This module provides utilities to validate and normalize MITRE ATT&CK technique IDs
for use in the H3 evaluation pipeline.

Validates:
- Main techniques (e.g., T1059)
- Subtechniques (e.g., T1059.001)
- Handles whitespace, casing, and stray characters

Normalizes:
- Consistent capitalization
- Trimming whitespace
- Flagging invalid/missing IDs
"""

import re
import logging
from typing import List, Set, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Pattern for valid ATT&CK technique IDs
# Main techniques: T[0-9]{4}
# Subtechniques: T[0-9]{4}\.[0-9]{3}
TECHNIQUE_PATTERN = re.compile(r'^T\d{4}(\.\d{3})?$', re.IGNORECASE)

# Common ATT&CK technique prefixes (for validation)
VALID_TECHNIQUE_PREFIXES = {'T'}


def normalize_technique_id(tech_id: str) -> Optional[str]:
    """
    Normalize a technique ID string.
    
    Args:
        tech_id: Raw technique ID string (may have whitespace, casing issues)
        
    Returns:
        Normalized technique ID (uppercase, trimmed) or None if invalid
    """
    if pd.isna(tech_id) or tech_id == '' or tech_id == ' ':
        return None
    
    # Convert to string and strip whitespace
    tech_id = str(tech_id).strip()
    
    # Remove any leading/trailing whitespace
    tech_id = tech_id.strip()
    
    # Convert to uppercase for consistency
    tech_id = tech_id.upper()
    
    # Validate pattern
    if TECHNIQUE_PATTERN.match(tech_id):
        return tech_id
    
    # Try to fix common issues
    # Remove any non-alphanumeric characters except T, digits, and dot
    cleaned = re.sub(r'[^T0-9.]', '', tech_id)
    
    # Try to extract pattern
    match = TECHNIQUE_PATTERN.search(cleaned)
    if match:
        return match.group(0)
    
    return None


def validate_technique_id(tech_id: str, valid_techniques: Optional[Set[str]] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate and normalize a technique ID.
    
    Args:
        tech_id: Raw technique ID string
        valid_techniques: Optional set of known valid technique IDs from mappings
        
    Returns:
        Tuple of (is_valid, normalized_id)
        - is_valid: True if the ID is valid
        - normalized_id: Normalized ID if valid, None otherwise
    """
    normalized = normalize_technique_id(tech_id)
    
    if normalized is None:
        return False, None
    
    # If we have a set of valid techniques, check if this one is in it
    if valid_techniques is not None:
        if normalized not in valid_techniques:
            logger.debug(f"Technique {normalized} not found in valid techniques set")
            # Still return True for pattern validity, but log the warning
            return True, normalized
    
    return True, normalized


def validate_technique_column(
    df: pd.DataFrame,
    technique_col: str = 'technique_id',
    valid_techniques: Optional[Set[str]] = None,
    drop_invalid: bool = False
) -> Tuple[pd.DataFrame, dict]:
    """
    Validate and normalize technique IDs in a DataFrame column.
    
    Args:
        df: DataFrame with technique IDs
        technique_col: Name of the column containing technique IDs
        valid_techniques: Optional set of known valid technique IDs
        drop_invalid: If True, drop rows with invalid technique IDs
        
    Returns:
        Tuple of (validated_df, diagnostics_dict)
        - validated_df: DataFrame with normalized technique IDs (and optionally invalid rows removed)
        - diagnostics_dict: Dictionary with validation statistics
    """
    if technique_col not in df.columns:
        logger.warning(f"Column {technique_col} not found in DataFrame")
        return df, {
            'total_rows': len(df),
            'valid_rows': 0,
            'invalid_rows': len(df),
            'unique_valid_techniques': 0,
            'unique_invalid_techniques': 0,
            'invalid_ids': []
        }
    
    diagnostics = {
        'total_rows': len(df),
        'valid_rows': 0,
        'invalid_rows': 0,
        'unique_valid_techniques': 0,
        'unique_invalid_techniques': 0,
        'invalid_ids': []
    }
    
    # Normalize all technique IDs
    normalized_ids = []
    invalid_mask = []
    
    for idx, tech_id in enumerate(df[technique_col]):
        is_valid, normalized = validate_technique_id(tech_id, valid_techniques)
        
        if is_valid and normalized is not None:
            normalized_ids.append(normalized)
            invalid_mask.append(False)
        else:
            normalized_ids.append(None)
            invalid_mask.append(True)
            if tech_id not in diagnostics['invalid_ids']:
                diagnostics['invalid_ids'].append(str(tech_id))
    
    # Update DataFrame
    df_validated = df.copy()
    df_validated[technique_col] = normalized_ids
    
    # Count statistics
    diagnostics['valid_rows'] = sum(1 for x in normalized_ids if x is not None)
    diagnostics['invalid_rows'] = sum(1 for x in normalized_ids if x is None)
    diagnostics['unique_valid_techniques'] = len(set(x for x in normalized_ids if x is not None))
    diagnostics['unique_invalid_techniques'] = len(set(x for x in df[technique_col] if x not in normalized_ids))
    
    # Drop invalid rows if requested
    if drop_invalid:
        df_validated = df_validated[~pd.Series(invalid_mask)].copy()
        logger.info(f"Dropped {diagnostics['invalid_rows']} rows with invalid technique IDs")
    
    return df_validated, diagnostics


def extract_valid_techniques_from_mapping(mapping_df: pd.DataFrame, technique_col: str = 'technique_id') -> Set[str]:
    """
    Extract set of valid technique IDs from a mapping DataFrame.
    
    Args:
        mapping_df: Mapping DataFrame
        technique_col: Name of the column containing technique IDs (or 'attack_id' as fallback)
        
    Returns:
        Set of normalized valid technique IDs
    """
    # Handle both 'technique_id' and 'attack_id' column names
    if technique_col not in mapping_df.columns:
        if 'attack_id' in mapping_df.columns:
            technique_col = 'attack_id'
        else:
            return set()
    
    valid_techniques = set()
    for tech_id in mapping_df[technique_col].dropna():
        is_valid, normalized = validate_technique_id(tech_id)
        if is_valid and normalized is not None:
            valid_techniques.add(normalized)
    
    return valid_techniques


def validate_risk_scores_file(
    file_path: str,
    valid_techniques: Optional[Set[str]] = None,
    drop_invalid: bool = False
) -> Tuple[pd.DataFrame, dict]:
    """
    Load and validate a risk scores CSV file.
    
    Args:
        file_path: Path to risk_scores.csv file
        valid_techniques: Optional set of valid technique IDs from mappings
        drop_invalid: If True, drop rows with invalid technique IDs
        
    Returns:
        Tuple of (validated_df, diagnostics_dict)
    """
    import pandas as pd
    from pathlib import Path
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"Risk scores file not found: {file_path}")
        return pd.DataFrame(), {
            'file_exists': False,
            'total_rows': 0,
            'valid_rows': 0,
            'invalid_rows': 0
        }
    
    # Load with keep_default_na=False to preserve empty strings
    df = pd.read_csv(file_path, keep_default_na=False)
    
    # Replace empty strings with pd.NA for consistent handling
    if 'technique_id' in df.columns:
        df['technique_id'] = df['technique_id'].replace('', pd.NA).replace(' ', pd.NA)
    
    # Validate technique IDs
    df_validated, diagnostics = validate_technique_column(
        df,
        technique_col='technique_id',
        valid_techniques=valid_techniques,
        drop_invalid=drop_invalid
    )
    
    diagnostics['file_exists'] = True
    diagnostics['file_path'] = str(file_path)
    
    return df_validated, diagnostics
