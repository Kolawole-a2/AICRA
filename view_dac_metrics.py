#!/usr/bin/env python3
"""
View DAC Metric Lookup Table
Compares deterministic vs learned defense-attack mappings and shows ransomware coverage.
"""

import pandas as pd
import yaml
from pathlib import Path
import json

# Load deterministic mapping from YAML
def load_deterministic_mapping():
    """Load deterministic ATT&CK to D3FEND mapping."""
    d3fend_path = Path("data/lookups/attack_to_d3fend.yaml")
    with open(d3fend_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get("mappings", {})

# Load family to attack mapping to identify ransomware families
def load_family_to_attack():
    """Load family to ATT&CK techniques mapping."""
    attack_path = Path("data/lookups/family_to_attack.yaml")
    with open(attack_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get("mappings", {})

# Check for learned mapping (might be in CSV or other format)
def load_learned_mapping():
    """Attempt to load learned mapping if it exists."""
    learned_paths = [
        Path("learned_mapping.csv"),
        Path("results/H3_full_evaluation/learned_mapping.csv"),
        Path("data/mappings/learned_mapping.csv"),
        Path("mappings/learned_mapping.csv"),
    ]
    
    for path in learned_paths:
        if path.exists():
            df = pd.read_csv(path)
            if "technique_id" in df.columns and "control_id" in df.columns:
                return df
    
    return None

# Build DAC metric lookup table
def build_dac_table():
    """Build the DAC metric lookup table."""
    print("Loading mappings...")
    
    # Load deterministic mapping
    det_mapping = load_deterministic_mapping()
    
    # Load family to attack (for ransomware coverage)
    family_to_attack = load_family_to_attack()
    
    # Identify ransomware families (families that use T1486 - Data Encrypted for Impact)
    # T1486 = Data Encrypted for Impact (primary ransomware indicator)
    # T1490 = Inhibit System Recovery (common ransomware technique)
    ransomware_families = []
    ransomware_techniques = set()
    for family, techniques in family_to_attack.items():
        if "T1486" in techniques or "T1490" in techniques:  # Key ransomware techniques
            ransomware_families.append(family)
            ransomware_techniques.update(techniques)
    
    # Build deterministic mapping table
    det_rows = []
    for technique_id, controls in det_mapping.items():
        for control_id in controls:
            det_rows.append({
                "mapping_type": "deterministic",
                "technique_id": technique_id,
                "control_id": control_id,
            })
    
    det_df = pd.DataFrame(det_rows)
    
    # Load learned mapping if available
    learned_df = load_learned_mapping()
    if learned_df is not None:
        learned_df["mapping_type"] = "learned"
        learned_df = learned_df.rename(columns={
            "technique_id": "technique_id",
            "control_id": "control_id"
        })
    else:
        # Create empty learned mapping for comparison
        learned_df = pd.DataFrame(columns=["mapping_type", "technique_id", "control_id"])
        print("Note: Learned mapping not found. Showing only deterministic mapping.")
    
    # Combine both mappings
    if not learned_df.empty:
        combined_df = pd.concat([det_df, learned_df], ignore_index=True)
    else:
        combined_df = det_df.copy()
    
    # Calculate ransomware coverage
    # Count how many ransomware families use each technique
    technique_to_ransomware_count = {}
    for family, techniques in family_to_attack.items():
        if family in ransomware_families:
            for tech in techniques:
                technique_to_ransomware_count[tech] = technique_to_ransomware_count.get(tech, 0) + 1
    
    # Add ransomware coverage metrics
    combined_df["ransomware_family_count"] = combined_df["technique_id"].map(
        technique_to_ransomware_count
    ).fillna(0).astype(int)
    
    # Calculate coverage per mapping type
    coverage_stats = {}
    for mapping_type in combined_df["mapping_type"].unique():
        subset = combined_df[combined_df["mapping_type"] == mapping_type]
        # Count unique ransomware families covered by techniques in this mapping
        covered_techniques = set(subset["technique_id"].unique())
        ransomware_covered = len([f for f, techs in family_to_attack.items() 
                                 if f in ransomware_families and any(t in covered_techniques for t in techs)])
        
        coverage_stats[mapping_type] = {
            "unique_techniques": subset["technique_id"].nunique(),
            "unique_controls": subset["control_id"].nunique(),
            "total_pairs": len(subset),
            "ransomware_family_count": ransomware_covered,
            "ransomware_technique_combinations": subset["ransomware_family_count"].sum(),
        }
    
    return combined_df, coverage_stats, len(ransomware_families)

# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("DAC Metric Lookup Table Viewer")
    print("=" * 80)
    print()
    
    # Build table
    dac_table, coverage_stats, total_ransomware_families = build_dac_table()
    
    # Display basic info
    print(f"Table Shape: {dac_table.shape}")
    print(f"Columns: {list(dac_table.columns)}")
    print()
    
    # Display first 20 rows
    print("=" * 80)
    print("First 20 Rows of DAC Metric Lookup Table")
    print("=" * 80)
    print()
    
    # Format for display
    display_cols = ["mapping_type", "technique_id", "control_id", "ransomware_family_count"]
    available_cols = [col for col in display_cols if col in dac_table.columns]
    
    print(dac_table[available_cols].head(20).to_string(index=False))
    print()
    
    # Display summary by mapping type
    print("=" * 80)
    print("DAC Coverage Summary by Mapping Type")
    print("=" * 80)
    print()
    
    summary_data = []
    for mapping_type, stats in coverage_stats.items():
        coverage_pct = (stats["ransomware_family_count"] / total_ransomware_families * 100) if total_ransomware_families > 0 else 0
        summary_data.append({
            "Mapping Type": mapping_type,
            "Unique Attack Techniques": stats["unique_techniques"],
            "Unique Defenses (D3FEND)": stats["unique_controls"],
            "Total Mapping Pairs": stats["total_pairs"],
            "Ransomware Families Covered": stats["ransomware_family_count"],
            "Ransomware Technique Combinations": stats["ransomware_technique_combinations"],
            "Coverage %": f"{coverage_pct:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print()
    
    # Plain English explanation
    print("=" * 80)
    print("Summary Explanation")
    print("=" * 80)
    print()
    
    for mapping_type, stats in coverage_stats.items():
        print(f"{mapping_type.capitalize()} Mapping:")
        print(f"  - Covers {stats['unique_techniques']} unique ATT&CK attack techniques")
        print(f"  - Maps to {stats['unique_controls']} unique D3FEND defense controls")
        print(f"  - Contains {stats['total_pairs']} total defense-attack mapping pairs")
        print(f"  - Covers {stats['ransomware_family_count']} out of {total_ransomware_families} ransomware families ({stats['ransomware_family_count']/total_ransomware_families*100:.1f}%)")
        print(f"  - Provides {stats['ransomware_technique_combinations']} ransomware family-technique mapping combinations")
        print()
    
    print(f"Total ransomware families identified: {total_ransomware_families}")
    print()
    
    # Save to CSV for reference (read-only, but useful for inspection)
    output_path = Path("dac_metric_lookup_table_view.csv")
    dac_table.to_csv(output_path, index=False)
    print(f"Full table saved to: {output_path} (for inspection only)")

