#!/usr/bin/env python3
"""Add mapping_behavior field to H3 results JSON if missing."""

import json
import pandas as pd
from pathlib import Path

repo_root = Path(__file__).parent.parent
json_path = repo_root / "results" / "H3_full_evaluation" / "H3_full_results.json"

# Load JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Check if mapping_behavior already exists
if "mapping_behavior" in data:
    print("✓ mapping_behavior already exists in JSON")
    print(json.dumps(data["mapping_behavior"], indent=2))
    exit(0)

print("Computing mapping_behavior...")

# Load mappings
det_path = repo_root / "data" / "mappings" / "deterministic_attack_defense_lookup.csv"
learned_path = repo_root / "data" / "mappings" / "learned_mapping.csv"

det = pd.read_csv(det_path)
learned = pd.read_csv(learned_path)

# Handle column name variations
det_col = "attack_id" if "attack_id" in det.columns else "technique_id"
det_ctrl_col = "defense_id" if "defense_id" in det.columns else "control_id"

# Create pairs
det_pairs = set(zip(det[det_col].astype(str), det[det_ctrl_col].astype(str)))
learned_pairs = set(zip(learned["technique_id"].astype(str), learned["control_id"].astype(str)))

# Group by technique
det_by_tech = {}
for tech, ctrl in det_pairs:
    tech_str = str(tech)
    if tech_str not in det_by_tech:
        det_by_tech[tech_str] = set()
    det_by_tech[tech_str].add(str(ctrl))

learned_by_tech = {}
for tech, ctrl in learned_pairs:
    tech_str = str(tech)
    if tech_str not in learned_by_tech:
        learned_by_tech[tech_str] = set()
    learned_by_tech[tech_str].add(str(ctrl))

# Count techniques with extra learned controls
techniques_with_extra_learned_controls = []
techniques_with_only_ransomware_controls = []

for tech in learned_by_tech:
    learned_ctrls = learned_by_tech[tech]
    det_ctrls = det_by_tech.get(tech, set())
    
    # Check if learned has controls NOT in deterministic
    learned_only_ctrls = learned_ctrls - det_ctrls
    if len(learned_only_ctrls) > 0:
        techniques_with_extra_learned_controls.append(tech)
    
    # Check if learned controls are a subset of deterministic
    if det_ctrls and learned_ctrls.issubset(det_ctrls) and len(learned_ctrls) > 0:
        techniques_with_only_ransomware_controls.append(tech)

learned_only_pairs = learned_pairs - det_pairs

# Create mapping_behavior
mapping_behavior = {
    "learned_is_broader": len(learned_pairs) > len(det_pairs) and len(learned_only_pairs) > 0,
    "learned_pairs_count": len(learned_pairs),
    "deterministic_pairs_count": len(det_pairs),
    "learned_only_pairs_count": len(learned_only_pairs),
    "techniques_with_extra_learned_controls": len(techniques_with_extra_learned_controls),
    "techniques_with_only_ransomware_controls": len(techniques_with_only_ransomware_controls),
    "total_techniques_in_learned": len(learned_by_tech),
    "total_techniques_in_deterministic": len(det_by_tech),
}

# Add to JSON (insert after mapping_overlap)
if "mapping_overlap" in data:
    # Create new dict with mapping_behavior inserted
    new_data = {}
    for key, value in data.items():
        new_data[key] = value
        if key == "mapping_overlap":
            new_data["mapping_behavior"] = mapping_behavior
    data = new_data
else:
    data["mapping_behavior"] = mapping_behavior

# Save updated JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("✓ Added mapping_behavior to JSON")
print("\nmapping_behavior:")
print(json.dumps(mapping_behavior, indent=2))
