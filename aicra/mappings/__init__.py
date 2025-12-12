"""Mappings module for ATT&CK→D3FEND learned and deterministic mappings."""

from aicra.mappings.learned_ml_mapping import (
    generate_learned_mapping,
    load_deterministic_mapping,
    load_attack_catalog,
    load_defense_catalog,
    convert_to_dac_format,
)

from aicra.mappings.heuristic_mapping import (
    HeuristicMappingConfig,
    load_attack_techniques_with_descriptions,
    load_d3fend_controls_with_descriptions,
    build_heuristic_mapping,
)

__all__ = [
    "generate_learned_mapping",
    "load_deterministic_mapping",
    "load_attack_catalog",
    "load_defense_catalog",
    "convert_to_dac_format",
    "HeuristicMappingConfig",
    "load_attack_techniques_with_descriptions",
    "load_d3fend_controls_with_descriptions",
    "build_heuristic_mapping",
]

