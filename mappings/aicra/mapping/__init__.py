"""Mapping modules for ATT&CK→D3FEND deterministic and heuristic mappings."""

from aicra.mapping.deterministic_lookup import (
    build_deterministic_lookup,
    save_deterministic_lookup,
    load_attack_catalog,
    load_defense_catalog,
    load_attack_defense_edges,
)

from aicra.mapping.heuristic_mapping import (
    HeuristicMappingConfig,
    build_heuristic_mapping,
    load_attack_techniques_with_descriptions,
    load_d3fend_controls_with_descriptions,
)

__all__ = [
    # Deterministic lookup
    "build_deterministic_lookup",
    "save_deterministic_lookup",
    "load_attack_catalog",
    "load_defense_catalog",
    "load_attack_defense_edges",
    # Heuristic mapping
    "HeuristicMappingConfig",
    "build_heuristic_mapping",
    "load_attack_techniques_with_descriptions",
    "load_d3fend_controls_with_descriptions",
]

