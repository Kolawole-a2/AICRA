"""Metrics module for AICRA evaluation metrics."""

from aicra.metrics.dac import (
    compute_coverage,
    compute_dac,
    compute_dac_between_mappings,
    compute_dac_per_attack,
)
from aicra.metrics.dac import (
    load_deterministic_mapping as load_dac_deterministic_mapping,
)
from aicra.metrics.dac import load_learned_mapping as load_dac_learned_mapping
from aicra.metrics.dac_embedding_eval import (
    compute_dac_per_attack as compute_dac_per_attack_embedding,
)
from aicra.metrics.dac_embedding_eval import (
    evaluate_dac_embedding,
)

__all__ = [
    "compute_dac",
    "compute_dac_between_mappings",
    "compute_dac_per_attack",
    "compute_coverage",
    "load_dac_deterministic_mapping",
    "load_dac_learned_mapping",
    "evaluate_dac_embedding",
    "compute_dac_per_attack_embedding",
]
