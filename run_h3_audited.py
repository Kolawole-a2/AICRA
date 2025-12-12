#!/usr/bin/env python3
"""Run H3 evaluation with audited code."""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from aicra.experiments.h3_evaluation import run_h3_evaluation

print("=" * 80)
print("H3 Evaluation - Audited Implementation")
print("=" * 80)

repo_root = Path(".")

result = run_h3_evaluation(
    splits_config_path=repo_root / "config/h3_splits.yaml",
    det_mapping_path=repo_root / "data/mappings/deterministic_lookup.csv",
    learned_mapping_path=repo_root / "data/mappings/learned_mapping.csv",
    ref_pairs_path=repo_root / "d3fend_reference_pairs.csv",
    output_dir=repo_root / "results/H3_full_evaluation",
    repo_root=repo_root,
)

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

# Show split evaluation summary
eval_summary = result.get('splits_evaluation_summary', {})
print(f"\n📊 SPLIT EVALUATION SUMMARY:")
print(f"   Total splits in config: {eval_summary.get('total_splits_in_config', len(result.get('splits_config', {})))}")
print(f"   Successfully evaluated: {eval_summary.get('successfully_evaluated', len(result.get('splits_evaluated', [])))}")
print(f"   Failed/Skipped: {eval_summary.get('failed_or_skipped', len(result.get('splits_skipped', [])))}")
print(f"\n   ✅ Evaluated splits: {result['splits_evaluated']}")
if result.get('splits_skipped'):
    print(f"   ❌ Skipped splits: {result['splits_skipped']}")

# Show per-split summary
print(f"\n📈 PER-SPLIT RESULTS:")
for split_result in result.get('per_split_results', []):
    split_name = split_result.get('split', 'unknown')
    n_samples = split_result.get('n_samples', 0)
    n_techniques = split_result.get('n_techniques', 0)
    det_dac_int = split_result.get('deterministic', {}).get('mapping_metrics', {}).get('dac_internal_%', 100.0)
    learned_dac_int = split_result.get('learned', {}).get('mapping_metrics', {}).get('dac_internal_%', 0.0)
    det_cov = split_result.get('deterministic', {}).get('mapping_metrics', {}).get('coverage_%', 0.0)
    learned_cov = split_result.get('learned', {}).get('mapping_metrics', {}).get('coverage_%', 0.0)
    print(f"   {split_name:15s}: {n_samples:6d} samples, {n_techniques:2d} techniques | "
          f"DAC_int: Det={det_dac_int:6.2f}%, Lrn={learned_dac_int:6.2f}% | "
          f"Cov: Det={det_cov:6.2f}%, Lrn={learned_cov:6.2f}%")

print(f"\n📋 MAPPING METADATA:")
print(f"\n   Deterministic Mapping:")
print(f"     Pairs: {result['deterministic_mapping_info']['n_pairs']}")
print(f"     Techniques: {result['deterministic_mapping_info']['n_unique_attack_techniques']}")
print(f"     Controls: {result['deterministic_mapping_info']['n_unique_defense_controls']}")
print(f"\n   Learned Mapping:")
print(f"     Pairs: {result['learned_mapping_info']['n_pairs']}")
print(f"     Techniques: {result['learned_mapping_info']['n_unique_attack_techniques']}")
print(f"     Controls: {result['learned_mapping_info']['n_unique_defense_controls']}")
print(f"\n   Reference Pairs:")
print(f"     Pairs: {result['reference_pairs_info']['n_pairs']}")
print(f"     Techniques: {result['reference_pairs_info']['n_unique_attack_techniques']}")
print(f"     Controls: {result['reference_pairs_info']['n_unique_defense_controls']}")

print("\n" + "=" * 80)
print("Results saved to: results/H3_full_evaluation/")
print("=" * 80)
