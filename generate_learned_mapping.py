#!/usr/bin/env python3
"""
Generate learned embedding-based attack-defense mapping.
"""

import logging
from pathlib import Path
from aicra.mappings.embedding_learned_mapping import build_learned_embedding_mapping

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Find deterministic path
deterministic_path = Path("mappings_project/data/mappings/deterministic_attack_defense_lookup.csv")
if not deterministic_path.exists():
    deterministic_path = Path("data/mappings/deterministic_attack_defense_lookup.csv")
if not deterministic_path.exists():
    raise FileNotFoundError(f"Could not find deterministic mapping at {deterministic_path}")

print(f"Using deterministic mapping: {deterministic_path}")

# Output directory
output_dir = Path("data/mappings")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {output_dir}")

# Build learned mapping
print("\n" + "=" * 80)
print("GENERATING LEARNED EMBEDDING MAPPING")
print("=" * 80 + "\n")

learned_mapping_df = build_learned_embedding_mapping(
    deterministic_path=deterministic_path,
    output_dir=output_dir,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    top_k=3,
)

print(f"\n✓ Successfully generated learned embedding mapping with {len(learned_mapping_df)} pairs")

# Verify it's different from deterministic
import pandas as pd
det_df = pd.read_csv(deterministic_path)
if "is_correct" in det_df.columns:
    det_df = det_df[det_df["is_correct"] == 1]

det_pairs = set(zip(det_df["attack_id"], det_df["defense_id"]))
lrn_pairs = set(zip(learned_mapping_df["attack_id"], learned_mapping_df["defense_id"]))

intersection = det_pairs & lrn_pairs
only_in_det = det_pairs - lrn_pairs
only_in_learned = lrn_pairs - det_pairs

print("\n" + "=" * 80)
print("VERIFICATION: Comparing learned vs deterministic mapping")
print("=" * 80)
print(f"Deterministic pairs: {len(det_pairs)}")
print(f"Learned pairs: {len(lrn_pairs)}")
print(f"Intersection: {len(intersection)}")
print(f"Only in deterministic: {len(only_in_det)}")
print(f"Only in learned: {len(only_in_learned)}")

if len(only_in_det) == 0 and len(only_in_learned) == 0:
    print("\n" + "=" * 80)
    print("❌ CRITICAL ERROR: Learned mapping is IDENTICAL to deterministic!")
    print("=" * 80)
    print("This should NEVER happen - the learned mapping is generated PURELY from embeddings.")
    print("Possible causes:")
    print("  1. Embedding model producing identical similarity scores")
    print("  2. Code is filtering/intersecting learned pairs with deterministic")
    print("  3. Deterministic pairs happen to match top-k embedding similarities")
    print("=" * 80)
    raise RuntimeError(
        "Learned mapping is identical to deterministic mapping. "
        "This indicates a bug - learned mapping should be generated PURELY from embeddings, "
        "not copied from or filtered by deterministic pairs."
    )
else:
    print("\n✓ Learned mapping is different from deterministic (as expected)")
    overlap_pct = (len(intersection) / len(det_pairs) * 100) if len(det_pairs) > 0 else 0
    print(f"Overlap: {len(intersection)}/{len(det_pairs)} ({overlap_pct:.1f}%)")
    print(f"Only in learned: {len(only_in_learned)} pairs")
    print(f"Only in deterministic: {len(only_in_det)} pairs")

print("=" * 80)
print(f"\nLearned mapping saved to: {output_dir / 'learned_embedding_attack_defense_mapping.csv'}")









