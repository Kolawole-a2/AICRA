"""
Learned ATT&CK→D3FEND Mapping using Embeddings (Method B).

This module implements a learned mapping pipeline using sentence-transformers
to embed ATT&CK technique names and D3FEND defense names, then computes cosine
similarity to find the most similar defenses for each attack.

IMPORTANT CONSTRAINT:
- The learned mapping does NOT use deterministic ATTACK-DEFENSE pairs as supervision.
- It ONLY uses text fields (attack_name, defense_name) to compute embeddings.
- No model is trained to predict defense_id from attack_id or from deterministic pairs.
- The learned mapping is based PURELY on semantic similarity in embedding space,
  not on copying ontology links.

For each ATTACK technique, it selects the top-k (default k=3) most similar
defenses based on cosine similarity of their embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)


def load_deterministic_lookup(path: Path) -> pd.DataFrame:
    """
    Load deterministic ATT&CK→D3FEND lookup table from CSV.

    NOTE: This function is used ONLY to extract unique (attack_id, attack_name)
    and (defense_id, defense_name) pairs. The deterministic ATTACK-DEFENSE pairs
    themselves are NOT used as supervision or labels for the learned mapping.

    Args:
        path: Path to deterministic_attack_defense_lookup.csv

    Returns:
        DataFrame with columns: attack_id, attack_name, defense_id, defense_name
        (duplicates dropped)
    """
    LOGGER.info(f"Loading deterministic mapping from {path}")

    if not path.exists():
        raise FileNotFoundError(f"Deterministic lookup not found at {path}")

    df = pd.read_csv(path)

    # Ensure required columns exist
    required_cols = ["attack_id", "attack_name", "defense_id", "defense_name"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter to correct mappings only if is_correct column exists
    if "is_correct" in df.columns:
        df = df[df["is_correct"] == 1]
        LOGGER.info(f"Filtered to {len(df)} correct mappings")

    # Drop duplicates
    df = df.drop_duplicates(subset=required_cols)

    LOGGER.info(f"Loaded {len(df)} deterministic mappings (after dropping duplicates)")
    LOGGER.info(
        f"Unique attacks: {df['attack_id'].nunique()}, Unique defenses: {df['defense_id'].nunique()}"
    )

    return df


def build_text_tables(det_lookup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build text tables for attacks and defenses from deterministic lookup.

    Args:
        det_lookup: DataFrame with attack_id, attack_name, defense_id, defense_name columns

    Returns:
        Tuple of (attack_df, defense_df):
        - attack_df: columns attack_id, attack_text (where attack_text = attack_name)
        - defense_df: columns defense_id, defense_text (where defense_text = defense_name)
    """
    LOGGER.info("Building text tables from deterministic lookup")

    # Extract unique attacks: attack_id, attack_text = attack_name
    attack_df = det_lookup[["attack_id", "attack_name"]].drop_duplicates()
    attack_df = attack_df.rename(columns={"attack_name": "attack_text"})
    attack_df = attack_df.reset_index(drop=True)

    # Extract unique defenses: defense_id, defense_text = defense_name
    defense_df = det_lookup[["defense_id", "defense_name"]].drop_duplicates()
    defense_df = defense_df.rename(columns={"defense_name": "defense_text"})
    defense_df = defense_df.reset_index(drop=True)

    LOGGER.info(f"Extracted {len(attack_df)} unique attacks")
    LOGGER.info(f"Extracted {len(defense_df)} unique defenses")

    return attack_df, defense_df


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SentenceTransformer:
    """
    Load SentenceTransformer model.

    Args:
        model_name: Name of sentence-transformers model (default: "sentence-transformers/all-MiniLM-L6-v2")

    Returns:
        SentenceTransformer model instance
    """
    LOGGER.info(f"Loading sentence-transformers model: {model_name}")
    model = SentenceTransformer(model_name)
    LOGGER.info(
        f"Model loaded: {model_name} (embedding dimension: {model.get_sentence_embedding_dimension()})"
    )
    return model


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Embed text strings using sentence-transformers model.

    Args:
        model: SentenceTransformer model instance
        texts: List of text strings to embed

    Returns:
        NumPy array of shape (n_texts, embedding_dim) with embeddings
    """
    LOGGER.info(
        f"Embedding {len(texts)} texts using {model.get_sentence_embedding_dimension()}-dimensional embeddings"
    )

    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    LOGGER.info(f"Generated embeddings with shape {embeddings.shape}")

    return embeddings


def compute_cosine_similarity_matrix(
    attack_embeddings: np.ndarray,
    defense_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between attack and defense embeddings.

    Args:
        attack_embeddings: Array of shape (n_attacks, embedding_dim)
        defense_embeddings: Array of shape (n_defenses, embedding_dim)

    Returns:
        Array of shape (n_attacks, n_defenses) with cosine similarity scores
    """
    LOGGER.info(
        f"Computing cosine similarity matrix: {attack_embeddings.shape[0]} attacks × {defense_embeddings.shape[0]} defenses"
    )

    similarity_matrix = cosine_similarity(attack_embeddings, defense_embeddings)

    LOGGER.info(f"Computed similarity matrix with shape {similarity_matrix.shape}")
    LOGGER.info(
        f"Similarity score range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]"
    )

    return similarity_matrix


def generate_learned_mapping(
    attack_df: pd.DataFrame,
    defense_df: pd.DataFrame,
    similarity_matrix: np.ndarray,
    top_k: int = 3,
    deterministic_pairs: set = None,
) -> pd.DataFrame:
    """
    Generate learned mapping by selecting top-k most similar defenses per attack.

    CRITICAL: This function generates pairs PURELY from embedding similarity.
    It does NOT filter, intersect, or use deterministic pairs in any way.

    Args:
        attack_df: DataFrame with attack_id column (from build_text_tables)
        defense_df: DataFrame with defense_id column (from build_text_tables)
        similarity_matrix: Array of shape (n_attacks, n_defenses) with similarity scores
        top_k: Number of top defenses to select per attack (default: 3)
        deterministic_pairs: Optional set of (attack_id, defense_id) tuples for verification only

    Returns:
        DataFrame with columns: attack_id, defense_id, similarity_score, rank, method
    """
    LOGGER.info(f"Generating learned mapping with top_k={top_k}")
    LOGGER.info(
        "CRITICAL: Generating pairs PURELY from embedding similarity - NO filtering by deterministic pairs"
    )

    mapping_rows = []

    for i, (_, attack_row) in enumerate(attack_df.iterrows()):
        attack_id = attack_row["attack_id"]

        # Get similarity scores for this attack
        attack_similarities = similarity_matrix[i]

        # Get top-k indices (sorted descending by similarity)
        # CRITICAL: We select based ONLY on similarity scores, NOT on whether pairs exist in deterministic
        top_k_indices = np.argsort(attack_similarities)[::-1][:top_k]

        # Build rows for this attack
        for rank, defense_idx in enumerate(top_k_indices, start=1):
            defense_id = defense_df.iloc[defense_idx]["defense_id"]
            similarity_score = float(attack_similarities[defense_idx])

            mapping_rows.append(
                {
                    "attack_id": attack_id,
                    "defense_id": defense_id,
                    "similarity_score": similarity_score,
                    "rank": rank,
                    "method": "embedding",
                }
            )

    learned_mapping_df = pd.DataFrame(mapping_rows)

    LOGGER.info(
        f"Generated learned mapping with {len(learned_mapping_df)} attack-defense pairs"
    )
    LOGGER.info(f"Unique attacks: {learned_mapping_df['attack_id'].nunique()}")
    LOGGER.info(f"Unique defenses: {learned_mapping_df['defense_id'].nunique()}")
    LOGGER.info(
        f"Average similarity score: {learned_mapping_df['similarity_score'].mean():.4f}"
    )

    # CRITICAL CHECK: Verify learned mapping is different from deterministic
    if deterministic_pairs is not None:
        learned_pairs = set(
            zip(
                learned_mapping_df["attack_id"],
                learned_mapping_df["defense_id"],
                strict=False,
            )
        )
        if learned_pairs == deterministic_pairs:
            LOGGER.error("=" * 80)
            LOGGER.error(
                "CRITICAL ERROR: Learned mapping is IDENTICAL to deterministic!"
            )
            LOGGER.error("=" * 80)
            LOGGER.error(f"Learned pairs: {len(learned_pairs)}")
            LOGGER.error(f"Deterministic pairs: {len(deterministic_pairs)}")
            LOGGER.error(
                "This should NEVER happen - learned mapping is based on embeddings, not deterministic pairs."
            )
            LOGGER.error("Possible causes:")
            LOGGER.error(
                "  1. Embedding model is producing identical similarity scores"
            )
            LOGGER.error(
                "  2. Code is filtering/intersecting learned pairs with deterministic"
            )
            LOGGER.error(
                "  3. Deterministic pairs happen to match top-k embedding similarities"
            )
            LOGGER.error("=" * 80)
            raise RuntimeError(
                "Learned mapping is identical to deterministic mapping. "
                "This indicates a bug - learned mapping should be generated PURELY from embeddings, "
                "not copied from or filtered by deterministic pairs. "
                f"Learned: {len(learned_pairs)} pairs, Deterministic: {len(deterministic_pairs)} pairs."
            )
        else:
            intersection = learned_pairs & deterministic_pairs
            only_in_learned = learned_pairs - deterministic_pairs
            only_in_det = deterministic_pairs - learned_pairs
            LOGGER.info(
                "✓ Learned mapping is different from deterministic (as expected)"
            )
            LOGGER.info(f"  Intersection: {len(intersection)} pairs")
            LOGGER.info(f"  Only in learned: {len(only_in_learned)} pairs")
            LOGGER.info(f"  Only in deterministic: {len(only_in_det)} pairs")

    return learned_mapping_df


def save_learned_mapping(
    learned_mapping_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save learned mapping to CSV and Parquet formats.

    Args:
        learned_mapping_df: DataFrame with learned mapping
        output_dir: Directory to save output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "learned_embedding_attack_defense_mapping.csv"
    parquet_path = output_dir / "learned_embedding_attack_defense_mapping.parquet"

    # Save to CSV
    learned_mapping_df.to_csv(csv_path, index=False)
    LOGGER.info(f"Saved learned mapping CSV to {csv_path}")

    # Save to Parquet
    learned_mapping_df.to_parquet(parquet_path, index=False)
    LOGGER.info(f"Saved learned mapping Parquet to {parquet_path}")

    # Log summary statistics
    num_attacks = learned_mapping_df["attack_id"].nunique()
    num_defenses = learned_mapping_df["defense_id"].nunique()
    num_pairs = len(learned_mapping_df)
    avg_similarity = learned_mapping_df["similarity_score"].mean()

    LOGGER.info("Learned mapping summary:")
    LOGGER.info(f"  - Unique attacks: {num_attacks}")
    LOGGER.info(f"  - Unique defenses: {num_defenses}")
    LOGGER.info(f"  - Total pairs: {num_pairs}")
    LOGGER.info(f"  - Average similarity: {avg_similarity:.4f}")


def save_learned_mapping_to_paths(
    learned_mapping_df: pd.DataFrame,
    out_csv: Path,
    out_parquet: Path,
) -> None:
    """
    Save learned mapping to specified CSV and Parquet paths.

    Args:
        learned_mapping_df: DataFrame with learned mapping
        out_csv: Path to output CSV file
        out_parquet: Path to output Parquet file
    """
    # Ensure parent directories exist
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    learned_mapping_df.to_csv(out_csv, index=False)
    LOGGER.info(f"Saved learned mapping CSV to {out_csv}")

    # Save to Parquet
    learned_mapping_df.to_parquet(out_parquet, index=False)
    LOGGER.info(f"Saved learned mapping Parquet to {out_parquet}")

    # Log summary statistics
    num_attacks = learned_mapping_df["attack_id"].nunique()
    num_defenses = learned_mapping_df["defense_id"].nunique()
    num_pairs = len(learned_mapping_df)
    avg_similarity = learned_mapping_df["similarity_score"].mean()

    LOGGER.info("Learned mapping summary:")
    LOGGER.info(f"  - Unique attacks: {num_attacks}")
    LOGGER.info(f"  - Unique defenses: {num_defenses}")
    LOGGER.info(f"  - Total pairs: {num_pairs}")
    LOGGER.info(f"  - Average similarity: {avg_similarity:.4f}")


def build_embedding_learned_mapping(
    det_lookup_path: Path,
    out_csv: Path,
    out_parquet: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Build learned ATT&CK→D3FEND mapping using embeddings.

    This function:
    1. Loads deterministic lookup table (ONLY to extract unique attack/defense names)
    2. Builds text tables (attack_text, defense_text)
    3. Embeds attack and defense texts using sentence-transformers
    4. Computes cosine similarity between all attack-defense pairs
    5. Generates learned mapping by selecting top-k defenses per attack
    6. Saves results to specified CSV and Parquet paths
    7. Raises RuntimeError if learned mapping is identical to deterministic

    IMPORTANT: The deterministic ATTACK-DEFENSE pairs are NOT used as supervision.
    The learned mapping is based PURELY on semantic similarity of text embeddings.

    Args:
        det_lookup_path: Path to deterministic_attack_defense_lookup.csv
                        (used only to extract unique attack/defense names)
        out_csv: Path to output CSV file
        out_parquet: Path to output Parquet file
        model_name: Name of sentence-transformers model (default: "sentence-transformers/all-MiniLM-L6-v2")
        top_k: Number of top defenses to select per attack (default: 3)

    Returns:
        DataFrame with learned mapping: attack_id, defense_id, similarity_score, rank, method

    Raises:
        RuntimeError: If learned mapping is identical to deterministic mapping
    """
    LOGGER.info("=" * 80)
    LOGGER.info("Building Learned ATT&CK→D3FEND Mapping using Embeddings")
    LOGGER.info("=" * 80)

    # Step 1: Load deterministic lookup
    df_deterministic = load_deterministic_lookup(det_lookup_path)

    # Step 2: Build text tables (attack_df with attack_text, defense_df with defense_text)
    attack_df, defense_df = build_text_tables(df_deterministic)

    # Step 3: Load sentence-transformers model
    model = get_embedding_model(model_name)

    # Step 4: Embed attack texts
    attack_texts = attack_df["attack_text"].fillna("").astype(str).tolist()
    attack_embeddings = embed_texts(model, attack_texts)

    # Step 5: Embed defense texts
    defense_texts = defense_df["defense_text"].fillna("").astype(str).tolist()
    defense_embeddings = embed_texts(model, defense_texts)

    # Step 6: Compute cosine similarity matrix
    similarity_matrix = compute_cosine_similarity_matrix(
        attack_embeddings, defense_embeddings
    )

    # Step 7: Extract deterministic pairs for verification ONLY (NOT for filtering)
    # CRITICAL: We extract deterministic pairs ONLY to verify learned mapping is different
    # We do NOT use them to filter, intersect, or modify the learned mapping
    # The learned mapping is generated PURELY from embedding similarity scores
    deterministic_pairs_for_check = set(
        zip(df_deterministic["attack_id"], df_deterministic["defense_id"], strict=False)
    )
    LOGGER.info(
        f"Extracted {len(deterministic_pairs_for_check)} deterministic pairs for verification ONLY (NOT for filtering)"
    )
    LOGGER.info(
        "CRITICAL: Learned mapping will be generated PURELY from embedding similarity, NOT filtered by deterministic pairs"
    )

    # Step 8: Generate learned mapping PURELY from embeddings
    # CRITICAL: deterministic_pairs_for_check is passed ONLY for verification, NOT for filtering
    # The generate_learned_mapping function selects top-k defenses based ONLY on similarity scores
    # It does NOT check if pairs exist in deterministic, does NOT filter, does NOT intersect
    learned_mapping_df = generate_learned_mapping(
        attack_df,
        defense_df,
        similarity_matrix,
        top_k=top_k,
        deterministic_pairs=deterministic_pairs_for_check,  # For verification only - raises RuntimeError if identical
    )

    # Step 9: Additional verification using DataFrame merge operations
    LOGGER.info("=" * 80)
    LOGGER.info(
        "VERIFICATION: Comparing learned vs deterministic using DataFrame merge"
    )
    LOGGER.info("=" * 80)

    # Load deterministic pairs as DataFrame
    det_df = pd.read_csv(det_lookup_path)
    if "is_correct" in det_df.columns:
        det_df = det_df[det_df["is_correct"] == 1]
    det_pairs = det_df[["attack_id", "defense_id"]].drop_duplicates()

    # Extract learned pairs as DataFrame
    learned_pairs = learned_mapping_df[["attack_id", "defense_id"]].drop_duplicates()

    # Compute intersection using DataFrame merge
    intersection = det_pairs.merge(
        learned_pairs, on=["attack_id", "defense_id"], how="inner"
    )

    # Compute "only in deterministic" using left merge with indicator
    only_in_det = det_pairs.merge(
        learned_pairs, on=["attack_id", "defense_id"], how="left", indicator=True
    )
    only_in_det = only_in_det[only_in_det["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )

    # Compute "only in learned" using left merge with indicator
    only_in_learned = learned_pairs.merge(
        det_pairs, on=["attack_id", "defense_id"], how="left", indicator=True
    )
    only_in_learned = only_in_learned[only_in_learned["_merge"] == "left_only"].drop(
        columns=["_merge"]
    )

    LOGGER.info("det_pairs: %d", len(det_pairs))
    LOGGER.info("learned_pairs: %d", len(learned_pairs))
    LOGGER.info("intersection: %d", len(intersection))
    LOGGER.info("only_in_det: %d", len(only_in_det))
    LOGGER.info("only_in_learned: %d", len(only_in_learned))

    # CRITICAL CHECK: If both only_in_det and only_in_learned are 0, mappings are identical
    if len(only_in_det) == 0 and len(only_in_learned) == 0:
        LOGGER.error("=" * 80)
        LOGGER.error("CRITICAL ERROR: Learned mapping is IDENTICAL to deterministic!")
        LOGGER.error("=" * 80)
        LOGGER.error(f"Deterministic pairs: {len(det_pairs)}")
        LOGGER.error(f"Learned pairs: {len(learned_pairs)}")
        LOGGER.error(f"Intersection: {len(intersection)}")
        LOGGER.error(f"Only in deterministic: {len(only_in_det)}")
        LOGGER.error(f"Only in learned: {len(only_in_learned)}")
        LOGGER.error("=" * 80)
        raise RuntimeError(
            "ERROR: learned_embedding_attack_defense_mapping is IDENTICAL to the deterministic lookup. "
            "The learned mapping must be derived from embedding similarity, not copied from deterministic."
        )
    else:
        LOGGER.info("✓ Learned mapping is different from deterministic (as expected)")
        overlap_pct = (
            (len(intersection) / len(det_pairs) * 100) if len(det_pairs) > 0 else 0
        )
        LOGGER.info(
            f"Overlap: {len(intersection)}/{len(det_pairs)} ({overlap_pct:.1f}%)"
        )
        LOGGER.info(f"Only in learned: {len(only_in_learned)} pairs")
        LOGGER.info(f"Only in deterministic: {len(only_in_det)} pairs")

    LOGGER.info("=" * 80)

    # Step 10: Save learned mapping to specified paths (only if verification passed)
    save_learned_mapping_to_paths(learned_mapping_df, out_csv, out_parquet)

    LOGGER.info("=" * 80)
    LOGGER.info("Learned embedding mapping pipeline completed successfully")
    LOGGER.info("=" * 80)

    return learned_mapping_df


def build_learned_embedding_mapping(
    deterministic_path: Path,
    output_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Build learned ATT&CK→D3FEND mapping using embeddings (legacy wrapper).

    This is a convenience wrapper around build_embedding_learned_mapping that uses
    an output directory instead of explicit paths.

    Args:
        deterministic_path: Path to deterministic_attack_defense_lookup.csv
        output_dir: Directory to save output files
        model_name: Name of sentence-transformers model
        top_k: Number of top defenses to select per attack

    Returns:
        DataFrame with learned mapping
    """
    out_csv = output_dir / "learned_embedding_attack_defense_mapping.csv"
    out_parquet = output_dir / "learned_embedding_attack_defense_mapping.parquet"

    return build_embedding_learned_mapping(
        det_lookup_path=deterministic_path,
        out_csv=out_csv,
        out_parquet=out_parquet,
        model_name=model_name,
        top_k=top_k,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    det_path = Path("data/mappings/deterministic_attack_defense_lookup.csv")
    out_csv = Path("data/mappings/learned_embedding_attack_defense_mapping.csv")
    out_parquet = Path("data/mappings/learned_embedding_attack_defense_mapping.parquet")

    build_embedding_learned_mapping(
        det_lookup_path=det_path,
        out_csv=out_csv,
        out_parquet=out_parquet,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        top_k=3,
    )

