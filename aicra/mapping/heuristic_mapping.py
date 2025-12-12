"""
Heuristic ATT&CK→D3FEND Mapping using Text Similarity.

This module implements a text-similarity-based heuristic mapping using ATT&CK technique
descriptions and D3FEND control descriptions. It uses sentence transformers to compute
semantic similarity between attack and defense descriptions, then selects the top-k most
similar defenses for each attack technique.

This heuristic mapping serves as a baseline for H3 experiments, allowing comparison
with the deterministic mapping in terms of:
- Mapping coverage (%)
- Defense–Attack Consistency (DAC %)
- Δ precision (actionable positives)
- Variance reduction in risk scores

Output: data/mappings/learned_mapping.csv with columns:
- technique_id
- control_id
- similarity_score
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Set

import logging
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False

try:
    import stix2
except ImportError:
    stix2 = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

LOGGER = logging.getLogger(__name__)


@dataclass
class HeuristicMappingConfig:
    """
    Configuration for heuristic mapping.
    
    For H3, the heuristic mapping is designed to be:
    - Generic and broad (not ransomware-specific)
    - Uses ALL (or almost all) D3FEND controls
    - Noisy and less aligned with ransomware defense
    - Expected to perform worse than deterministic mapping
    
    Default parameters are set to create a broad, generic mapping:
    - top_k=10: Maps each technique to many controls (broad coverage)
    - min_similarity=0.25: Low threshold to include more controls (noisier)
    """
    top_k: int = 10  # number of controls per technique (broad, generic mapping)
    min_similarity: float = 0.25  # low threshold for broad, noisy mapping
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    seed: int = 42


def set_seeds(seed: int) -> None:
    """Set random seeds for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_attack_techniques_with_descriptions(
    stix_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load ATT&CK techniques with descriptions.
    
    Tries to load from CSV first (if provided), otherwise extracts from STIX JSON.
    If CSV doesn't have descriptions, falls back to STIX JSON.
    
    Args:
        stix_path: Path to enterprise-attack.json STIX bundle
        csv_path: Path to attack_techniques.csv (expected columns: technique_id, name, description)
        
    Returns:
        DataFrame with columns: technique_id, name, description
    """
    LOGGER.info("Loading ATT&CK techniques with descriptions...")
    
    # Try CSV first if provided
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        # Normalize column names
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if "technique_id" in col_lower or ("attack_id" in col_lower and "technique" in col_lower):
                col_mapping[col] = "technique_id"
            elif col_lower == "attack_id":
                col_mapping[col] = "technique_id"
            elif "name" in col_lower:
                col_mapping[col] = "name"
            elif "description" in col_lower:
                col_mapping[col] = "description"
        
        if col_mapping:
            df = df.rename(columns=col_mapping)
        
        if "description" in df.columns and df["description"].notna().any():
            required_cols = ["technique_id", "name", "description"]
            if all(col in df.columns for col in required_cols):
                df = df[required_cols].copy()
                df = df.dropna(subset=["technique_id"])
                LOGGER.info(f"Loaded {len(df)} ATT&CK techniques from CSV with descriptions")
                return df
    
    # Fall back to STIX JSON
    if stix_path and stix_path.exists():
        if stix2 is None:
            raise ImportError(
                "stix2 is required to load ATT&CK techniques from STIX JSON. "
                "Install with: pip install stix2"
            )
        
        LOGGER.info(f"Extracting ATT&CK techniques from STIX bundle: {stix_path}")
        
        with stix_path.open("r", encoding="utf-8") as f:
            stix_data = json.load(f)
        
        memory_store = stix2.MemoryStore(stix_data=stix_data.get("objects", []))
        attack_patterns = memory_store.query([stix2.Filter("type", "=", "attack-pattern")])
        
        data = []
        for pattern in attack_patterns:
            attack_id = None
            for ref in pattern.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    attack_id = ref.get("external_id")
                    break
            
            if attack_id:
                description = pattern.get("description", "")
                # Clean description (remove markdown, extra whitespace)
                if description:
                    description = " ".join(description.split())
                
                data.append({
                    "technique_id": attack_id,
                    "name": pattern.get("name", ""),
                    "description": description or "",
                })
        
        df = pd.DataFrame(data)
        df = df.dropna(subset=["technique_id"])
        LOGGER.info(f"Extracted {len(df)} ATT&CK techniques from STIX with descriptions")
        return df
    
    # Try to load from expected ontology CSV location
    ontology_path = Path("data/ontology/attack_techniques.csv")
    if ontology_path.exists():
        df = pd.read_csv(ontology_path)
        required_cols = ["technique_id", "name", "description"]
        if all(col in df.columns for col in required_cols):
            df = df[required_cols].copy()
            df = df.dropna(subset=["technique_id"])
            LOGGER.info(f"Loaded {len(df)} ATT&CK techniques from ontology CSV")
            return df
    
    raise FileNotFoundError(
        "Could not find ATT&CK techniques with descriptions. "
        "Provide either stix_path or csv_path, or create data/ontology/attack_techniques.csv"
    )


def load_d3fend_controls_with_descriptions(
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load D3FEND controls with descriptions.
    
    Tries to load from CSV (with "Definition" column), or from expected ontology location.
    
    Args:
        csv_path: Path to d3fend_controls.csv (expected columns: control_id, name, description)
        
    Returns:
        DataFrame with columns: control_id, name, description
    """
    LOGGER.info("Loading D3FEND controls with descriptions...")
    
    # Try provided CSV path
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        
        # Normalize column names
        col_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if col.upper() == "ID" or col_lower == "id" or ("control_id" in col_lower):
                col_mapping[col] = "control_id"
            elif "name" in col_lower or ("technique" in col_lower and "d3fend" in col_lower):
                col_mapping[col] = "name"
            elif "description" in col_lower or col == "Definition":
                col_mapping[col] = "description"
        
        if col_mapping:
            df = df.rename(columns=col_mapping)
        
        # If we have control_id but not name, use control_id as name
        if "control_id" in df.columns and "name" not in df.columns:
            df["name"] = df["control_id"]
        
        # If we have control_id but not description, use empty string
        if "control_id" in df.columns and "description" not in df.columns:
            df["description"] = ""
        
        if "control_id" in df.columns:
            required_cols = ["control_id", "name", "description"]
            df = df[required_cols].copy()
            df = df.dropna(subset=["control_id"])
            LOGGER.info(f"Loaded {len(df)} D3FEND controls from CSV")
            return df
    
    # Try expected ontology CSV location
    ontology_path = Path("data/ontology/d3fend_controls.csv")
    if ontology_path.exists():
        df = pd.read_csv(ontology_path)
        required_cols = ["control_id", "name", "description"]
        if all(col in df.columns for col in required_cols):
            df = df[required_cols].copy()
            df = df.dropna(subset=["control_id"])
            LOGGER.info(f"Loaded {len(df)} D3FEND controls from ontology CSV")
            return df
    
    raise FileNotFoundError(
        "Could not find D3FEND controls with descriptions. "
        "Provide csv_path, or create data/ontology/d3fend_controls.csv"
    )


def build_heuristic_mapping(
    attack_path: str | Path,
    d3fend_path: str | Path,
    config: HeuristicMappingConfig,
    check_diversity: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[np.ndarray]]:
    """
    Build heuristic ATT&CK→D3FEND mapping using text similarity.
    
    Args:
        attack_path: Path to ATT&CK techniques CSV or STIX JSON
        d3fend_path: Path to D3FEND controls CSV
        config: Configuration for heuristic mapping
        
    Returns:
        Tuple of:
        - DataFrame with columns: technique_id, control_id, similarity_score
        - attack_df (for diversity checking)
        - defense_df (for diversity checking)
        - similarity_matrix (for diversity checking)
    """
    # Set seeds for deterministic behavior
    set_seeds(config.seed)
    
    LOGGER.info(f"Building heuristic mapping with config: top_k={config.top_k}, "
                f"min_similarity={config.min_similarity}, model={config.model_name}")
    
    # Load data
    attack_path_obj = Path(attack_path)
    if attack_path_obj.suffix == ".json":
        attack_df = load_attack_techniques_with_descriptions(stix_path=attack_path_obj)
    else:
        attack_df = load_attack_techniques_with_descriptions(csv_path=attack_path_obj)
    
    defense_df = load_d3fend_controls_with_descriptions(csv_path=Path(d3fend_path))
    
    LOGGER.info(f"Loaded {len(attack_df)} ATT&CK techniques")
    LOGGER.info(f"Loaded {len(defense_df)} D3FEND controls")
    
    # Filter out rows with missing descriptions
    attack_df = attack_df[
        attack_df["description"].notna() & (attack_df["description"] != "")
    ].copy()
    defense_df = defense_df[
        defense_df["description"].notna() & (defense_df["description"] != "")
    ].copy()
    
    LOGGER.info(f"Using {len(attack_df)} attacks and {len(defense_df)} controls with descriptions")
    
    if len(attack_df) == 0 or len(defense_df) == 0:
        raise ValueError("Need at least one attack and one control with descriptions")
    
    # Build standard text format: name + ". " + description
    attack_texts = []
    for _, row in attack_df.iterrows():
        text = f"{row['name']}. {row['description']}"
        attack_texts.append(text)
    
    control_texts = []
    for _, row in defense_df.iterrows():
        text = f"{row['name']}. {row['description']}"
        control_texts.append(text)
    
    # Compute embeddings and similarity
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        LOGGER.info(f"Using sentence-transformers model: {config.model_name}")
        model = SentenceTransformer(config.model_name)
        
        # Compute embeddings in batches
        LOGGER.info("Computing embeddings for techniques...")
        attack_embeddings = model.encode(attack_texts, show_progress_bar=True, batch_size=32)
        
        LOGGER.info("Computing embeddings for controls...")
        control_embeddings = model.encode(control_texts, show_progress_bar=True, batch_size=32)
        
        # Compute similarity matrix using sentence-transformers utility
        LOGGER.info("Computing cosine similarity matrix...")
        # Convert to torch tensors for cos_sim
        if TORCH_AVAILABLE:
            attack_tensor = torch.from_numpy(attack_embeddings)
            control_tensor = torch.from_numpy(control_embeddings)
            similarity_matrix = cos_sim(attack_tensor, control_tensor).numpy()
        else:
            # Fallback to numpy cosine similarity
            similarity_matrix = cosine_similarity(attack_embeddings, control_embeddings)
    
    elif TFIDF_AVAILABLE:
        LOGGER.warning("sentence-transformers not available, falling back to TF-IDF")
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        
        LOGGER.info("Computing TF-IDF vectors...")
        all_texts = attack_texts + control_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        attack_tfidf = tfidf_matrix[:len(attack_texts)]
        control_tfidf = tfidf_matrix[len(attack_texts):]
        
        LOGGER.info("Computing cosine similarity matrix...")
        similarity_matrix = cosine_similarity(attack_tfidf, control_tfidf)
    
    else:
        raise ImportError(
            "Neither sentence-transformers nor scikit-learn available. "
            "Install with: pip install sentence-transformers scikit-learn"
        )
    
    # Build mapping DataFrame
    mappings = []
    for i, attack_row in attack_df.iterrows():
        technique_id = attack_row["technique_id"]
        
        # Get similarities for this technique
        tech_idx = attack_df.index.get_loc(i)
        similarities = similarity_matrix[tech_idx, :]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:config.top_k]
        
        for idx in top_k_indices:
            similarity = float(similarities[idx])
            
            # Filter by minimum similarity
            if similarity >= config.min_similarity:
                control_row = defense_df.iloc[idx]
                mappings.append({
                    "technique_id": technique_id,
                    "control_id": control_row["control_id"],
                    "similarity_score": similarity,
                })
    
    result_df = pd.DataFrame(mappings)
    
    # Ensure columns exist even if empty
    if len(result_df) == 0:
        result_df = pd.DataFrame(columns=["technique_id", "control_id", "similarity_score"])
        LOGGER.warning("No mappings generated - all similarities below threshold")
    else:
        LOGGER.info(f"Generated {len(result_df)} heuristic mappings")
        LOGGER.info(f"Average similarity score: {result_df['similarity_score'].mean():.4f}")
        LOGGER.info(f"Min similarity score: {result_df['similarity_score'].min():.4f}")
        LOGGER.info(f"Max similarity score: {result_df['similarity_score'].max():.4f}")
        LOGGER.info(f"Mappings per technique: {len(result_df) / len(attack_df):.2f} (avg)")
    
    # Check diversity if requested
    if check_diversity:
        try:
            result_df = ensure_diversity_from_deterministic(
                learned_mapping_df=result_df,
                deterministic_path=None,  # Will auto-discover
                attack_df=attack_df,
                defense_df=defense_df,
                similarity_matrix=similarity_matrix,
                config=config,
            )
        except RuntimeError as e:
            LOGGER.error(f"Failed to ensure diversity: {e}")
            LOGGER.error("Returning mapping anyway, but it may be identical to deterministic")
    
    # Sanity check: Ensure learned mapping is broader than deterministic
    try:
        validate_learned_is_broader(result_df)
    except RuntimeError as e:
        LOGGER.error("=" * 80)
        LOGGER.error("SANITY CHECK FAILED: Learned mapping is not broader than deterministic!")
        LOGGER.error("=" * 80)
        raise
    
    return result_df, attack_df, defense_df, similarity_matrix


def validate_learned_is_broader(learned_mapping_df: pd.DataFrame) -> None:
    """
    Sanity check: Validate that learned mapping is broader/noisier than deterministic.
    
    Requirements:
    1. Learned mapping has MORE pairs than deterministic
    2. Learned mapping contains controls NOT in deterministic mapping
    3. For each technique, learned controls are NOT a subset of deterministic controls
    
    Raises:
        RuntimeError: If learned mapping is not broader than deterministic
    """
    # Try to find deterministic mapping
    candidates = [
        Path("data/mappings/deterministic_attack_defense_lookup.csv"),
        Path("data/mappings/deterministic_lookup.csv"),
    ]
    deterministic_path = None
    for candidate in candidates:
        if candidate.exists():
            deterministic_path = candidate
            break
    
    if deterministic_path is None or not deterministic_path.exists():
        LOGGER.warning("Deterministic mapping not found, skipping broader-than-deterministic check")
        return
    
    LOGGER.info("=" * 80)
    LOGGER.info("SANITY CHECK: Validating learned mapping is broader than deterministic")
    LOGGER.info("=" * 80)
    
    # Load deterministic mapping
    det_df = pd.read_csv(deterministic_path)
    
    # Normalize column names
    det_tech_col = "attack_id" if "attack_id" in det_df.columns else "technique_id"
    det_ctrl_col = "defense_id" if "defense_id" in det_df.columns else "control_id"
    
    # Create pairs
    det_pairs = set(
        zip(det_df[det_tech_col].dropna().astype(str), det_df[det_ctrl_col].dropna().astype(str))
    )
    learned_pairs = set(
        zip(learned_mapping_df["technique_id"].dropna().astype(str), 
            learned_mapping_df["control_id"].dropna().astype(str))
    )
    
    # Check 1: Learned has more pairs than deterministic
    if len(learned_pairs) <= len(det_pairs):
        raise RuntimeError(
            f"Learned mapping has {len(learned_pairs)} pairs, but deterministic has {len(det_pairs)} pairs. "
            f"Learned mapping must have MORE pairs than deterministic. "
            f"Adjust top_k/min_similarity or logic."
        )
    
    # Check 2: Learned contains controls NOT in deterministic
    learned_only_pairs = learned_pairs - det_pairs
    if len(learned_only_pairs) == 0:
        raise RuntimeError(
            "Learned mapping contains NO controls that are not in deterministic mapping. "
            "All learned pairs are subsets of deterministic pairs. "
            "This means learned mapping is NOT broader/noisier. "
            "Adjust top_k/min_similarity or logic."
        )
    
    # Check 3: For each technique, learned controls are NOT a subset of deterministic controls
    # Group by technique
    det_by_tech = {}
    for tech, ctrl in det_pairs:
        if tech not in det_by_tech:
            det_by_tech[tech] = set()
        det_by_tech[tech].add(ctrl)
    
    learned_by_tech = {}
    for tech, ctrl in learned_pairs:
        if tech not in learned_by_tech:
            learned_by_tech[tech] = set()
        learned_by_tech[tech].add(ctrl)
    
    # Check if any technique has learned controls that are a subset of deterministic
    techniques_with_only_det_controls = []
    for tech in learned_by_tech:
        if tech in det_by_tech:
            learned_ctrls = learned_by_tech[tech]
            det_ctrls = det_by_tech[tech]
            if learned_ctrls.issubset(det_ctrls):
                techniques_with_only_det_controls.append(tech)
    
    if len(techniques_with_only_det_controls) == len(learned_by_tech):
        # All techniques have learned controls that are subsets of deterministic
        raise RuntimeError(
            f"For ALL {len(learned_by_tech)} techniques, learned controls are subsets of deterministic controls. "
            f"Learned mapping is NOT broader than deterministic. "
            f"Adjust top_k/min_similarity or logic."
        )
    
    # Log success metrics
    LOGGER.info("✓ SANITY CHECK PASSED: Learned mapping is broader than deterministic")
    LOGGER.info(f"  Deterministic pairs: {len(det_pairs)}")
    LOGGER.info(f"  Learned pairs: {len(learned_pairs)}")
    LOGGER.info(f"  Learned-only pairs: {len(learned_only_pairs)}")
    LOGGER.info(f"  Techniques with extra learned controls: {len(learned_by_tech) - len(techniques_with_only_det_controls)}/{len(learned_by_tech)}")
    if techniques_with_only_det_controls:
        LOGGER.warning(f"  Techniques with only deterministic controls: {len(techniques_with_only_det_controls)}")
    LOGGER.info("=" * 80)


def ensure_diversity_from_deterministic(
    learned_mapping_df: pd.DataFrame,
    deterministic_path: Optional[Path] = None,
    attack_df: Optional[pd.DataFrame] = None,
    defense_df: Optional[pd.DataFrame] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    config: Optional[HeuristicMappingConfig] = None,
) -> pd.DataFrame:
    """
    Ensure learned mapping is different from deterministic mapping.
    
    If mappings are identical, adjust by:
    1. Relaxing min_similarity slightly and re-picking controls, OR
    2. Dropping overlapping highest-similarity controls and including next-best
    
    Args:
        learned_mapping_df: Generated learned mapping DataFrame
        deterministic_path: Path to deterministic mapping CSV
        attack_df: Attack techniques DataFrame (needed for re-generation)
        defense_df: Defense controls DataFrame (needed for re-generation)
        similarity_matrix: Similarity matrix (needed for re-generation)
        config: Configuration (needed for re-generation)
        
    Returns:
        Adjusted learned mapping DataFrame that is different from deterministic
    """
    if deterministic_path is None:
        # Try to find deterministic mapping
        candidates = [
            Path("data/mappings/deterministic_lookup.csv"),
            Path("data/mappings/deterministic_attack_defense_lookup.csv"),
        ]
        for candidate in candidates:
            if candidate.exists():
                deterministic_path = candidate
                break
    
    if deterministic_path is None or not deterministic_path.exists():
        LOGGER.warning("Deterministic mapping not found, skipping diversity check")
        return learned_mapping_df
    
    LOGGER.info("Checking learned mapping diversity against deterministic...")
    
    # Load deterministic mapping
    det_df = pd.read_csv(deterministic_path)
    
    # Normalize column names
    det_tech_col = "attack_id" if "attack_id" in det_df.columns else "technique_id"
    det_ctrl_col = "defense_id" if "defense_id" in det_df.columns else "control_id"
    
    det_pairs = set(
        zip(det_df[det_tech_col].dropna().astype(str), det_df[det_ctrl_col].dropna().astype(str))
    )
    learned_pairs = set(
        zip(learned_mapping_df["technique_id"].dropna().astype(str), 
            learned_mapping_df["control_id"].dropna().astype(str))
    )
    
    if det_pairs == learned_pairs and len(det_pairs) > 0:
        LOGGER.warning("=" * 80)
        LOGGER.warning("WARNING: Learned mapping is IDENTICAL to deterministic!")
        LOGGER.warning("=" * 80)
        LOGGER.warning("Attempting to increase diversity...")
        
        # Strategy 1: Relax min_similarity slightly and include more controls
        if config and attack_df is not None and defense_df is not None and similarity_matrix is not None:
            relaxed_min_sim = max(0.30, config.min_similarity - 0.05)
            LOGGER.info(f"Trying relaxed min_similarity: {relaxed_min_sim:.2f}")
            
            # Re-generate with relaxed threshold and increased top_k
            adjusted_config = HeuristicMappingConfig(
                top_k=config.top_k + 1,  # Increase top_k
                min_similarity=relaxed_min_sim,
                model_name=config.model_name,
                seed=config.seed,
            )
            
            mappings = []
            for i, attack_row in attack_df.iterrows():
                technique_id = attack_row["technique_id"]
                tech_idx = attack_df.index.get_loc(i)
                similarities = similarity_matrix[tech_idx, :]
                
                # Get top-k+1 indices (increased)
                top_k_indices = np.argsort(similarities)[::-1][:adjusted_config.top_k]
                
                for idx in top_k_indices:
                    similarity = float(similarities[idx])
                    if similarity >= adjusted_config.min_similarity:
                        control_row = defense_df.iloc[idx]
                        mappings.append({
                            "technique_id": technique_id,
                            "control_id": control_row["control_id"],
                            "similarity_score": similarity,
                        })
            
            adjusted_df = pd.DataFrame(mappings)
            adjusted_pairs = set(
                zip(adjusted_df["technique_id"].dropna().astype(str),
                    adjusted_df["control_id"].dropna().astype(str))
            )
            
            if adjusted_pairs != det_pairs:
                LOGGER.info("✓ Diversity adjustment successful with relaxed threshold")
                return adjusted_df
            else:
                LOGGER.warning("Relaxed threshold still produced identical mapping")
        
        # Strategy 2: For each technique, prioritize non-deterministic controls
        if attack_df is not None and defense_df is not None and similarity_matrix is not None:
            LOGGER.info("Trying strategy: prioritize non-deterministic controls...")
            
            mappings = []
            for i, attack_row in attack_df.iterrows():
                technique_id = attack_row["technique_id"]
                tech_idx = attack_df.index.get_loc(i)
                similarities = similarity_matrix[tech_idx, :]
                
                # Get deterministic controls for this technique
                det_controls_for_tech = {
                    ctrl for tech, ctrl in det_pairs if tech == str(technique_id)
                }
                
                # Get all indices sorted by similarity
                all_indices = np.argsort(similarities)[::-1]
                
                # Separate into deterministic and non-deterministic controls
                det_controls_list = []
                non_det_controls_list = []
                
                for idx in all_indices:
                    similarity = float(similarities[idx])
                    if similarity >= config.min_similarity:
                        control_id = str(defense_df.iloc[idx]["control_id"])
                        control_info = {
                            "control_id": control_id,
                            "similarity_score": similarity,
                        }
                        
                        if control_id in det_controls_for_tech:
                            det_controls_list.append(control_info)
                        else:
                            non_det_controls_list.append(control_info)
                
                # Prioritize non-deterministic controls, but include some deterministic if needed
                selected_controls = []
                
                # First, add non-deterministic controls (up to top_k)
                for ctrl_info in non_det_controls_list[:config.top_k]:
                    selected_controls.append(ctrl_info["control_id"])
                    mappings.append({
                        "technique_id": technique_id,
                        "control_id": ctrl_info["control_id"],
                        "similarity_score": ctrl_info["similarity_score"],
                    })
                
                # If we need more controls and have deterministic ones, add them
                # But limit deterministic to at most top_k - 1 to ensure some difference
                max_det = max(0, config.top_k - len(selected_controls) - 1)
                for ctrl_info in det_controls_list[:max_det]:
                    if len(selected_controls) < config.top_k:
                        selected_controls.append(ctrl_info["control_id"])
                        mappings.append({
                            "technique_id": technique_id,
                            "control_id": ctrl_info["control_id"],
                            "similarity_score": ctrl_info["similarity_score"],
                        })
            
            adjusted_df = pd.DataFrame(mappings)
            adjusted_pairs = set(
                zip(adjusted_df["technique_id"].dropna().astype(str),
                    adjusted_df["control_id"].dropna().astype(str))
            )
            
            if adjusted_pairs != det_pairs:
                LOGGER.info("✓ Diversity adjustment successful by prioritizing non-deterministic controls")
                return adjusted_df
            else:
                LOGGER.error("=" * 80)
                LOGGER.error("CRITICAL: Learned mapping is still identical after diversity adjustment!")
                LOGGER.error("=" * 80)
                raise RuntimeError(
                    "Learned mapping is still identical to deterministic after diversity adjustment. "
                    "This indicates that embedding similarities are producing the same top-k controls "
                    "as the deterministic mapping. Try increasing top_k further or using a different "
                    "embedding model."
                )
        else:
            # Cannot adjust - missing required data
            LOGGER.error("=" * 80)
            LOGGER.error("CRITICAL: Learned mapping is IDENTICAL to deterministic!")
            LOGGER.error("=" * 80)
            raise RuntimeError(
                "Learned mapping is identical to deterministic mapping. "
                "Cannot adjust automatically. Regenerate with higher top_k or different parameters."
            )
    else:
        LOGGER.info("✓ Learned mapping is different from deterministic (as expected)")
        intersection = det_pairs & learned_pairs
        overlap_pct = (len(intersection) / len(det_pairs) * 100.0) if len(det_pairs) > 0 else 0.0
        union_pairs = det_pairs | learned_pairs
        jaccard = (len(intersection) / len(union_pairs) * 100.0) if len(union_pairs) > 0 else 0.0
        LOGGER.info(f"  Overlap: {len(intersection)}/{len(det_pairs)} ({overlap_pct:.1f}%)")
        LOGGER.info(f"  Jaccard similarity: {jaccard:.2f}%")
    
    return learned_mapping_df


def main() -> None:
    """Main entry point for CLI."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    parser = argparse.ArgumentParser(
        description="Build heuristic ATT&CK→D3FEND mapping using text similarity"
    )
    parser.add_argument(
        "--attack",
        type=str,
        help="Path to attack_techniques.csv or enterprise-attack.json",
    )
    parser.add_argument(
        "--d3fend",
        type=str,
        help="Path to d3fend_controls.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/mappings/learned_mapping.csv",
        help="Output path for learned_mapping.csv (default: data/mappings/learned_mapping.csv)",
    )
    parser.add_argument(
        "--top-k",
        "--top_k",
        type=int,
        default=10,
        help="Number of controls per technique (default: 10 for broad, generic mapping)",
    )
    parser.add_argument(
        "--min-similarity",
        "--min_similarity",
        type=float,
        default=0.25,
        help="Minimum similarity threshold (default: 0.25 for broad, noisy mapping)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic behavior (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Auto-discover paths if not provided
    if not args.attack:
        # Try common locations
        candidates = [
            Path("data/ontology/attack_techniques.csv"),
            Path("data/mitre/raw/enterprise-attack.json"),
            Path("../mappings/data/mitre/raw/enterprise-attack.json"),
        ]
        for candidate in candidates:
            if candidate.exists():
                args.attack = str(candidate)
                LOGGER.info(f"Auto-discovered attack data at: {args.attack}")
                break
    
    if not args.d3fend:
        # Try common locations
        candidates = [
            Path("data/ontology/d3fend_controls.csv"),
            Path("data/mitre/raw/d3fend.csv"),
            Path("../mappings/data/mitre/raw/d3fend.csv"),
        ]
        for candidate in candidates:
            if candidate.exists():
                args.d3fend = str(candidate)
                LOGGER.info(f"Auto-discovered D3FEND data at: {args.d3fend}")
                break
    
    if not args.attack or not args.d3fend:
        parser.error(
            "Must provide --attack and --d3fend paths, or ensure files exist in "
            "data/ontology/ or data/mitre/raw/"
        )
    
    # Build config
    config = HeuristicMappingConfig(
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        model_name=args.model,
        seed=args.seed,
    )
    
    # Build mapping (diversity check is done inside build_heuristic_mapping)
    mapping_df, _, _, _ = build_heuristic_mapping(
        attack_path=args.attack,
        d3fend_path=args.d3fend,
        config=config,
        check_diversity=True,
    )
    
    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(out_path, index=False)
    LOGGER.info(f"Saved learned mapping to {out_path}")
    LOGGER.info(f"Total mappings: {len(mapping_df)}")
    LOGGER.info("Heuristic mapping complete!")
    LOGGER.info("")
    LOGGER.info("Next steps:")
    LOGGER.info("  1. Diagnose overlap: python scripts/diagnose_mapping_overlap.py")
    LOGGER.info("  2. Run H3 evaluation: python run_h3_evaluation.py")


if __name__ == "__main__":
    main()
