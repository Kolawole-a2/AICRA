"""
Learned ATT&CK→D3FEND Mapping using ML Classifier (Method C).

This module implements a multi-label ML classifier that predicts D3FEND defenses
from ATT&CK techniques based on text features. The classifier is trained on the
deterministic mapping (gold standard) but does not use ontology edges at inference time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

RANDOM_SEED = 42
LOGGER = logging.getLogger(__name__)


def load_deterministic_mapping(mapping_path: Path) -> pd.DataFrame:
    """
    Load deterministic ATT&CK→D3FEND lookup table.

    Args:
        mapping_path: Path to deterministic_attack_defense_lookup.csv

    Returns:
        DataFrame with columns: attack_id, attack_name, defense_id, defense_name, is_correct
    """
    LOGGER.info(f"Loading deterministic mapping from {mapping_path}")
    df = pd.read_csv(mapping_path)

    # Validate required columns
    required_cols = ["attack_id", "attack_name", "defense_id", "defense_name"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter to correct mappings only (is_correct == 1)
    if "is_correct" in df.columns:
        df = df[df["is_correct"] == 1]
        LOGGER.info(f"Filtered to {len(df)} correct mappings")

    LOGGER.info(f"Loaded {len(df)} deterministic mappings")
    LOGGER.info(
        f"Unique attacks: {df['attack_id'].nunique()}, Unique defenses: {df['defense_id'].nunique()}"
    )

    return df


def load_attack_catalog(catalog_path: Path) -> pd.DataFrame:
    """
    Load ATT&CK catalog.

    Args:
        catalog_path: Path to attack_catalog.csv

    Returns:
        DataFrame with columns: attack_id, attack_name
    """
    LOGGER.info(f"Loading ATT&CK catalog from {catalog_path}")
    df = pd.read_csv(catalog_path)

    required_cols = ["attack_id", "attack_name"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    LOGGER.info(f"Loaded {len(df)} ATT&CK techniques")
    return df


def load_defense_catalog(catalog_path: Path) -> pd.DataFrame:
    """
    Load D3FEND defense catalog.

    Args:
        catalog_path: Path to defense_catalog.csv

    Returns:
        DataFrame with columns: defense_id, defense_name
    """
    LOGGER.info(f"Loading D3FEND defense catalog from {catalog_path}")
    df = pd.read_csv(catalog_path)

    required_cols = ["defense_id", "defense_name"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    LOGGER.info(f"Loaded {len(df)} D3FEND defenses")
    return df


def prepare_training_data(
    deterministic_mapping: pd.DataFrame,
    attack_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare training data from deterministic mapping.

    Groups defenses by attack_id to create multi-label targets.

    Args:
        deterministic_mapping: DataFrame with attack_id, defense_id pairs
        attack_catalog: DataFrame with attack_id, attack_name

    Returns:
        Tuple of (features_df, labels_series) where:
        - features_df: DataFrame with attack_id and attack_name (text features)
        - labels_series: Series with attack_id -> list of defense_ids
    """
    LOGGER.info("Preparing training data from deterministic mapping")

    # Group defenses by attack_id
    attack_to_defenses = (
        deterministic_mapping.groupby("attack_id")["defense_id"].apply(list).to_dict()
    )

    # Get unique attack_ids that have mappings
    attack_ids_with_mappings = list(attack_to_defenses.keys())

    # Merge with attack catalog to get attack_name
    training_attacks = attack_catalog[
        attack_catalog["attack_id"].isin(attack_ids_with_mappings)
    ].copy()

    # Create labels series
    labels = pd.Series(
        [attack_to_defenses[aid] for aid in training_attacks["attack_id"]],
        index=training_attacks["attack_id"],
    )

    LOGGER.info(f"Prepared {len(training_attacks)} training examples")
    LOGGER.info(f"Average defenses per attack: {labels.apply(len).mean():.2f}")

    return training_attacks[["attack_id", "attack_name"]], labels


def extract_text_features(
    attack_names: pd.Series,
    vectorizer: TfidfVectorizer | None = None,
) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Extract TF-IDF features from attack names.

    Args:
        attack_names: Series of attack names (text)
        vectorizer: Optional pre-fitted vectorizer. If None, fits a new one.

    Returns:
        Tuple of (feature_matrix, fitted_vectorizer)
    """
    LOGGER.info("Extracting TF-IDF features from attack names")

    # Convert to list of strings, handling NaN values
    texts = attack_names.fillna("").astype(str).tolist()

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1,  # Include all terms
            stop_words="english",
        )
        feature_matrix = vectorizer.fit_transform(texts)
        LOGGER.info(
            f"Fitted TF-IDF vectorizer with {len(vectorizer.vocabulary_)} features"
        )
    else:
        feature_matrix = vectorizer.transform(texts)
        LOGGER.info(f"Transformed {len(texts)} texts using pre-fitted vectorizer")

    return feature_matrix.toarray(), vectorizer


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    mlb: MultiLabelBinarizer,
) -> tuple[OneVsRestClassifier, dict[str, Any]]:
    """
    Train multi-label classifier.

    Args:
        X_train: Training feature matrix (TF-IDF features)
        y_train: Training labels (binary matrix from MultiLabelBinarizer)
        mlb: Fitted MultiLabelBinarizer

    Returns:
        Tuple of (trained_classifier, metrics_dict)
    """
    LOGGER.info("Training multi-label classifier")
    LOGGER.info(
        f"Training on {X_train.shape[0]} examples with {X_train.shape[1]} features"
    )
    LOGGER.info(f"Predicting {y_train.shape[1]} defense classes")

    # Use OneVsRestClassifier with LogisticRegression
    base_classifier = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=1000,
        class_weight="balanced",  # Handle class imbalance
    )
    classifier = OneVsRestClassifier(base_classifier)

    # Train
    classifier.fit(X_train, y_train)

    # Evaluate on training set (for logging)
    y_pred_train = classifier.predict(X_train)
    train_f1 = f1_score(y_train, y_pred_train, average="micro")

    LOGGER.info(f"Training F1-score (micro): {train_f1:.4f}")

    metrics = {
        "train_f1_micro": train_f1,
        "n_features": X_train.shape[1],
        "n_classes": y_train.shape[1],
        "n_samples": X_train.shape[0],
    }

    return classifier, metrics


def predict_defenses(
    classifier: OneVsRestClassifier,
    mlb: MultiLabelBinarizer,
    X: np.ndarray,
    threshold: float = 0.5,
) -> tuple[list[list[str]], np.ndarray]:
    """
    Predict defenses for given attack features.

    Args:
        classifier: Trained multi-label classifier
        mlb: Fitted MultiLabelBinarizer
        X: Feature matrix
        threshold: Probability threshold for binary predictions

    Returns:
        Tuple of (predicted_defense_lists, confidence_scores)
        - predicted_defense_lists: List of lists of defense_ids
        - confidence_scores: Array of shape (n_samples, n_classes) with probabilities
    """
    LOGGER.info(f"Predicting defenses for {X.shape[0]} attacks")

    # Get probability predictions
    probabilities = classifier.predict_proba(X)

    # Get binary predictions using threshold
    binary_predictions = (probabilities >= threshold).astype(int)

    # Convert back to defense_id lists
    predicted_defense_lists = mlb.inverse_transform(binary_predictions)

    LOGGER.info(
        f"Average defenses per attack: {np.mean([len(d) for d in predicted_defense_lists]):.2f}"
    )

    return predicted_defense_lists, probabilities


def generate_learned_mapping(
    deterministic_mapping_path: Path,
    attack_catalog_path: Path,
    defense_catalog_path: Path,
    output_path: Path,
    test_size: float = 0.2,
    prediction_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Generate learned ATT&CK→D3FEND mapping using ML classifier.

    This function:
    1. Loads deterministic mapping (gold standard) for training
    2. Loads attack and defense catalogs
    3. Trains a multi-label classifier on text features
    4. Predicts defenses for all attacks in the catalog
    5. Generates a learned mapping table with confidence scores

    Args:
        deterministic_mapping_path: Path to deterministic_attack_defense_lookup.csv
        attack_catalog_path: Path to attack_catalog.csv
        defense_catalog_path: Path to defense_catalog.csv
        output_path: Path to save learned mapping CSV
        test_size: Fraction of data to use for testing (for evaluation only)
        prediction_threshold: Probability threshold for binary predictions

    Returns:
        DataFrame with learned mapping: attack_id, attack_name, defense_id, defense_name, confidence
    """
    LOGGER.info("=" * 80)
    LOGGER.info("Generating Learned ATT&CK→D3FEND Mapping using ML Classifier")
    LOGGER.info("=" * 80)

    # Load data
    deterministic_mapping = load_deterministic_mapping(deterministic_mapping_path)
    attack_catalog = load_attack_catalog(attack_catalog_path)
    defense_catalog = load_defense_catalog(defense_catalog_path)

    # Prepare training data
    training_attacks, training_labels = prepare_training_data(
        deterministic_mapping,
        attack_catalog,
    )

    # Extract text features
    X, vectorizer = extract_text_features(training_attacks["attack_name"])

    # Encode multi-label targets
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(training_labels)

    LOGGER.info(f"Encoded {y.shape[1]} unique defense classes")

    # Split for evaluation (optional, but useful for metrics)
    if test_size > 0:
        X_train, X_test, y_train, y_test, _, _ = train_test_split(
            X,
            y,
            training_attacks["attack_id"].values,
            test_size=test_size,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
    else:
        X_train, X_test = X, X
        y_train, y_test = y, y

    # Train classifier
    classifier, train_metrics = train_classifier(X_train, y_train, mlb)

    # Evaluate on test set if split
    if test_size > 0:
        y_pred_test = classifier.predict(X_test)
        test_f1 = f1_score(y_test, y_pred_test, average="micro")
        LOGGER.info(f"Test F1-score (micro): {test_f1:.4f}")

        # Log classification report
        LOGGER.info("\nClassification Report (Test Set):")
        LOGGER.info(
            "\n" + classification_report(y_test, y_pred_test, target_names=mlb.classes_)
        )

    # Predict on ALL attacks in catalog (for final mapping)
    LOGGER.info("\nPredicting defenses for all attacks in catalog...")
    all_attacks = attack_catalog[["attack_id", "attack_name"]].copy()
    X_all, _ = extract_text_features(all_attacks["attack_name"], vectorizer=vectorizer)

    predicted_defenses, confidence_scores = predict_defenses(
        classifier,
        mlb,
        X_all,
        threshold=prediction_threshold,
    )

    # Build learned mapping DataFrame
    learned_mapping_rows = []
    defense_id_to_name = dict(
        zip(
            defense_catalog["defense_id"], defense_catalog["defense_name"], strict=False
        )
    )

    for _i, attack_id in enumerate(all_attacks["attack_id"]):
        attack_name = all_attacks[all_attacks["attack_id"] == attack_id][
            "attack_name"
        ].iloc[0]
        defense_ids = predicted_defenses[_i]
        confidences = confidence_scores[_i]

        for defense_id in defense_ids:
            if defense_id in defense_id_to_name:
                defense_name = defense_id_to_name[defense_id]
                # Get confidence score for this defense
                defense_idx = mlb.classes_.tolist().index(defense_id)
                confidence = confidences[defense_idx]

                learned_mapping_rows.append(
                    {
                        "attack_id": attack_id,
                        "attack_name": attack_name,
                        "defense_id": defense_id,
                        "defense_name": defense_name,
                        "confidence": confidence,
                    }
                )

    learned_mapping_df = pd.DataFrame(learned_mapping_rows)

    LOGGER.info(
        f"\nGenerated learned mapping with {len(learned_mapping_df)} attack-defense pairs"
    )
    LOGGER.info(f"Unique attacks: {learned_mapping_df['attack_id'].nunique()}")
    LOGGER.info(f"Unique defenses: {learned_mapping_df['defense_id'].nunique()}")
    LOGGER.info(f"Average confidence: {learned_mapping_df['confidence'].mean():.4f}")

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    learned_mapping_df.to_csv(output_path, index=False)
    LOGGER.info(f"Saved learned mapping to {output_path}")

    return learned_mapping_df


def load_deterministic_lookup(path: Path) -> pd.DataFrame:
    """
    Load deterministic_attack_defense_lookup.csv using pd.read_csv.

    Logs the number of rows and unique attacks/defenses.

    Args:
        path: Path to deterministic_attack_defense_lookup.csv

    Returns:
        DataFrame with deterministic lookup data
    """
    LOGGER.info(f"Loading deterministic lookup from {path}")
    df = pd.read_csv(path)

    num_rows = len(df)
    num_unique_attacks = df["attack_id"].nunique() if "attack_id" in df.columns else 0
    num_unique_defenses = (
        df["defense_id"].nunique() if "defense_id" in df.columns else 0
    )

    LOGGER.info(f"Loaded {num_rows} rows")
    LOGGER.info(
        f"Unique attacks: {num_unique_attacks}, Unique defenses: {num_unique_defenses}"
    )

    return df


def build_attack_to_defenses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by attack_id to construct one row per attack_id with defense_ids as a list.

    Input: deterministic lookup DataFrame with at least attack_id, attack_name, defense_id.

    Creates:
    - attack_id
    - attack_name (take first non-null)
    - defense_ids (a Python list of all linked defense_ids for that attack)

    Removes attacks with no defenses or empty lists.

    Args:
        df: DataFrame with attack_id, attack_name, defense_id columns

    Returns:
        DataFrame with one row per attack_id, including defense_ids as a list
    """
    LOGGER.info("Building attack to defenses mapping")

    # Validate required columns
    required_cols = ["attack_id", "defense_id"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Prepare aggregation dictionary
    agg_dict = {"defense_id": list}
    if "attack_name" in df.columns:
        # Use first non-null value
        def first_non_null(x):
            non_null = x.dropna()
            return non_null.iloc[0] if len(non_null) > 0 else None

        agg_dict["attack_name"] = first_non_null

    # Group by attack_id to collect defense_ids
    grouped = df.groupby("attack_id").agg(agg_dict).reset_index()

    # Rename defense_id to defense_ids
    grouped = grouped.rename(columns={"defense_id": "defense_ids"})

    # Remove duplicates from defense_ids lists
    grouped["defense_ids"] = grouped["defense_ids"].apply(
        lambda x: list(set(x)) if isinstance(x, list) else []
    )

    # Filter out attacks with no defenses or empty lists
    grouped = grouped[grouped["defense_ids"].apply(lambda x: len(x) > 0)]

    num_attacks = len(grouped)
    avg_defenses = grouped["defense_ids"].apply(len).mean()

    LOGGER.info(f"Built mapping for {num_attacks} attack_ids")
    LOGGER.info(f"Average number of defenses per attack: {avg_defenses:.2f}")

    return grouped


def prepare_text_features(
    df_attacks: pd.DataFrame,
) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Prepare text features from attack names (and optionally descriptions).

    Feature text should be:
    - attack_name, and if available, optionally concatenated with a longer description
      (if attack_desc or similar exists; detect this gracefully).

    Uses TfidfVectorizer (unigrams + bigrams, default parameters).
    Fits on all attack texts and transforms into a TF-IDF matrix X.

    Args:
        df_attacks: DataFrame with at least attack_name column

    Returns:
        Tuple of (X: TF-IDF matrix, vectorizer: fitted TfidfVectorizer)
    """
    LOGGER.info("Preparing text features from attack names")

    if "attack_name" not in df_attacks.columns:
        raise ValueError("DataFrame must contain 'attack_name' column")

    # Prepare text features
    texts = []
    for _idx, row in df_attacks.iterrows():
        text_parts = [str(row["attack_name"]) if pd.notna(row["attack_name"]) else ""]

        # Check for description columns (attack_desc, attack_description, description)
        desc_cols = ["attack_desc", "attack_description", "description"]
        for desc_col in desc_cols:
            if desc_col in df_attacks.columns and pd.notna(row.get(desc_col)):
                text_parts.append(str(row[desc_col]))
                break  # Use first available description column

        texts.append(" ".join(text_parts).strip())

    # Use TfidfVectorizer with unigrams and bigrams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    LOGGER.info(
        f"Prepared TF-IDF features: {X.shape[0]} samples, {X.shape[1]} features"
    )

    return X.toarray(), vectorizer


def prepare_labels(
    df_attacks: pd.DataFrame,
) -> tuple[np.ndarray, MultiLabelBinarizer, list[str]]:
    """
    Use MultiLabelBinarizer to transform the defense_ids list into a multi-hot label matrix Y.

    Args:
        df_attacks: DataFrame with defense_ids column (list of defense_id strings)

    Returns:
        Tuple of:
        - Y: multi-hot label matrix (numpy array)
        - mlb: fitted MultiLabelBinarizer
        - classes: list of defense_id labels (from mlb.classes_)
    """
    LOGGER.info("Preparing multi-label encoding")

    if "defense_ids" not in df_attacks.columns:
        raise ValueError("DataFrame must contain 'defense_ids' column")

    # Extract defense_ids lists
    defense_lists = df_attacks["defense_ids"].tolist()

    # Fit MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(defense_lists)

    classes = list(mlb.classes_)

    LOGGER.info(f"Encoded {len(classes)} unique defense classes")
    LOGGER.info(f"Label matrix shape: {Y.shape}")

    return Y, mlb, classes


def train_attack_to_defense_classifier(
    df_deterministic: pd.DataFrame,
) -> dict[str, Any]:
    """
    Train a multi-label classifier that predicts D3FEND defense_ids for each ATT&CK attack_id,
    using attack text features and deterministic mapping as training labels.

    Returns a dictionary with:
      - 'model': fitted OneVsRestClassifier
      - 'vectorizer': TfidfVectorizer
      - 'mlb': MultiLabelBinarizer
      - 'classes': List[str] (defense_ids)
      - 'df_attacks': DataFrame used for training
      - 'metrics': Dict[str, Any] with simple train/val metrics

    Args:
        df_deterministic: DataFrame with deterministic lookup (attack_id, attack_name, defense_id)

    Returns:
        Dictionary containing model, vectorizer, mlb, classes, df_attacks, and metrics
    """
    LOGGER.info("=" * 80)
    LOGGER.info("Training attack to defense classifier")
    LOGGER.info("=" * 80)

    # Step 1: Build attack to defenses mapping
    df_attacks = build_attack_to_defenses(df_deterministic)

    # Step 2: Prepare text features
    X, vectorizer = prepare_text_features(df_attacks)

    # Step 3: Prepare labels
    Y, mlb, defense_classes = prepare_labels(df_attacks)

    # Step 4: Split into train/validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    LOGGER.info(f"Train set: {X_train.shape[0]} samples")
    LOGGER.info(f"Validation set: {X_val.shape[0]} samples")

    # Step 5: Train classifier
    base_classifier = LogisticRegression(
        max_iter=200,
        n_jobs=None,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    classifier = OneVsRestClassifier(base_classifier)

    LOGGER.info("Fitting classifier...")
    classifier.fit(X_train, Y_train)

    # Step 6: Evaluate on validation set
    Y_val_pred = classifier.predict(X_val)

    # Compute metrics
    f1_micro = f1_score(Y_val, Y_val_pred, average="micro")
    f1_macro = f1_score(Y_val, Y_val_pred, average="macro")

    LOGGER.info(f"Validation F1-score (micro): {f1_micro:.4f}")
    LOGGER.info(f"Validation F1-score (macro): {f1_macro:.4f}")

    # Log classification report to LOGGER
    report = classification_report(
        Y_val, Y_val_pred, target_names=mlb.classes_, output_dict=True
    )
    LOGGER.info("\nClassification Report (Validation Set):")
    LOGGER.info(
        "\n" + classification_report(Y_val, Y_val_pred, target_names=mlb.classes_)
    )

    # Prepare metrics dictionary
    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "n_train_samples": X_train.shape[0],
        "n_val_samples": X_val.shape[0],
        "n_features": X_train.shape[1],
        "n_classes": Y_train.shape[1],
        "classification_report": report,
    }

    # Return results dictionary
    return {
        "model": classifier,
        "vectorizer": vectorizer,
        "mlb": mlb,
        "classes": defense_classes,
        "df_attacks": df_attacks,
        "metrics": metrics,
    }


def _prepare_attack_text(df_attacks: pd.DataFrame) -> list[str]:
    """
    Helper function to prepare attack text features (same logic as in prepare_text_features).

    Args:
        df_attacks: DataFrame with at least attack_name column

    Returns:
        List of text strings (one per attack)
    """
    texts = []
    for _idx, row in df_attacks.iterrows():
        text_parts = [str(row["attack_name"]) if pd.notna(row["attack_name"]) else ""]

        # Check for description columns (attack_desc, attack_description, description)
        desc_cols = ["attack_desc", "attack_description", "description"]
        for desc_col in desc_cols:
            if desc_col in df_attacks.columns and pd.notna(row.get(desc_col)):
                text_parts.append(str(row[desc_col]))
                break  # Use first available description column

        texts.append(" ".join(text_parts).strip())

    return texts


def build_learned_mapping_table(
    model: OneVsRestClassifier,
    vectorizer: TfidfVectorizer,
    mlb: MultiLabelBinarizer,
    df_attacks: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    For each attack_id in df_attacks, use the trained classifier to predict
    the most likely defense_ids and build a learned mapping table.

    Returns a DataFrame with columns:
      - attack_id
      - attack_name
      - defense_id
      - score        (predicted probability or decision function)
      - rank         (1 for best, 2 for second, etc.)
      - method       (e.g., 'ml_classifier_v1')

    Args:
        model: Trained OneVsRestClassifier
        vectorizer: Fitted TfidfVectorizer
        mlb: Fitted MultiLabelBinarizer
        df_attacks: DataFrame with attack_id and attack_name columns
        top_k: Number of top defenses to select per attack

    Returns:
        DataFrame with learned mapping table
    """
    LOGGER.info(
        f"Building learned mapping table for {len(df_attacks)} attacks (top_k={top_k})"
    )

    # Prepare text features using the same logic as in training
    texts = _prepare_attack_text(df_attacks)
    X_all = vectorizer.transform(texts)

    # Get scores - use predict_proba if available, otherwise decision_function
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_all)
        # OneVsRestClassifier.predict_proba() may return a list of arrays, convert to numpy array
        if isinstance(scores, list):
            scores = np.array(scores).T  # Transpose to get (n_samples, n_classes)
        scores = np.asarray(scores)
    else:
        scores = model.decision_function(X_all)
        if isinstance(scores, list):
            scores = np.array(scores).T
        scores = np.asarray(scores)

    # Build mapping table
    mapping_rows = []
    defense_classes = mlb.classes_

    for _i, (_, row) in enumerate(df_attacks.iterrows()):
        attack_id = row["attack_id"]
        attack_name = row["attack_name"] if pd.notna(row["attack_name"]) else ""

        # Get scores for this attack across all defense classes
        attack_scores = scores[_i]

        # Rank defenses by score descending
        ranked_indices = np.argsort(attack_scores)[::-1]

        # Select top_k
        top_k_indices = ranked_indices[:top_k]

        # Build rows for this attack
        for rank, idx in enumerate(top_k_indices, start=1):
            defense_id = defense_classes[idx]
            score = float(attack_scores[idx])

            mapping_rows.append(
                {
                    "attack_id": attack_id,
                    "attack_name": attack_name,
                    "defense_id": defense_id,
                    "score": score,
                    "rank": rank,
                    "method": "ml_classifier_v1",
                }
            )

    df_learned = pd.DataFrame(mapping_rows)

    num_attacks = df_learned["attack_id"].nunique()
    num_pairs = len(df_learned)

    LOGGER.info(
        f"Built learned mapping: {num_attacks} unique attack_ids, {num_pairs} attack-defense pairs"
    )

    return df_learned


def save_learned_mapping(df_learned: pd.DataFrame, out_dir: Path) -> None:
    """
    Save the learned mapping table to CSV and Parquet under data/mappings/.

    Args:
        df_learned: DataFrame with learned mapping table
        out_dir: Output directory (should be data/mappings/)
    """
    LOGGER.info(f"Saving learned mapping to {out_dir}")

    # Ensure out_dir exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define output paths
    csv_path = out_dir / "learned_attack_defense_mapping.csv"
    parquet_path = out_dir / "learned_attack_defense_mapping.parquet"

    # Save to CSV
    df_learned.to_csv(csv_path, index=False)
    LOGGER.info(f"Saved CSV to {csv_path}")

    # Save to Parquet
    df_learned.to_parquet(parquet_path, index=False)
    LOGGER.info(f"Saved Parquet to {parquet_path}")

    # Log basic stats
    num_attacks = df_learned["attack_id"].nunique()
    num_defenses = df_learned["defense_id"].nunique()
    num_pairs = len(df_learned)
    avg_score = df_learned["score"].mean()

    LOGGER.info("Learned mapping stats:")
    LOGGER.info(f"  - Unique attacks: {num_attacks}")
    LOGGER.info(f"  - Unique defenses: {num_defenses}")
    LOGGER.info(f"  - Total pairs: {num_pairs}")
    LOGGER.info(f"  - Average score: {avg_score:.4f}")


def convert_to_dac_format(learned_mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert learned mapping DataFrame to format expected by DAC metric module.

    Converts attack_id -> technique_id and defense_id -> control_id.

    Args:
        learned_mapping_df: DataFrame with attack_id, defense_id columns

    Returns:
        DataFrame with technique_id, control_id columns (and other columns preserved)
    """
    dac_df = learned_mapping_df.copy()

    # Rename columns if they exist
    if "attack_id" in dac_df.columns:
        dac_df = dac_df.rename(columns={"attack_id": "technique_id"})
    if "defense_id" in dac_df.columns:
        dac_df = dac_df.rename(columns={"defense_id": "control_id"})

    return dac_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    deterministic_path = Path("data/mappings/deterministic_attack_defense_lookup.csv")
    out_dir = Path("data/mappings")

    df_det = load_deterministic_lookup(deterministic_path)
    result = train_attack_to_defense_classifier(df_det)
    df_learned = build_learned_mapping_table(
        model=result["model"],
        vectorizer=result["vectorizer"],
        mlb=result["mlb"],
        df_attacks=result["df_attacks"],
        top_k=3,
    )
    save_learned_mapping(df_learned, out_dir)

    LOGGER.info("Training metrics: %s", result["metrics"])
