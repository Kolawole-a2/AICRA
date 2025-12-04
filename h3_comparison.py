#!/usr/bin/env python3
"""
H3 Experiment: Deterministic vs Learned ATT&CK→D3FEND Mapping Comparison

Hypothesis (H3):
"Deterministic ATT&CK–D3FEND lookup yields higher risk-score precision and consistency than learned mapping."

This script compares deterministic and learned mappings using:
- Coverage (%)
- Defense–attack consistency (%)
- Δ-precision (for ACTIONABLE positives)
- Variance/IQR reduction of risk scores

Performs statistical tests, produces JSON + Markdown summaries, plots, and SHA256 hashes of mapping files.
"""

import os
import json
import hashlib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import wilcoxon, ttest_rel, pearsonr
import matplotlib.pyplot as plt


# ------------------ CONFIG ------------------
IN_RISK = "risk_scores.csv"
IN_DET = "deterministic_lookup.csv"
IN_LRN = "learned_mapping.csv"
IN_REF = "d3fend_reference_pairs.csv"
IN_IMP = "impact.csv"  # optional

OUT_DIR = "results/H3_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# If you want to demote unmapped positives' scores (stability proxy), set factor < 1.0
DEMOTION_FACTOR = 0.90

SUCCESS_RULES = {
    "coverage_min_%": 85.0,
    "consistency_min_%": 90.0,
    "delta_precision_min": 0.0,
    "variance_reduction_min": 0.0
}


# ------------------ HELPERS ------------------
def file_hash(path):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv_or_fail(path, required_cols):
    """Load CSV and validate required columns exist."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def mapping_metrics(lookup_df, ref_df):
    """
    Compute mapping quality metrics:
    - Coverage: % of techniques with >=1 control
    - Consistency: overlap with canonical MITRE pairs
    - Correctness: % validated (if 'validated' column exists)
    """
    # Coverage: % of techniques with >=1 control
    tech_total = lookup_df["technique_id"].nunique()
    tech_with_control = lookup_df.dropna(subset=["control_id"])["technique_id"].nunique()
    coverage = (tech_with_control / tech_total * 100.0) if tech_total else 0.0

    # Consistency: overlap with canonical MITRE pairs
    lookup_pairs = set(map(tuple, lookup_df[["technique_id", "control_id"]].dropna().values.tolist()))
    ref_pairs = set(map(tuple, ref_df[["technique_id", "control_id"]].dropna().values.tolist()))
    inter = lookup_pairs & ref_pairs
    consistency = (len(inter) / len(lookup_pairs) * 100.0) if len(lookup_pairs) else 0.0

    # Optional correctness if 'validated' exists
    correctness = None
    if "validated" in lookup_df.columns:
        total_pairs = len(lookup_df)
        correctness = (lookup_df["validated"].fillna(0).sum() / total_pairs * 100.0) if total_pairs else 0.0

    return {
        "coverage_%": round(coverage, 2),
        "consistency_%": round(consistency, 2),
        "correctness_%": round(correctness, 2) if correctness is not None else "N/A",
        "pairs_count": len(lookup_pairs)
    }


def actionable_precision(df, mapping_df, ref_df):
    """
    ACTIONABLE positive = predicted_label==1 AND (technique_id, control_id) exists & is canonical-consistent.
    We measure precision only over actionable positives (helps compare mapping quality effect on Register).
    """
    # Build canonical set
    ref_pairs = set(map(tuple, ref_df[["technique_id", "control_id"]].dropna().values.tolist()))
    # For each technique, set of controls produced by mapping
    map_group = mapping_df.dropna().groupby("technique_id")["control_id"].apply(set).to_dict()

    preds = df.copy()
    preds["actionable"] = 0

    # Consider a technique actionable if ANY mapped control is canonical-consistent
    def is_actionable(row):
        if row["predicted_label"] != 1:
            return 0
        tech = row["technique_id"]
        if tech not in map_group:
            return 0
        for ctrl in map_group[tech]:
            if (tech, ctrl) in ref_pairs:
                return 1
        return 0

    preds["actionable"] = preds.apply(is_actionable, axis=1)

    # Precision on actionable subset: TP / (TP+FP) where predicted_label==1 and actionable==1
    subset = preds[(preds["predicted_label"] == 1) & (preds["actionable"] == 1)]
    if subset.empty:
        return {"precision": 0.0, "f1": 0.0, "n_actionable": 0}
    precision = precision_score(subset["true_label"], subset["predicted_label"], zero_division=0)
    f1 = f1_score(subset["true_label"], subset["predicted_label"], zero_division=0)
    return {"precision": round(precision, 4), "f1": round(f1, 4), "n_actionable": int(len(subset))}


def variance_consistency(df, mapping_df, demotion_factor=0.90):
    """
    Score stability proxy: demote probabilities for positives whose technique lacks any mapped control.
    Lower variance after mapping = more consistent Register scoring.
    """
    # Techniques with ANY control in mapping
    techniques_with_controls = set(mapping_df["technique_id"].dropna().unique().tolist())
    mapped = df.copy()

    def adjust(row):
        if row["predicted_label"] == 1 and row["technique_id"] not in techniques_with_controls:
            return row["risk_score"] * demotion_factor
        return row["risk_score"]

    mapped["risk_score_mapped"] = mapped.apply(adjust, axis=1)
    base_var = np.var(df["risk_score"], ddof=1)
    map_var = np.var(mapped["risk_score_mapped"], ddof=1)
    base_iqr = df["risk_score"].quantile(0.75) - df["risk_score"].quantile(0.25)
    map_iqr = mapped["risk_score_mapped"].quantile(0.75) - mapped["risk_score_mapped"].quantile(0.25)
    return {
        "baseline_var": round(base_var, 6),
        "mapped_var": round(map_var, 6),
        "variance_reduction": round(base_var - map_var, 6),
        "baseline_iqr": round(base_iqr, 6),
        "mapped_iqr": round(map_iqr, 6),
        "iqr_reduction": round(base_iqr - map_iqr, 6)
    }


def brier_ece(y_true, y_prob, bins=10):
    """Compute Brier score and Expected Calibration Error (ECE)."""
    brier = brier_score_loss(y_true, y_prob)
    # ECE (equal-width bins)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return round(brier, 6), round(ece, 6)


def bootstrap_diff(a, b, n=1000, seed=42):
    """Returns mean diff and 95% CI for paired arrays."""
    rng = np.random.default_rng(seed)
    diffs = []
    a = np.array(a)
    b = np.array(b)
    assert len(a) == len(b)
    n_obs = len(a)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        diffs.append((a[idx] - b[idx]).mean())
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# ------------------ LOAD ------------------
print("Loading input files...")
risk = load_csv_or_fail(IN_RISK, {"asset_id", "risk_score", "predicted_label", "true_label", "technique_id"})
det = load_csv_or_fail(IN_DET, {"technique_id", "control_id"})
lrn = load_csv_or_fail(IN_LRN, {"technique_id", "control_id"})
refp = load_csv_or_fail(IN_REF, {"technique_id", "control_id"})

impact = None
if os.path.exists(IN_IMP):
    impact = load_csv_or_fail(IN_IMP, {"asset_id", "impact_usd"})
    print(f"Loaded optional impact file: {IN_IMP}")

print(f"Loaded {len(risk)} risk score records")
print(f"Loaded {len(det)} deterministic mapping pairs")
print(f"Loaded {len(lrn)} learned mapping pairs")
print(f"Loaded {len(refp)} reference pairs")


# ------------------ HASHES ------------------
print("\nComputing file hashes...")
hashes = {
    "deterministic_lookup_sha256": file_hash(IN_DET),
    "learned_mapping_sha256": file_hash(IN_LRN),
    "reference_pairs_sha256": file_hash(IN_REF)
}
with open(os.path.join(OUT_DIR, "hashes.txt"), "w") as f:
    for k, v in hashes.items():
        f.write(f"{k}: {v}\n")
print("Hashes saved to hashes.txt")


# ------------------ MAPPING METRICS ------------------
print("\nComputing mapping metrics...")
mm_det = mapping_metrics(det, refp)
mm_lrn = mapping_metrics(lrn, refp)
print(f"Deterministic: Coverage={mm_det['coverage_%']}%, Consistency={mm_det['consistency_%']}%")
print(f"Learned: Coverage={mm_lrn['coverage_%']}%, Consistency={mm_lrn['consistency_%']}%")


# ------------------ ACTIONABLE PRECISION / F1 ------------------
print("\nComputing actionable precision...")
ap_det = actionable_precision(risk, det, refp)
ap_lrn = actionable_precision(risk, lrn, refp)
print(f"Deterministic actionable precision: {ap_det['precision']} (n={ap_det['n_actionable']})")
print(f"Learned actionable precision: {ap_lrn['precision']} (n={ap_lrn['n_actionable']})")


# For paired stats on precision deltas, proxy with per-asset correctness on actionable set (binary vectors)
def actionable_binary_vector(df, mapping_df, ref_df):
    """Create binary vector of correct actionable predictions."""
    ref_pairs = set(map(tuple, ref_df[["technique_id", "control_id"]].dropna().values.tolist()))
    map_group = mapping_df.dropna().groupby("technique_id")["control_id"].apply(set).to_dict()
    # actionable positive predicted
    mask = (df["predicted_label"] == 1)
    sub = df[mask].copy()

    def is_actionable(row):
        tech = row["technique_id"]
        if tech not in map_group:
            return 0
        for ctrl in map_group[tech]:
            if (tech, ctrl) in ref_pairs:
                return 1
        return 0

    sub["actionable"] = sub.apply(is_actionable, axis=1)
    # Correct prediction among actionable positives
    sub["correct_actionable"] = ((sub["actionable"] == 1) & (sub["true_label"] == 1)).astype(int)
    return sub["correct_actionable"].values


vec_det = actionable_binary_vector(risk, det, refp)
vec_lrn = actionable_binary_vector(risk, lrn, refp)
# Align lengths (same actionable-positive mask)
n = min(len(vec_det), len(vec_lrn))
paired = (vec_det[:n], vec_lrn[:n])

# Wilcoxon (non-parametric) if possible
p_wilcoxon = None
try:
    if len(paired[0]) > 0 and (paired[0] != paired[1]).any():
        p_wilcoxon = wilcoxon(paired[0], paired[1], zero_method="wilcox", correction=True).pvalue
except Exception as e:
    print(f"Wilcoxon test failed: {e}")
    p_wilcoxon = None


# ------------------ VARIANCE / IQR CONSISTENCY ------------------
print("\nComputing variance consistency...")
vc_det = variance_consistency(risk, det, DEMOTION_FACTOR)
vc_lrn = variance_consistency(risk, lrn, DEMOTION_FACTOR)
print(f"Deterministic variance reduction: {vc_det['variance_reduction']}")
print(f"Learned variance reduction: {vc_lrn['variance_reduction']}")


# ------------------ DISCRIMINATION & CALIBRATION (unchanged by mapping, reported for completeness) ------------------
print("\nComputing baseline model metrics...")
try:
    auroc = roc_auc_score(risk["true_label"], risk["risk_score"])
except Exception:
    auroc = None
try:
    prauc = average_precision_score(risk["true_label"], risk["risk_score"])
except Exception:
    prauc = None
brier, ece = brier_ece(risk["true_label"], risk["risk_score"])


# ------------------ EXPECTED LOSS (optional if impact provided) ------------------
expected_loss = None
if impact is not None:
    print("\nComputing expected loss...")
    df_el = risk.merge(impact, on="asset_id", how="left")
    df_el["impact_usd"] = df_el["impact_usd"].fillna(
        df_el["impact_usd"].median() if not df_el["impact_usd"].isna().all() else 1.0
    )
    df_el["el"] = df_el["risk_score"] * df_el["impact_usd"]
    expected_loss = float(df_el["el"].sum())
    print(f"Expected loss sum: ${expected_loss:,.2f}")


# ------------------ METRICS PACK ------------------
print("\nAssembling results...")
metrics = {
    "hashes": hashes,
    "deterministic_mapping": mm_det,
    "learned_mapping": mm_lrn,
    "actionable_precision": {
        "deterministic": ap_det,
        "learned": ap_lrn,
        "delta_precision": round(ap_det["precision"] - ap_lrn["precision"], 4),
        "delta_f1": round(ap_det["f1"] - ap_lrn["f1"], 4),
        "wilcoxon_pvalue": p_wilcoxon
    },
    "variance_consistency": {
        "deterministic": vc_det,
        "learned": vc_lrn,
        "delta_variance_reduction": round(vc_det["variance_reduction"] - vc_lrn["variance_reduction"], 6),
        "delta_iqr_reduction": round(vc_det["iqr_reduction"] - vc_lrn["iqr_reduction"], 6)
    },
    "model_baseline": {
        "auroc": auroc,
        "prauc": prauc,
        "brier": brier,
        "ece": ece
    },
    "expected_loss_sum": expected_loss
}

with open(os.path.join(OUT_DIR, "H3_results.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Results saved to {OUT_DIR}/H3_results.json")


# ------------------ SUMMARY MD ------------------
def pass_fail(val, thr, greater=True):
    """Check if value passes threshold."""
    if val is None or isinstance(val, str):
        return "N/A"
    return "PASS" if ((val >= thr) if greater else (val <= thr)) else "FAIL"


cov_ok_det = pass_fail(metrics["deterministic_mapping"]["coverage_%"], SUCCESS_RULES["coverage_min_%"])
con_ok_det = pass_fail(metrics["deterministic_mapping"]["consistency_%"], SUCCESS_RULES["consistency_min_%"])
cov_ok_lrn = pass_fail(metrics["learned_mapping"]["coverage_%"], SUCCESS_RULES["coverage_min_%"])
con_ok_lrn = pass_fail(metrics["learned_mapping"]["consistency_%"], SUCCESS_RULES["consistency_min_%"])

summary = f"""
# H3 Deterministic vs Learned Mapping — Comparison

## Mapping Integrity

- Deterministic Coverage: **{metrics['deterministic_mapping']['coverage_%']}%** ({cov_ok_det})  
- Deterministic Consistency: **{metrics['deterministic_mapping']['consistency_%']}%** ({con_ok_det})  
- Learned Coverage: **{metrics['learned_mapping']['coverage_%']}%** ({cov_ok_lrn})  
- Learned Consistency: **{metrics['learned_mapping']['consistency_%']}%** ({con_ok_lrn})

## Actionable Precision (Register-Level)

- Deterministic Precision: **{metrics['actionable_precision']['deterministic']['precision']}**  
- Learned Precision: **{metrics['actionable_precision']['learned']['precision']}**  
- Δ Precision (Det - Learn): **{metrics['actionable_precision']['delta_precision']}**  
- Wilcoxon p-value (paired actionable correctness): **{metrics['actionable_precision']['wilcoxon_pvalue']}**

## Score Consistency (Stability)

- Deterministic Variance Reduction: **{metrics['variance_consistency']['deterministic']['variance_reduction']}**  
- Learned Variance Reduction: **{metrics['variance_consistency']['learned']['variance_reduction']}**  
- Δ Variance Reduction (Det - Learn): **{metrics['variance_consistency']['delta_variance_reduction']}**

## Baseline Discrimination & Calibration (for context)

- AUROC: **{metrics['model_baseline']['auroc']}**, PR-AUC: **{metrics['model_baseline']['prauc']}**  
- Brier: **{metrics['model_baseline']['brier']}**, ECE: **{metrics['model_baseline']['ece']}**

## Expected Loss (optional)

- Sum(Expected Loss): **{metrics['expected_loss_sum']}**

## Reproducibility

- Deterministic lookup SHA256: `{hashes['deterministic_lookup_sha256']}`
- Learned mapping SHA256: `{hashes['learned_mapping_sha256']}`
- Reference pairs SHA256: `{hashes['reference_pairs_sha256']}`

## Decision Guide

- Prefer deterministic mapping if it achieves **coverage ≥ {SUCCESS_RULES['coverage_min_%']}%**, **consistency ≥ {SUCCESS_RULES['consistency_min_%']}%**, **Δ precision > 0**, and **variance reduction > 0**.
"""

with open(os.path.join(OUT_DIR, "H3_summary.md"), "w") as f:
    f.write(summary)
print(f"Summary saved to {OUT_DIR}/H3_summary.md")

print("\n" + summary)


# ------------------ PLOTS ------------------
def bar_plot(metric_name, det_val, lrn_val, title, fname):
    """Create a bar plot comparing deterministic vs learned metrics."""
    plt.figure(figsize=(8, 6))
    bars = plt.bar(["Deterministic", "Learned"], [det_val, lrn_val], color=['#2e7d32', '#1976d2'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}' if isinstance(height, float) and height < 1 else f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plots", fname), dpi=150)
    plt.close()


os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)
print("\nGenerating plots...")

bar_plot("Coverage (%)",
         metrics["deterministic_mapping"]["coverage_%"],
         metrics["learned_mapping"]["coverage_%"],
         "Mapping Coverage", "coverage.png")

bar_plot("Consistency (%)",
         metrics["deterministic_mapping"]["consistency_%"],
         metrics["learned_mapping"]["consistency_%"],
         "Defense–Attack Consistency", "consistency.png")

bar_plot("Actionable Precision",
         metrics["actionable_precision"]["deterministic"]["precision"],
         metrics["actionable_precision"]["learned"]["precision"],
         "Actionable Precision (Register)", "precision.png")

bar_plot("Variance Reduction",
         metrics["variance_consistency"]["deterministic"]["variance_reduction"],
         metrics["variance_consistency"]["learned"]["variance_reduction"],
         "Risk-Score Variance Reduction", "variance_reduction.png")

print(f"Plots saved to {OUT_DIR}/plots/")

print(f"\n{'='*60}")
print(f"All artifacts saved under: {OUT_DIR}")
print(f"{'='*60}")


