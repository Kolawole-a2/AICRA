# Canonical H1/H2 Experiments vs. Optional Rebuild Pipeline: Explanation for Praxis Defense

## Executive Summary

Your praxis has **two distinct but complementary components**:

1. **Canonical H1/H2/H3 Experiments** (Primary Research Validation)
   - These are the **core hypothesis validation experiments** that test your research hypotheses
   - They produce the **primary research results** for your praxis
   - These are **required** and form the scientific foundation

2. **Optional H1/H2 Rebuild Pipeline** (Post-Hoc Analysis & Operational Artifacts)
   - This is a **post-hoc analysis tool** that generates operational artifacts (risk registers)
   - It does **NOT modify** the canonical experiment results
   - It is **optional** and serves demonstration/exposition purposes
   - It generates **ransomware-only risk registers** for praxis demonstration

---

## Detailed Comparison

### Canonical H1/H2 Experiments (Primary Research)

**Purpose**: Validate research hypotheses H1 and H2

**What They Do**:
- **H1**: Tests whether static PE features enable reliable ransomware classification (AUROC ≥ 0.95)
- **H2**: Tests whether calibration and cost-aware thresholding improve decision quality

**Key Characteristics**:
- Located in: `aicra/experiments/h1_classification.py` and `aicra/experiments/h2_calibration_thresholds.py`
- Results stored in: `results/H1_classification/` and `results/H2_calibration_thresholds/`
- Produces: **Aggregate metrics** (AUROC, PR-AUC, Brier, ECE, etc.) for hypothesis validation
- Focus: **Statistical validation** of research hypotheses
- Output: JSON metrics files and summary markdown reports

**Why This is Required for Praxis**:
- These experiments directly test your stated hypotheses
- They produce the quantitative evidence needed to support your research claims
- They are the **scientific foundation** of your praxis
- Results are used in your dissertation/praxis document

**Example Results**:
- H1: AUROC = 0.9866, PR-AUC = 0.9869, Precision = 0.9459
- H2: Brier improvement, ECE reduction, cost-optimal threshold selection

---

### Optional H1/H2 Rebuild Pipeline (Post-Hoc Analysis)

**Purpose**: Generate operational artifacts (risk registers) for praxis demonstration

**What It Does**:
- Reuses the same EMBER-2024 data and LightGBM model
- Generates **per-sample risk scores** across multiple splits (smoke_test, small_ember, main, full_ember)
- Creates **ransomware-only risk registers** (operational artifacts)
- Produces per-split plots and metrics for demonstration

**Key Characteristics**:
- Located in: `scripts/h1h2_rebuild/`
- Results stored in: `results/h1h2_rebuild/<split>/` and `register/h1h2_rebuild/<split>/`
- Produces: **Per-sample scores** and **risk registers** (operational artifacts)
- Focus: **Operational demonstration** and **praxis exposition**
- Output: Risk register CSV files, per-split metrics, plots

**Why This is Optional**:
- It does **NOT test hypotheses** - it generates operational artifacts
- It is **read-only** with respect to canonical experiments (does not modify H1/H2/H3 results)
- It serves to **demonstrate the end-to-end pipeline** and generate risk registers
- It provides **concrete examples** of how the system works in practice

**Example Outputs**:
- `register/h1h2_rebuild/<split>/ransomware_only_risk_register.csv` - Per-sample risk registers
- `results/h1h2_rebuild/<split>/metrics.json` - Per-split metrics for demonstration

---

## Key Distinctions

| Aspect | Canonical H1/H2 | Optional Rebuild Pipeline |
|--------|----------------|---------------------------|
| **Purpose** | Hypothesis validation | Operational artifact generation |
| **Output Type** | Aggregate metrics | Per-sample scores + risk registers |
| **Required for Praxis?** | ✅ **YES** (core research) | ⚠️ **OPTIONAL** (demonstration) |
| **Modifies Canonical Results?** | N/A (these ARE the canonical results) | ❌ **NO** (read-only) |
| **Scientific Value** | Primary research evidence | Operational demonstration |
| **Use in Dissertation** | Direct hypothesis testing results | Supporting material (if included) |

---

## Why Both Are Listed in README

### 1. **Clear Separation of Concerns**

The README clearly distinguishes:
- **Primary experiments** (H1, H2, H3) - Required for praxis validation
- **Optional pipeline** - For demonstration and operational artifacts

This separation helps readers understand:
- What is **required** to validate your hypotheses
- What is **optional** for demonstration purposes

### 2. **Transparency and Reproducibility**

By listing both, you demonstrate:
- **Transparency**: You show all components of your system, not just the core experiments
- **Reproducibility**: Others can reproduce both the research validation AND the operational artifacts
- **Completeness**: You show the full end-to-end pipeline, not just hypothesis testing

### 3. **Praxis Demonstration**

The rebuild pipeline generates **concrete operational artifacts** (risk registers) that:
- Demonstrate how your system works in practice
- Show the integration of H1/H2/H3 components
- Provide tangible outputs for praxis defense

---

## Defense Strategy for Your Praxis

### When Asked: "Why do you have both?"

**Answer**: 
"My praxis has two complementary components:

1. **Canonical H1/H2/H3 experiments** validate my research hypotheses and produce the primary quantitative evidence. These are the core scientific contributions.

2. **The optional rebuild pipeline** generates operational artifacts (risk registers) that demonstrate the end-to-end system in practice. It is read-only with respect to the canonical experiments and serves demonstration purposes.

This separation ensures that:
- My hypothesis validation results remain unchanged and reproducible
- I can demonstrate the operational value of my system through concrete risk registers
- The scientific rigor of my core experiments is preserved while allowing for practical demonstration"

### When Asked: "Which one should I use?"

**Answer**:
"For **hypothesis validation** (required for praxis):
- Use the canonical H1/H2/H3 experiments
- These produce the primary research results

For **operational demonstration** (optional):
- Use the rebuild pipeline
- This generates risk registers and per-sample outputs for demonstration"

### When Asked: "Why is the rebuild pipeline optional?"

**Answer**:
"The rebuild pipeline is optional because:
- It does **not test hypotheses** - it generates operational artifacts
- The canonical H1/H2/H3 experiments already validate all research hypotheses
- The rebuild pipeline serves **demonstration purposes** (showing risk registers, per-sample outputs)
- It is explicitly marked as 'optional' and 'post-hoc analysis' in the documentation
- It does not modify or affect the canonical experiment results"

---

## Recommended Praxis Structure

### In Your Dissertation/Praxis Document:

1. **Primary Results Section**: Use results from canonical H1/H2/H3 experiments
   - These are your **primary research contributions**
   - These validate your hypotheses

2. **Operational Demonstration Section** (Optional): Reference the rebuild pipeline outputs
   - Show example risk registers
   - Demonstrate end-to-end system operation
   - This is **supporting material**, not primary research

### In Your Defense:

- **Emphasize**: Canonical H1/H2/H3 experiments are the core scientific contribution
- **Explain**: Rebuild pipeline is for operational demonstration and is clearly marked as optional
- **Demonstrate**: Show that rebuild pipeline does not modify canonical results (read-only)

---

## Summary

**For Your Praxis**:
- ✅ **Canonical H1/H2/H3 experiments** are **REQUIRED** - these validate your hypotheses
- ⚠️ **Optional rebuild pipeline** is **OPTIONAL** - this demonstrates operational artifacts

**Both are listed in README because**:
1. They serve different purposes (research validation vs. operational demonstration)
2. They are clearly separated and labeled
3. The rebuild pipeline is explicitly marked as "optional" and "post-hoc analysis"
4. This provides transparency and completeness

**You can defend this by**:
- Explaining the clear separation of concerns
- Emphasizing that canonical experiments are the primary research contribution
- Showing that rebuild pipeline is read-only and serves demonstration purposes
- Demonstrating that both are clearly documented and labeled

