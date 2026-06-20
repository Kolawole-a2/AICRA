> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# AICRA Requirements Audit Report

**Date:** 2025-01-27  
**Auditor:** AI Pair Programmer  
**Scope:** Complete codebase audit for four core requirements  
**Status:** Analysis Only (No Code Modifications)

---

## 1. Repository Scan Summary

### Key Directories and Files

**Evaluation & Metrics:- `aicra/experiments/h1_classification.py` - H1 experiment with AUROC, PR-AUC, Brier, ECE, Lift@k
- `aicra/experiments/h2_calibration_thresholds.py` - H2 experiment with calibration and cost-sensitive thresholds
- `aicra/experiments/h3_evaluation.py` - H3 experiment with DAC metrics
- `aicra/utils/evaluate.py` - Standalone evaluation script with metrics computation
- `aicra/core/evaluation.py` - Core evaluation functions (cost_sensitive_threshold, compute_lift_at_k, ECE)

**Training & Calibration:- `aicra/pipelines/training.py` - Training pipeline with bagging (seeds parameter)
- `aicra/core/calibration.py` - Calibrator class (Platt/Isotonic)
- `aicra/pipelines/calibration.py` - CalibrationPipeline with auto-selection and post-ensemble checks
- `aicra/utils/train_lightgbm.py` - Training script with focal loss support

**Risk Bucketing & Controls:- `aicra/register.py` - Risk register generation with bucketing (0.33, 0.66 thresholds)
- `aicra/utils/policy_writer.py` - Policy writer with risk buckets and controls
- `data/lookups/risk_bucket_controls.yaml` - High/Medium/Low control definitions
- `aicra/pipelines/mapping.py` - ATT&CK→D3FEND mapping pipeline
- `deterministic_lookup.csv` - Deterministic ATT&CK→D3FEND lookup table

**Banking Narratives & Policy:- `aicra/pipelines/policy.py` - PolicyPipeline with cost-sensitive threshold optimization
- `aicra/pipelines/cost_optimization.py` - CostOptimizer class
- `policies/policy.json` - Example policy JSON output
- `aicra/experiments/h2_calibration_thresholds.py` - Expected Loss computation

**Data Loading & Splits:- `aicra/utils/data_loader.py` - Data loading (no explicit time-ordered split in main function)
- `aicra/pipelines/evaluation.py` - EvaluationPipeline with `_time_ordered_split()` method
- `aicra/core/data.py` - Dataset class with timestamps support

---

## 2. Requirement-by-Requirement Status

### REQUIREMENT 1: Evaluation Protocol & Metrics

#### Status: **PARTIAL** ⚠️

#### Evidence:

**✅ Time-Ordered Split:- **Location:** `aicra/pipelines/evaluation.py`, lines 176-187
  ```python
  def _time_ordered_split(self, timestamps: pd.Series) -> tuple[np.ndarray, np.ndarray]:
      sorted_indices = timestamps.argsort()
      split_point = int(0.8 * len(sorted_indices))
      train_idx = sorted_indices[:split_point]
      test_idx = sorted_indices[split_point:]
  ```
- **Issue:** This method exists but is **not consistently used** in main experiments (H1, H2, H3)
- **H1 Experiment:** Uses `load_ember_2024()` which does **random split** (see `aicra/utils/data_loader.py`, lines 59-67)
- **H2 Experiment:** Uses same random split from `load_ember_2024()`
- **H3 Experiment:** Loads pre-computed risk scores from CSV files (no split logic visible)

**✅ Out-of-Family Generalization:- **Location:** `aicra/experiments/h1_classification.py`, lines 191-214
  ```python
  if hasattr(test_data, 'families') and test_data.families is not None:
      families_clean = pd.Series(families).fillna('unknown').astype(str)
      unique_families = np.unique(families_clean)
      for family in unique_families:
          family_mask = families_clean == family
          if family_mask.sum() > 10:
              family_auroc = roc_auc_score(y_true_test[family_mask], y_prob_test[family_mask])
  ```
- **Issue:** This evaluates **per-family AUROC** but does **NOT** implement a true out-of-family test (train on some families, test on held-out families)
- **Alternative:** `aicra/pipelines/evaluation.py`, lines 189-229 has `_evaluate_out_of_family_generalization()` that holds out one family, but it's not used in main experiments

**✅ Metrics Coverage:- **AUROC:** ✅ `aicra/experiments/h1_classification.py:173`, `aicra/core/evaluation.py:44`
- **PR-AUC:** ✅ `aicra/experiments/h1_classification.py:174`, `aicra/core/evaluation.py:45`
- **Brier Score:** ✅ `aicra/experiments/h1_classification.py:175`, `aicra/core/evaluation.py:46`
- **ECE:** ✅ `aicra/experiments/h1_classification.py:176`, `aicra/core/evaluation.py:27-30`
- **Lift@k:** ✅ `aicra/experiments/h1_classification.py:181-183` (Lift@1%, 5%, 10%)
- **Confusion Matrix:** ✅ `aicra/core/evaluation.py:50` (computed at threshold)

**⚠️ Operational Decision Threshold (FN≫FP):- **Location:** `aicra/core/evaluation.py`, lines 54-67 (`cost_sensitive_threshold()`)
- **Location:** `aicra/pipelines/cost_optimization.py`, lines 44-93 (`CostOptimizer.optimize_threshold()`)
- **Issue:** Cost-sensitive threshold optimization exists, but:
  1. **Not consistently applied** in H1/H2 experiments (H1 uses fixed `operational_threshold=0.5`)
  2. **No explicit FN≫FP tuning** in main evaluation scripts (default cost ratios may not reflect banking preference)
  3. **Confusion matrix** is computed at threshold, but threshold may not be optimized for FN≫FP

#### Gaps & Risks:

1. **Time-ordered split not enforced:** Main experiments use random splits, risking temporal leakage
2. **Out-of-family test incomplete:** Per-family evaluation exists, but no true train/test family split
3. **FN≫FP threshold not explicitly tuned:** Cost-sensitive optimization exists but may not be configured for banking (FN cost >> FP cost)
4. **Confusion matrix at wrong threshold:** May be computed at F1-optimal or default threshold, not FN≫FP-optimal

#### Proposed Changes (Do NOT Apply):

**File:** `aicra/utils/data_loader.py`
- **Function:** `load_ember_2024()`
- **Change:** Add `time_ordered: bool = True` parameter and implement time-based split:
  ```python
  def load_ember_2024(
      return_val: bool = False,
      val_split: float = 0.1,
      seed: int = 42,
      time_ordered: bool = True,  # NEW
  ) -> Tuple[Dataset, Dataset] | Tuple[Dataset, Dataset, Dataset]:
      # ... load data ...
      if time_ordered and train.timestamps is not None:
          # Sort by timestamp and split chronologically
          sorted_idx = train.timestamps.argsort()
          split_point = int(len(sorted_idx) * (1 - val_split if return_val else 0.8))
          # ... create time-ordered splits ...
  ```

**File:** `aicra/experiments/h1_classification.py`
- **Function:** `run_h1_classification_experiment()`
- **Change:** Add out-of-family split logic:
  ```python
  # Before training, split families
  train_families = set(train_data.families.unique()[:len(train_data.families.unique()) // 2])
  test_families = set(test_data.families.unique()) - train_families
  # Filter test_data to only include held-out families
  test_oof = test_data[test_data.families.isin(test_families)]
  # Evaluate on test_oof
  ```

**File:** `aicra/experiments/h1_classification.py`
- **Function:** `run_h1_classification_experiment()`
- **Change:** Replace fixed threshold with FN≫FP-optimized threshold:
  ```python
  from ..core.evaluation import cost_sensitive_threshold
  # Tune threshold for banking (FN cost >> FP cost)
  banking_threshold = cost_sensitive_threshold(
      y_true_test, y_prob_test, 
      cost_fn=100.0,  # High cost for false negatives
      cost_fp=1.0     # Low cost for false positives
  )
  # Compute confusion matrix at banking_threshold
  y_pred_test = (y_prob_test >= banking_threshold).astype(int)
  cm = confusion_matrix(y_true_test, y_pred_test)
  metrics["operational_threshold"] = banking_threshold
  metrics["confusion_matrix"] = {"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])}
  ```

---

### REQUIREMENT 2: Shift/Noise Mitigation & Calibration

#### Status: **FULL** ✅

#### Evidence:

**✅ Bagged Seeds/Models:- **Location:** `aicra/pipelines/training.py`, lines 85-119
  ```python
  def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, seeds: int) -> Any:
      model_seeds = np.random.randint(0, 2**31, seeds).tolist()
      models = []
      for seed in model_seeds:
          model = LGBMClassifier(..., random_state=seed, ...)
          model.fit(X_df, y)
          models.append(model)
      return BaggedLightGBM(models=models)
  ```
- **Config:** `config/h1_config.yaml:16` sets `seeds: 5`
- **FFNN:** `aicra/pipelines/training.py:131-176` also implements bagging with multiple seeds

**✅ Calibration Checks:- **Location:** `aicra/pipelines/calibration.py`, lines 27-82
  ```python
  def run(self, train_data, val_data, y_prob_train, y_prob_val, method="auto", ...):
      calibrator = self._create_calibrator(method)
      calibrator.fit(y_prob_train, train_data.labels.values)
      y_prob_calibrated = calibrator.transform(y_prob_val)
      brier_uncalibrated = brier_score_loss(val_data.labels.values, y_prob_val)
      brier_calibrated = brier_score_loss(val_data.labels.values, y_prob_calibrated)
  ```
- **Post-calibration ECE:** `aicra/pipelines/calibration.py:84-131` implements `_post_ensemble_calibration_check()` that monitors ECE
- **H2 Experiment:** `aicra/experiments/h2_calibration_thresholds.py:219-244` fits calibrator and evaluates Brier/ECE before/after

**✅ Robust Loss Functions:- **Focal Loss:** `aicra/pipelines/training.py:202-220`
  ```python
  class FocalLoss:
      def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
          self.alpha = alpha  # α > 0.5 ✅
          self.gamma = gamma  # γ ≈ 2 ✅
  ```
- **Usage:** `aicra/pipelines/training.py:154` uses `FocalLoss(alpha=0.75, gamma=2.0)` for FFNN
- **Class-Balanced Loss:** `aicra/pipelines/training.py:107` uses `class_weight="balanced"` for LightGBM
- **Alternative:** `aicra/utils/train_lightgbm.py:13-17` implements `focal_loss_sample_weight()` with `alpha=0.75, gamma=2.0`

#### Gaps & Risks:

**None identified.** All three components (bagging, calibration, robust loss) are fully implemented.

#### Proposed Changes (Do NOT Apply):

**None required.** Requirement is fully satisfied.

---

### REQUIREMENT 3: Risk Bucketing + Prescriptive Controls (ATT&CK/D3FEND-Informed)

#### Status: **PARTIAL** ⚠️

#### Evidence:

**✅ Risk Bucketing:- **Location:** `aicra/register.py`, lines 60-61
  ```python
  bins=[0.0, 0.33, 0.66, 1.0],
  labels=["Low", "Medium", "High"],
  ```
- **Location:** `aicra/pipelines/smoke.py`, lines 701-704 (same thresholds)
- **Usage:** `aicra/utils/policy_writer.py:77` computes `expected_loss = susceptibility * impact`

**✅ Control Definitions:- **Location:** `data/lookups/risk_bucket_controls.yaml`
  ```yaml
  High:
    controls:
      - "Enable Attack Surface Reduction (ASR) rules"
      - "Implement Local Administrator Password Solution (LAPS)"
      - "Ensure immutable and offline backups"
      - "Deploy Application Allowlisting"
  Medium:
    controls:
      - "Enforce Multi-Factor Authentication (MFA)"
      - "Implement strict EDR policies"
  Low:
    controls:
      - "Maintain continuous security monitoring"
  ```

**✅ ATT&CK→D3FEND Mapping:- **Location:** `aicra/pipelines/mapping.py` - `MappingPipeline` class
- **Location:** `deterministic_lookup.csv` - Deterministic lookup table
- **Location:** `aicra/utils/policy_writer.py:42-67` - Loads attack_map and d3fend_graph, maps families → techniques → controls

**⚠️ Binding Buckets to Controls:- **Location:** `aicra/utils/policy_writer.py:77-79` - Computes expected_loss but does **NOT** attach controls to buckets
- **Location:** `aicra/register.py:34-66` - Computes buckets but does **NOT** attach controls from `risk_bucket_controls.yaml`
- **Issue:** Risk buckets exist, controls exist, but they are **not automatically bound** in the register generation pipeline

**⚠️ ATT&CK-Informed Playbooks:- **Location:** `aicra/utils/policy_writer.py:83-84` - Writes register JSON but does **NOT** generate playbooks
- **Issue:** No code generates ATT&CK-informed playbooks (JSON/YAML/text reports) that combine buckets + techniques + controls

#### Gaps & Risks:

1. **Buckets not bound to controls:** Register generation does not automatically attach controls from `risk_bucket_controls.yaml` to High/Medium/Low buckets
2. **No playbook generation:** Missing code to generate ATT&CK-informed playbooks/recommendations
3. **Controls not surfaced:** Controls are computed per-sample but not aggregated into bucket-level recommendations

#### Proposed Changes (Do NOT Apply):

**File:** `aicra/register.py`
- **Function:** `compute_register()`
- **Change:** Add control attachment logic:
  ```python
  import yaml
  from pathlib import Path
  
  def compute_register(df: pd.DataFrame, policy: Policy, impact_column: str | None = None) -> pd.DataFrame:
      # ... existing bucketing logic ...
      
      # Load risk bucket controls
      settings = get_settings()
      controls_yaml = settings.data_dir / "lookups" / "risk_bucket_controls.yaml"
      with open(controls_yaml, 'r') as f:
          bucket_controls = yaml.safe_load(f)['risk_buckets']
      
      # Attach controls to buckets
      def get_controls_for_bucket(bucket):
          return bucket_controls.get(bucket, {}).get('controls', [])
      
      df['prescriptive_controls'] = df['susceptibility_bucket'].apply(get_controls_for_bucket)
      
      return df
  ```

**File:** `aicra/pipelines/mapping.py` (or new file `aicra/pipelines/playbook.py`)
- **Function:** New function `generate_attack_playbook()`
- **Change:** Create playbook generator:
  ```python
  def generate_attack_playbook(
      register_df: pd.DataFrame,
      mapping_pipeline: MappingPipeline,
      output_path: Path
  ) -> Dict[str, Any]:
      """
      Generate ATT&CK-informed playbook from risk register.
      
      For each High/Medium risk asset:
      1. Extract technique_id
      2. Lookup ATT&CK technique details
      3. Map to D3FEND controls
      4. Generate prescriptive actions
      """
      playbook = {
          "version": "1.0.0",
          "timestamp": datetime.now().isoformat(),
          "high_risk_assets": [],
          "medium_risk_assets": [],
          "recommendations": []
      }
      
      # Process High risk assets
      high_risk = register_df[register_df['susceptibility_bucket'] == 'High']
      for _, row in high_risk.iterrows():
          techniques = mapping_pipeline.get_techniques_for_family(row['family'])
          controls = mapping_pipeline.get_controls_for_techniques(techniques)
          playbook['high_risk_assets'].append({
              "asset_id": row['asset_id'],
              "risk_score": row['risk_score'],
              "techniques": techniques,
              "prescriptive_controls": controls,
              "expected_loss": row['expected_loss']
          })
      
      # ... similar for Medium ...
      
      # Save playbook
      with open(output_path, 'w') as f:
          json.dump(playbook, f, indent=2)
      
      return playbook
  ```

**File:** `aicra/utils/policy_writer.py`
- **Function:** `main()`
- **Change:** Add `--generate-playbook` flag and call playbook generator:
  ```python
  parser.add_argument("--generate-playbook", action="store_true")
  # ... after writing register ...
  if args.generate_playbook:
      from ..pipelines.playbook import generate_attack_playbook
      playbook_path = Path(args.out).parent / "attack_playbook.json"
      generate_attack_playbook(df, mapping_pipeline, playbook_path)
  ```

---

### REQUIREMENT 4: Banking Narratives: Expected Loss, Cost-Sensitive Thresholds, Auditable Policy

#### Status: **PARTIAL** ⚠️

#### Evidence:

**✅ Expected Loss Computation:- **Location:** `aicra/utils/policy_writer.py:77`
  ```python
  df["expected_loss"] = df["susceptibility"] * args.impact
  ```
- **Location:** `aicra/experiments/h2_calibration_thresholds.py:88-111`
  ```python
  def compute_expected_loss(y_true, y_prob, threshold, cost_fn=10.0, cost_fp=1.0):
      # Expected Loss = p(ransomware) * impact_cost
      total_loss = (cost_fn * fn) + (cost_fp * fp)
      return float(total_loss / total_samples)
  ```
- **Issue:** Expected Loss is computed, but `impact` is a **single default value** (`args.impact`), not parameterized per asset/segment/scenario

**✅ Cost-Sensitive Threshold Optimization:- **Location:** `aicra/core/evaluation.py:54-67`
  ```python
  def cost_sensitive_threshold(y_true, y_prob, cost_fn, cost_fp):
      # Minimize: fn * cost_fn + fp * cost_fp
      for t in thresholds:
          cost = fn * cost_fn + fp * cost_fp
          if cost < best_cost:
              best_t = t
  ```
- **Location:** `aicra/pipelines/policy.py:84-128` - `optimize_cost_sensitive_threshold()`
- **Location:** `aicra/pipelines/cost_optimization.py:44-93` - `CostOptimizer.optimize_threshold()`
- **Usage:** `aicra/experiments/h2_calibration_thresholds.py:256-259` uses cost-sensitive threshold

**✅ Policy JSON Persistence:- **Location:** `aicra/pipelines/policy.py:130-180`
  ```python
  @dataclass
  class Policy:
      threshold: float
      cost_false_negative: float
      cost_false_positive: float
      impact_default: float
      version: str = "1.0.0"
      timestamp: str = ""
      model_id: str = ""
      calibration_id: str = ""
      lookup_versions: Dict[str, str] = None
  ```
- **Location:** `policies/policy.json` - Example output:
  ```json
  {
    "threshold": 0.5,
    "cost_false_negative": 100.0,
    "cost_false_positive": 5.0,
    "impact_default": 1000000.0,
    "version": "1.0.0",
    "timestamp": "2025-10-18T21:11:21.961966"
  }
  ```

**⚠️ Impact Parameterization:- **Issue:** `impact_default` is a single value, not parameterized per:
  - Asset type (endpoint vs server vs database)
  - Customer segment (retail vs commercial)
  - Scenario (data breach vs operational disruption)
- **Location:** `aicra/utils/policy_writer.py:24` - `--impact` is a single float, not a table/mapping

**⚠️ Banking-Specific Parameters:- **Issue:** Policy JSON does not include:
  - Impact ranges (min/max per asset type)
  - Loss per endpoint/scenario
  - Scenario tags (e.g., "data_breach", "ransomware_encryption")
  - Customer segment mappings

#### Gaps & Risks:

1. **Impact not parameterized:** Expected Loss uses single `impact_default`, not per-asset/segment/scenario
2. **Banking parameters missing:** Policy JSON lacks impact ranges, scenario tags, segment mappings
3. **Cost ratios may not reflect banking:** Default `cost_fn=10.0, cost_fp=1.0` may not be banking-appropriate (should be higher ratio, e.g., 100:1)

#### Proposed Changes (Do NOT Apply):

**File:** `aicra/pipelines/policy.py`
- **Class:** `Policy`
- **Change:** Extend Policy dataclass:
  ```python
  @dataclass
  class Policy:
      threshold: float
      cost_false_negative: float
      cost_false_positive: float
      impact_default: float
      # NEW: Banking-specific parameters
      impact_ranges: Dict[str, Dict[str, float]] = None  # {"endpoint": {"min": 1e5, "max": 1e7}, ...}
      scenario_impacts: Dict[str, float] = None  # {"data_breach": 5e6, "ransomware_encryption": 1e7, ...}
      customer_segment_impacts: Dict[str, float] = None  # {"retail": 1e5, "commercial": 1e7, ...}
      loss_per_endpoint: float = 10000.0
      scenario_tags: List[str] = None
      version: str = "1.0.0"
      timestamp: str = ""
      # ... existing fields ...
  ```

**File:** `aicra/utils/policy_writer.py`
- **Function:** `main()`
- **Change:** Add impact parameterization:
  ```python
  parser.add_argument("--impact-table", type=Path, help="CSV with asset_id,impact columns")
  parser.add_argument("--scenario", type=str, default="ransomware_encryption")
  
  # Load impact table if provided
  if args.impact_table and args.impact_table.exists():
      impact_df = pd.read_csv(args.impact_table)
      impact_dict = dict(zip(impact_df['asset_id'], impact_df['impact']))
      df['expected_loss'] = df.apply(
          lambda row: row['susceptibility'] * impact_dict.get(row['asset_id'], args.impact),
          axis=1
      )
  else:
      # Use scenario-based impact
      scenario_impacts = {
          "data_breach": 5_000_000,
          "ransomware_encryption": 10_000_000,
          "operational_disruption": 2_000_000
      }
      impact = scenario_impacts.get(args.scenario, args.impact)
      df['expected_loss'] = df['susceptibility'] * impact
  ```

**File:** `aicra/pipelines/policy.py`
- **Function:** `create_policy()`
- **Change:** Add banking-specific parameter loading:
  ```python
  def create_policy(
      self,
      y_true: np.ndarray,
      y_prob: np.ndarray,
      model_id: str = "",
      calibration_id: str = "",
      impact_table_path: Optional[Path] = None,
      banking_config_path: Optional[Path] = None  # NEW
  ) -> Policy:
      # ... existing threshold optimization ...
      
      # Load banking-specific config
      if banking_config_path and banking_config_path.exists():
          with open(banking_config_path, 'r') as f:
              banking_config = yaml.safe_load(f)
      else:
          banking_config = {
              "impact_ranges": {
                  "endpoint": {"min": 100000, "max": 10000000},
                  "server": {"min": 500000, "max": 50000000}
              },
              "scenario_impacts": {
                  "data_breach": 5000000,
                  "ransomware_encryption": 10000000
              },
              "customer_segment_impacts": {
                  "retail": 100000,
                  "commercial": 10000000
              }
          }
      
      policy = Policy(
          threshold=optimal_threshold,
          cost_false_negative=self.settings.cost_fn,  # Should be 100.0+ for banking
          cost_false_positive=self.settings.cost_fp,   # Should be 1.0 for banking
          impact_default=self.settings.impact_default,
          impact_ranges=banking_config.get("impact_ranges"),
          scenario_impacts=banking_config.get("scenario_impacts"),
          customer_segment_impacts=banking_config.get("customer_segment_impacts"),
          loss_per_endpoint=banking_config.get("loss_per_endpoint", 10000.0),
          # ... existing fields ...
      )
  ```

---

## 3. Validation Checklist

### Requirement 1: Evaluation Protocol & Metrics

- [ ] **Time-Ordered Split:  - [ ] Run: `python -m aicra.experiments.h1_classification --output results/H1_test --time-ordered`
  - [ ] Check: `results/H1_test/metrics.json` - verify `train_max_timestamp < test_min_timestamp`
  - [ ] **Gap:** H1 does not accept `--time-ordered` flag (needs implementation)

- [ ] **Out-of-Family Test:  - [ ] Run: `python -m aicra.experiments.h1_classification --output results/H1_test`
  - [ ] Check: `results/H1_test/metrics.json` - verify `oof_auroc_mean` exists
  - [ ] **Gap:** Current implementation evaluates per-family, not true held-out family split

- [ ] **Metrics Verification:  - [ ] Run: `python -m aicra.experiments.h1_classification --output results/H1_test`
  - [ ] Check: `results/H1_test/metrics.json` contains:
    - `auroc` ✅
    - `pr_auc` ✅
    - `brier_score` ✅
    - `ece` ✅
    - `lift_at_1pct`, `lift_at_5pct`, `lift_at_10pct` ✅
    - `confusion` (with tn, fp, fn, tp) ✅

- [ ] **FN≫FP Threshold:  - [ ] Run: `python -m aicra.experiments.h2_calibration_thresholds --cost-fn 100.0 --cost-fp 1.0`
  - [ ] Check: `results/H2_calibration_thresholds/metrics.json` - verify `cost_optimized.calibrated.threshold` is tuned
  - [ ] **Gap:** H1 does not use cost-optimized threshold (uses fixed 0.5)

### Requirement 2: Shift/Noise Mitigation & Calibration

- [ ] **Bagged Models:  - [ ] Check: `config/h1_config.yaml` - verify `seeds: 5`
  - [ ] Run: `python -m aicra.experiments.h1_classification`
  - [ ] Check: `models/h1_lgbm.joblib` - verify it's a `BaggedLightGBM` with 5 models

- [ ] **Calibration:  - [ ] Run: `python -m aicra.experiments.h2_calibration_thresholds`
  - [ ] Check: `results/H2_calibration_thresholds/metrics.json` - verify:
    - `calibration.brier_uncalibrated` > `calibration.brier_calibrated`
    - `calibration.ece_uncalibrated` > `calibration.ece_calibrated`

- [ ] **Robust Loss:  - [ ] Check: `aicra/pipelines/training.py:154` - verify `FocalLoss(alpha=0.75, gamma=2.0)`
  - [ ] Check: `aicra/pipelines/training.py:107` - verify `class_weight="balanced"` for LightGBM

### Requirement 3: Risk Bucketing + Prescriptive Controls

- [ ] **Risk Buckets:  - [ ] Run: `python -m aicra.register` (if exists) or check `register/risk_register_main.json`
  - [ ] Check: Verify `susceptibility_bucket` column has "High", "Medium", "Low"
  - [ ] Check: Verify thresholds are 0.33 and 0.66

- [ ] **Controls Binding:  - [ ] Check: `data/lookups/risk_bucket_controls.yaml` exists
  - [ ] Check: `register/risk_register_main.json` - verify `prescriptive_controls` column exists
  - [ ] **Gap:** Controls may not be automatically attached (needs verification)

- [ ] **ATT&CK→D3FEND Mapping:  - [ ] Check: `deterministic_lookup.csv` exists with `technique_id,control_id` columns
  - [ ] Run: `python -m aicra.pipelines.mapping` (if exists)
  - [ ] Check: Verify register contains `attack_techniques` and `d3fend_controls` columns

- [ ] **Playbook Generation:  - [ ] **Gap:** No playbook generation script found - needs implementation

### Requirement 4: Banking Narratives

- [ ] **Expected Loss:  - [ ] Check: `register/risk_register_main.json` - verify `expected_loss` column exists
  - [ ] Verify: `expected_loss = susceptibility * impact` formula
  - [ ] **Gap:** Impact is single value, not parameterized

- [ ] **Cost-Sensitive Threshold:  - [ ] Run: `python -m aicra.experiments.h2_calibration_thresholds --cost-fn 100.0 --cost-fp 1.0`
  - [ ] Check: `results/H2_calibration_thresholds/metrics.json` - verify `cost_optimized.calibrated.threshold`

- [ ] **Policy JSON:  - [ ] Check: `policies/policy.json` exists
  - [ ] Verify JSON contains:
    - `threshold` ✅
    - `cost_false_negative` ✅
    - `cost_false_positive` ✅
    - `impact_default` ✅
    - `version` ✅
    - `timestamp` ✅
  - [ ] **Gap:** Missing `impact_ranges`, `scenario_impacts`, `customer_segment_impacts`

---

## Summary

| Requirement | Status | Key Gaps |
|------------|--------|----------|
| **1. Evaluation Protocol & Metrics** | **PARTIAL** | Time-ordered split not enforced; Out-of-family test incomplete; FN≫FP threshold not consistently applied |
| **2. Shift/Noise Mitigation & Calibration** | **FULL** | None |
| **3. Risk Bucketing + Controls** | **PARTIAL** | Buckets not bound to controls; No playbook generation |
| **4. Banking Narratives** | **PARTIAL** | Impact not parameterized; Banking-specific parameters missing from policy JSON |

**Overall Status:** **PARTIAL** - Core functionality exists but requires hardening for production banking use.

--**End of Audit Report