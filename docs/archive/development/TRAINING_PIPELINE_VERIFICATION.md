# Training Pipeline Verification

**Date:** 2025-12-10  
**Status:** ✅ All requirements verified and implemented

---

## Requirements Coverage

### ✅ 1. Model Training Options

#### LightGBM (Histogram-based, Default Regularization, GOSS Off)
- **Location:** `aicra/pipelines/training.py` (lines 85-119)
- **Implementation:  - `boosting_type="gbdt"` - Histogram-based tree learner (default)
  - `force_col_wise=True` - Uses histogram-based algorithm
  - GOSS explicitly OFF (commented: "GOSS is explicitly OFF per constraints")
  - Default regularization (no custom `lambda_l1`/`lambda_l2` parameters)
  - Configurable via `aicra/config.py`: `goss: bool = False`

#### Small FFNN Option
- **Location:** `aicra/pipelines/training.py` (lines 131-176)
- **Implementation:  - Small architecture: `input_dim → 128 → 64 → 2`
  - Dropout (0.2) for regularization
  - Focal loss (α=0.75, γ=2.0) for class imbalance
  - Bagged ensemble support

---

### ✅ 2. PE Static Features

**Location:** `aicra/pipelines/features_pe.py`

#### Byte Histograms
- **Method:** `_extract_byte_histogram()` (lines 34-51)
- **Features:** 256-bin byte histogram (normalized to probabilities)
- **Output:** `byte_hist_000` through `byte_hist_255`

#### PE Headers
- **Method:** `_extract_pe_headers()` (lines 53-122)
- **Features:  - `pe_machine` - Machine type
  - `pe_num_sections` - Number of sections
  - `pe_timestamp` - Compilation timestamp
  - `pe_entry_point` - Entry point address
  - `pe_magic` - Optional header magic
  - `pe_section_count` - Section count
  - `pe_section_size_mean/std/max` - Section size statistics
  - `pe_section_flags_mean` - Section characteristics

#### Section Entropy
- **Method:** `_extract_entropy_stats()` (lines 124-172)
- **Features:  - `entropy_overall` - Overall file entropy
  - `entropy_section_mean/median/max/std` - Section entropy statistics
  - `entropy_section_count` - Number of sections with entropy data

#### Integration
- **Location:** `aicra/pipelines/training.py` (lines 55-63)
- PE features are extracted and combined with EMBER features:
  ```python
  pe_features = build_pe_features(train_data.file_paths)
  X = np.hstack([train_data.features.values, pe_features.values])
  ```

---

### ✅ 3. Output p(ransomware)

**Location:** `aicra/models/lightgbm.py` and `aicra/pipelines/training.py`

- Models output `p(ransomware)` via `predict_proba()` method
- Returns probability of positive class (ransomware = 1)
- Bagged models average predictions across ensemble

---

### ✅ 4. Calibration to Susceptibility Score S∈[0,1]

**Location:** `aicra/pipelines/calibration.py`

#### Platt Scaling (Sigmoid)
- **Class:** `PlattCalibrator` (lines 250-275)
- Uses logistic regression on logits
- Output: Calibrated probabilities S∈[0,1]

#### Isotonic Regression
- **Class:** `IsotonicCalibrator` (lines 277-295)
- Non-parametric monotonic calibration
- Output: Calibrated probabilities S∈[0,1]

#### Auto Selection
- **Method:** `_select_best_calibration_method()` (lines 151-203)
- Automatically selects best method via cross-validation Brier score

#### Application in Pipeline
- **Location:** `aicra/pipelines/test_runner.py` (lines 526-568)
- **Updated:** Register generation now applies calibration:
  ```python
  # Generate raw predictions p(ransomware)
  y_prob_raw = y_prob[:, 1]
  
  # Apply calibration to produce Susceptibility Score S∈[0,1]
  calibrator = joblib.load(calibrator_path)
  susceptibility_scores = calibrator.transform(y_prob_raw)  # S∈[0,1]
  
  # Use calibrated scores in register
  register_df["probability"] = susceptibility_scores
  ```

---

## End-to-End Flow

1. **Training:   - Extract PE static features (byte histograms, PE headers, section entropy)
   - Combine with EMBER features
   - Train LightGBM (hist, default reg, GOSS off) or small FFNN
   - Output: Model that predicts `p(ransomware)`

2. **Calibration:   - Load model and generate raw `p(ransomware)` predictions
   - Apply Platt or Isotonic calibration
   - Output: Calibrated **Susceptibility Score S∈[0,1]3. **Register Generation:   - Use calibrated susceptibility scores (not raw probabilities)
   - Compute Expected Loss = S × $5,000,000
   - Bucket into High/Medium/Low risk
   - Attach prescriptive controls

---

## Verification Checklist

- ✅ LightGBM uses histogram-based tree learner (`boosting_type="gbdt"`)
- ✅ GOSS is explicitly OFF
- ✅ Default regularization (no custom lambda parameters)
- ✅ Small FFNN option available
- ✅ PE static features extracted:
  - ✅ Byte histograms (256 bins)
  - ✅ PE headers (machine, sections, timestamps, entry point, etc.)
  - ✅ Section entropy (overall and per-section statistics)
- ✅ Models output `p(ransomware)` via `predict_proba()`
- ✅ Calibration applied (Platt or Isotonic)
- ✅ Calibrated output is **Susceptibility Score S∈[0,1]- ✅ Registers use calibrated susceptibility scores (not raw probabilities)

--**All Requirements Met** ✅

