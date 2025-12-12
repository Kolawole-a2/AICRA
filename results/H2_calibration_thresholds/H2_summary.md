# H2 Calibration and Thresholding Experiment Results

## Hypothesis

Calibration and cost-aware thresholding produce more decision-aligned susceptibility scores than uncalibrated F1-optimized thresholds.

## Calibration Results

- **Brier Score (uncalibrated)**: 0.0426
- **Brier Score (calibrated)**: 0.0500
- **Brier Improvement**: -0.0074
- **ECE (uncalibrated)**: 0.0066
- **ECE (calibrated)**: 0.0457
- **ECE Improvement**: -0.0391

## Threshold Comparison

### F1-Optimized Threshold

**Uncalibrated:**
- Threshold: 0.4586
- F1: 0.9416
- Expected Loss: 0.3027

**Calibrated:**
- Threshold: 0.2268
- F1: 0.9416
- Expected Loss: 0.3027

### Cost-Optimized Threshold

Cost structure: FN=10.0, FP=1.0

**Uncalibrated:**
- Threshold: 0.1040
- Expected Loss: 0.1729
- Precision: 0.8213
- Recall: 0.9854

**Calibrated:**
- Threshold: 0.0100
- Expected Loss: 0.2148
- Precision: 0.9047
- Recall: 0.9654

## Conclusion

✗ H2 is **partially supported**: Results show mixed evidence.
