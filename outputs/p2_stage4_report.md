# Person 2 — Stage 4 Final Validity Model Report

**Owner**: Person 2 (Data Quality, Anomaly Detection & Classification Lead)  
**Project**: CPRI State-Level Hackathon — Task 01  
**Scope**: P2-Stage 4 Final Valid vs Invalid Classification Pipeline  
**Date**: September 2026  

---

## 1. Executive Summary

P2-Stage 4 constructs the final, reproducible **Valid vs Invalid classifier** for CPRI electrical test screening.

Using leakage-free **5-fold Stratified Cross-Validation** on 1,000 historical training records, followed by **out-of-fold decision threshold tuning**, the final validity classifier achieves:

- **Final Model**: `Random Forest (Balanced)`
- **Final Feature Set**: `Set C (Combined: Raw + Quality + Contextual)`
- **Optimal Decision Threshold**: **`0.35`**
- **Invalid F1 Score**: **`0.9344`**
- **Invalid Recall**: **`0.9030`** (Detects 121.0/134 Invalid tests)
- **Invalid Precision**: **`0.9680`**
- **Balanced Accuracy**: **`0.9492`**
- **Accuracy**: **`0.9830`**
- **ROC-AUC**: **`0.9888`**
- **PR-AUC**: **`0.9664`**

---

## 2. Explicit Methodological Disclosures

> [!IMPORTANT]
> **Mandatory Methodological Disclosures**:
> 1. **Person 1 Feature Availability Status**: Genuine Person 1 feature files were **NOT present** in the repository. We did not invent or falsely label replacement features. Validated **Person 2 Engineered Features** (`Apparent_Power_kVA`, `Sensor_Spread`, `Sensor_Mean`, `Sensor_S1_S2_Ratio`, `Sensor_S3_S4_Ratio`) were used as fallbacks and are explicitly documented as such.
> 2. **Stage 3 Anomaly Score Leakage Prevention**: Anomaly scores were **not** calculated globally for CV input. `IsolationForest` was fitted fold-by-fold strictly inside CV pipelines to generate fold-isolated features, or evaluated as a post-hoc signal.
> 3. **Threshold Selection Disclosure**: Threshold optimization was performed on out-of-fold predictions. As documented, the threshold-optimized OOF metric represents a model-selection estimate.
> 4. **Validation Methodology Limitations**: Stratified 5-fold CV evaluates historical test distribution. Generalization to unseen hardware series relies on physical range robustness.

---

## 3. Candidate Feature Sets & Constant Features

### Candidate Feature Sets:
- **Set A (Stage 2 Best)**: Raw physical inputs + active Stage 1 quality flags.
- **Set B (Person 2 Fallback Contextual)**: Raw physical inputs + Person 2 fallback engineered features.
- **Set C (Combined)**: Raw physical inputs + Stage 1 quality flags + Person 2 fallback engineered features.

### Excluded Constant Features:
- **`missing_critical_measurement`**: Constant value `False`
- **`non_finite_value`**: Constant value `False`
- **`duplicate_full_row`**: Constant value `False`
- **`duplicate_test_id`**: Constant value `False`
- **`malformed_numeric`**: Constant value `False`
- **`invalid_voltage`**: Constant value `False`
- **`invalid_current`**: Constant value `False`
- **`invalid_duration`**: Constant value `False`
- **`invalid_ambient_temp`**: Constant value `False`

---

## 4. Candidate Model Comparison Table (5-Fold Stratified CV, Default 0.50 Threshold)

| Model                              | Feature_Set                                  |   Invalid_Precision |   Invalid_Recall |   Invalid_F1 |   Invalid_F1_Std |   Macro_F1 |   Accuracy |   Balanced_Accuracy |   ROC_AUC |   PR_AUC |   CV_Folds |
|:-----------------------------------|:---------------------------------------------|--------------------:|-----------------:|-------------:|-----------------:|-----------:|-----------:|--------------------:|----------:|---------:|-----------:|
| Logistic Regression (Balanced)     | Set A (Stage 2 Best: Raw + Quality)          |              0.4    |           0.5075 |       0.4474 |           0.0614 |     0.6742 |      0.832 |              0.6948 |    0.6523 |   0.4575 |          5 |
| Random Forest (Balanced)           | Set A (Stage 2 Best: Raw + Quality)          |              0.9535 |           0.6119 |       0.7455 |           0.0819 |     0.857  |      0.944 |              0.8037 |    0.9791 |   0.9134 |          5 |
| Random Forest + Fold Anomaly Score | Set A (Stage 2 Best: Raw + Quality)          |              0.9451 |           0.6418 |       0.7644 |           0.077  |     0.8673 |      0.947 |              0.818  |    0.9565 |   0.8691 |          5 |
| Gradient Boosting                  | Set A (Stage 2 Best: Raw + Quality)          |              0.9494 |           0.5597 |       0.7042 |           0.0609 |     0.8345 |      0.937 |              0.7775 |    0.9669 |   0.8821 |          5 |
| Logistic Regression (Balanced)     | Set B (Raw + Person 2 Contextual)            |              0.8611 |           0.694  |       0.7686 |           0.0555 |     0.8684 |      0.944 |              0.8384 |    0.8633 |   0.741  |          5 |
| Random Forest (Balanced)           | Set B (Raw + Person 2 Contextual)            |              0.9397 |           0.8134 |       0.872  |           0.0296 |     0.9269 |      0.968 |              0.9027 |    0.9817 |   0.9423 |          5 |
| Random Forest + Fold Anomaly Score | Set B (Raw + Person 2 Contextual)            |              0.955  |           0.791  |       0.8653 |           0.035  |     0.9233 |      0.967 |              0.8926 |    0.9777 |   0.9359 |          5 |
| Gradient Boosting                  | Set B (Raw + Person 2 Contextual)            |              0.9811 |           0.7761 |       0.8667 |           0.0256 |     0.9242 |      0.968 |              0.8869 |    0.9741 |   0.9327 |          5 |
| Logistic Regression (Balanced)     | Set C (Combined: Raw + Quality + Contextual) |              0.9106 |           0.8358 |       0.8716 |           0.0604 |     0.9263 |      0.967 |              0.9116 |    0.9601 |   0.8854 |          5 |
| Random Forest (Balanced)           | Set C (Combined: Raw + Quality + Contextual) |              0.9913 |           0.8507 |       0.9157 |           0.0384 |     0.9518 |      0.979 |              0.9248 |    0.9888 |   0.9664 |          5 |
| Random Forest + Fold Anomaly Score | Set C (Combined: Raw + Quality + Contextual) |              0.9823 |           0.8284 |       0.8988 |           0.0427 |     0.9423 |      0.975 |              0.913  |    0.9781 |   0.9426 |          5 |
| Gradient Boosting                  | Set C (Combined: Raw + Quality + Contextual) |              0.9815 |           0.791  |       0.876  |           0.0341 |     0.9295 |      0.97  |              0.8944 |    0.9831 |   0.9453 |          5 |

---

## 5. Decision Threshold Optimization Table (Winning Model)

|   Threshold |   Invalid_Precision |   Invalid_Recall |   Invalid_F1 |   Accuracy |   Balanced_Accuracy |   TN |   FP |   FN |   TP |
|------------:|--------------------:|-----------------:|-------------:|-----------:|--------------------:|-----:|-----:|-----:|-----:|
|        0.1  |              0.4681 |           0.9851 |       0.6346 |      0.848 |              0.9059 |  716 |  150 |    2 |  132 |
|        0.15 |              0.6436 |           0.9701 |       0.7738 |      0.924 |              0.9435 |  794 |   72 |    4 |  130 |
|        0.2  |              0.7862 |           0.9328 |       0.8532 |      0.957 |              0.9468 |  832 |   34 |    9 |  125 |
|        0.25 |              0.8446 |           0.9328 |       0.8865 |      0.968 |              0.9531 |  843 |   23 |    9 |  125 |
|        0.3  |              0.8971 |           0.9104 |       0.9037 |      0.974 |              0.9471 |  852 |   14 |   12 |  122 |
|        0.35 |              0.968  |           0.903  |       0.9344 |      0.983 |              0.9492 |  862 |    4 |   13 |  121 |
|        0.4  |              0.9675 |           0.8881 |       0.9261 |      0.981 |              0.9417 |  862 |    4 |   15 |  119 |
|        0.45 |              0.9829 |           0.8582 |       0.9163 |      0.979 |              0.9279 |  864 |    2 |   19 |  115 |
|        0.5  |              0.9913 |           0.8507 |       0.9157 |      0.979 |              0.9248 |  865 |    1 |   20 |  114 |
|        0.55 |              0.9912 |           0.8358 |       0.9069 |      0.977 |              0.9173 |  865 |    1 |   22 |  112 |
|        0.6  |              0.991  |           0.8209 |       0.898  |      0.975 |              0.9099 |  865 |    1 |   24 |  110 |
|        0.65 |              1      |           0.791  |       0.8833 |      0.972 |              0.8955 |  866 |    0 |   28 |  106 |
|        0.7  |              1      |           0.7612 |       0.8644 |      0.968 |              0.8806 |  866 |    0 |   32 |  102 |
|        0.75 |              1      |           0.7015 |       0.8246 |      0.96  |              0.8507 |  866 |    0 |   40 |   94 |
|        0.8  |              1      |           0.6045 |       0.7535 |      0.947 |              0.8022 |  866 |    0 |   53 |   81 |
|        0.85 |              1      |           0.4627 |       0.6327 |      0.928 |              0.7313 |  866 |    0 |   72 |   62 |
|        0.9  |              1      |           0.3284 |       0.4944 |      0.91  |              0.6642 |  866 |    0 |   90 |   44 |

---

## 6. Hybrid Score Evaluation

- **Result**: Hybrid score (0.9237) did not provide a clear improvement over supervised model (0.9344). Retained simpler supervised model.

---

## 7. Diagnostic Error Analysis

- **False Positives (Valid classified as Invalid)**: **4 records**
  - Legitimate high-load test conditions (e.g. high load current or ambient temp) triggering sensitive sensor ratio thresholds.
- **False Negatives (Invalid classified as Valid)**: **13 records**
  - Subtle functional invalidations where physical parameters remain within nominal ranges without DAQ drops.

---

## 8. Test_Data Final Inference (350 Records)

- Evaluated on `Test_Data` without `Validity_Label` or `Reference_Parameter`.
- **Test Records Predicted Invalid**: **121.0 expected proportion in production** (Output saved to `outputs/p2_stage4_final_test_predictions.csv`).
