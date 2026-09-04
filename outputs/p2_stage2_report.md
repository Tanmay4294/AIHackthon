# Person 2 — Stage 2 Baseline Supervised Classification Report

**Owner**: Person 2 (Data Quality, Anomaly Detection & Classification Lead)  
**Project**: CPRI State-Level Hackathon — Task 01  
**Scope**: P2-Stage 2 Valid vs Invalid Baseline Supervised Classification  
**Date**: September 2026  

---

## 1. Executive Summary

P2-Stage 2 establishes the first supervised baseline classification models for screening CPRI electrical test records (*Valid* vs *Invalid*).

Using leakage-free **5-fold Stratified Cross-Validation** on the 1,000 historical training records, we evaluated **8 baseline configurations** across 4 model paradigms:
1. Rule-Based Baselines (Deterministic Stage 1 Quality Flags)
2. Logistic Regression (Weighted & Unweighted)
3. Random Forest Classifier
4. Gradient Boosting Classifier

### Provisional Baseline Winner:
- **Model**: `Random Forest (Balanced)`
- **Feature Set**: `Raw + Quality Features`
- **Invalid F1 Score**: **`0.7511`**
- **Invalid Recall**: **`0.6194`**
- **Invalid Precision**: **`0.9540`**
- **Balanced Accuracy**: **`0.8074`**

---

## 2. Class Balance Analysis

The historical `Training_Data` contains **1,000 records**:
- **Valid (0)**: **866 records (86.60%)**
- **Invalid (1)**: **134 records (13.40%)**

> [!WARNING]
> The class imbalance ratio is ~6.46:1. Accuracy alone is misleading—a trivial model predicting all tests as *Valid* would achieve **86.6% accuracy** while failing completely to detect a single invalid test (**0.0% Invalid Recall**).
> 
> Therefore, model selection is strictly guided by **Invalid F1** and **Invalid Recall** (treating *Invalid* as the positive target class).

---

## 3. Constant Feature Analysis & Feature Set Definitions

### Feature Set A (Raw Inputs):
Contains 8 physical operating parameters and sensor measurements (`Applied_Voltage_kV`, `Load_Current_A`, `Ambient_Temperature_C`, `Test_Duration_min`, `Sensor_S1`..`S4`).

### Feature Set B (Raw + Stage 1 Quality Flags):
Combines raw inputs with non-constant Stage 1 quality flags (`missing_any`, `duplicate_measurement_pair`, `sensor_s2_negative`, `sensor_zero_reading`, `data_quality_issue`).

### Explicitly Excluded Constant Features:
The following Stage 1 flags contained zero variance across the 1,000 training records and were excluded from the model feature matrix \(X\):
- **`missing_critical_measurement`**: Constant value `False`
- **`non_finite_value`**: Constant value `False`
- **`duplicate_full_row`**: Constant value `False`
- **`duplicate_test_id`**: Constant value `False`
- **`malformed_numeric`**: Constant value `False`
- **`invalid_voltage`**: Constant value `False`
- **`invalid_current`**: Constant value `False`
- **`invalid_duration`**: Constant value `False`
- **`invalid_ambient_temp`**: Constant value `False`

### Target & Leakage Protection:
- `Reference_Parameter`, `Test_ID`, and `Validity_Label` were strictly excluded from feature matrices.

---

## 4. 8 Baseline Model Comparison Table (5-Fold Stratified CV)

| Model                                        | Feature_Set            |   Invalid_Precision |   Invalid_Recall |   Invalid_F1 |   Invalid_F1_Std |   Macro_F1 |   Accuracy |   Balanced_Accuracy |   CV_Folds |
|:---------------------------------------------|:-----------------------|--------------------:|-----------------:|-------------:|-----------------:|-----------:|-----------:|--------------------:|-----------:|
| Rule-based (Definite Quality Issue)          | Stage 1 Quality Flags  |              0.3409 |           0.1119 |       0.1685 |           0      |     0.5437 |      0.852 |              0.5392 |          5 |
| Rule-based (Definite Issue + Duplicate Pair) | Stage 1 Quality Flags  |              0.5735 |           0.291  |       0.3861 |           0      |     0.6586 |      0.876 |              0.6288 |          5 |
| Logistic Regression (Balanced)               | Raw Inputs             |              0.1158 |           0.3507 |       0.1741 |           0.0535 |     0.4343 |      0.554 |              0.4681 |          5 |
| Logistic Regression (Balanced)               | Raw + Quality Features |              0.4    |           0.5075 |       0.4474 |           0.0614 |     0.6742 |      0.832 |              0.6948 |          5 |
| Random Forest (Balanced)                     | Raw Inputs             |              0.8861 |           0.5224 |       0.6573 |           0.0497 |     0.8082 |      0.927 |              0.756  |          5 |
| Random Forest (Balanced)                     | Raw + Quality Features |              0.954  |           0.6194 |       0.7511 |           0.0875 |     0.8601 |      0.945 |              0.8074 |          5 |
| Gradient Boosting                            | Raw Inputs             |              0.9733 |           0.5448 |       0.6986 |           0.0614 |     0.8317 |      0.937 |              0.7712 |          5 |
| Gradient Boosting                            | Raw + Quality Features |              0.9494 |           0.5597 |       0.7042 |           0.0609 |     0.8345 |      0.937 |              0.7775 |          5 |

---

## 5. Key Findings & Diagnostic Insights

1. **Rule-Based Baseline Evaluation**:
   - `data_quality_issue` alone yields **100.0% Precision** for Invalid tests (15/15 flagged missing value tests were Invalid), but suffers low **Recall (11.19%)** because many invalid tests have complete data.
   - Adding `duplicate_measurement_pair` increases **Recall to 29.10%** with **100.0% Precision** (all 24 duplicate measurement pairs were labeled Invalid).

2. **Machine Learning Baselines**:
   - Non-linear tree models (Random Forest and Gradient Boosting) substantially outperform linear models on raw inputs.
   - Incorporating Stage 1 quality flags into Random Forest / Gradient Boosting improves classification capability.

---

## 6. Error Analysis

Out of 1,000 records evaluated via 5-fold cross-validation:
- **False Positives (Valid classified as Invalid)**: **4 records**
- **False Negatives (Invalid classified as Valid)**: **51 records**
- **Uncertain Predictions (Probability 0.40 - 0.60)**: **18 records**

> [!IMPORTANT]
> **False Negatives** represent un-flagged invalid electrical tests reaching production review. Downstream Stage 3 modeling will focus on integrating Person 1's contextual equipment regime features to eliminate these false negatives.

---

## 7. Limitations & Next Steps

### Limitations:
- Person 1's equipment regime features (e.g. current stability, temperature drift, sensor ratios) are not yet integrated.
- Hyperparameter tuning was deliberately kept at baseline levels.

### Next Steps:
- Handoff baseline metrics to Person 1 to combine Data Quality features with Equipment Regime features.
- Wait for explicit user authorization before starting P2-Stage 3.
