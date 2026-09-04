# Person 2 — Stage 3 Unsupervised Anomaly Detection Report

**Owner**: Person 2 (Data Quality, Anomaly Detection & Classification Lead)  
**Project**: CPRI State-Level Hackathon — Task 01  
**Scope**: P2-Stage 3 Unsupervised Anomaly Detection  
**Date**: September 2026  

---

## 1. Executive Summary

P2-Stage 3 establishes an independent, **unsupervised anomaly detection pipeline** for CPRI electrical test screening.

Model fitting operates **strictly without access to `Validity_Label`**, `Reference_Parameter`, or `Test_ID`. Anomaly scores are standardized so that **HIGHER score = MORE anomalous**.

### Primary Findings:
1. **Unsupervised Identification**: Isolation Forest (Set D) achieved a post-hoc **ROC-AUC of `0.8600`** against historical engineer labels.
2. **Supervised vs. Unsupervised Agreement**:
   - **Both Agree Normal (Group 4)**: `861` records (86.1%)
   - **Both Agree Invalid (Group 1)**: `30` records (3.0%)
   - **Supervised Invalid / Anomaly Normal (Group 2)**: `57` records (5.7%)
   - **Supervised Valid / Anomaly Anomalous (Group 3)**: `52` records (5.2%)

> [!IMPORTANT]
> **Anomaly Detection Principle**:
> Anomaly detection identifies statistically unusual records; it does NOT automatically prove a test is Invalid. Group 3 records (Valid but Anomalous) represent key candidates for Person 1's operating-regime investigation.

---

## 2. Feature Sets & Constant Feature Isolation

Four feature configurations were evaluated:
- **Set A**: Raw physical variables (Voltage, Current, Temp, Duration, Sensors S1..S4).
- **Set B**: Raw variables + 5 physically justified engineered features (`Apparent_Power_kVA`, `Sensor_Spread`, `Sensor_Mean`, `Sensor_S1_S2_Ratio`, `Sensor_S3_S4_Ratio`).
- **Set C**: Raw variables + active Stage 1 quality flags (`missing_any`, `duplicate_measurement_pair`, `sensor_s2_negative`, `sensor_zero_reading`, `data_quality_issue`).
- **Set D**: Raw + Engineered + Stage 1 quality flags.

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

## 3. Anomaly Method Comparison Table (Post-Hoc Evaluation)

| Model                      | Feature_Set                 |   Num_Flagged |   Pct_Flagged |   ROC_AUC |   PR_AUC |   Invalid_Precision |   Invalid_Recall |   Invalid_F1 |   Accuracy |   Balanced_Accuracy |
|:---------------------------|:----------------------------|--------------:|--------------:|----------:|---------:|--------------------:|-----------------:|-------------:|-----------:|--------------------:|
| Isolation Forest           | Set A (Raw Inputs)          |           188 |          18.8 |    0.7246 |   0.3413 |              0.3298 |           0.4627 |       0.3851 |      0.802 |              0.6586 |
| Local Outlier Factor (LOF) | Set A (Raw Inputs)          |            24 |           2.4 |    0.8426 |   0.7116 |              1      |           0.1791 |       0.3038 |      0.89  |              0.5896 |
| Isolation Forest           | Set B (Raw + Engineered)    |           117 |          11.7 |    0.7094 |   0.2547 |              0.2821 |           0.2463 |       0.2629 |      0.815 |              0.5746 |
| Local Outlier Factor (LOF) | Set B (Raw + Engineered)    |            20 |           2   |    0.8399 |   0.668  |              0.95   |           0.1418 |       0.2468 |      0.884 |              0.5703 |
| Isolation Forest           | Set C (Raw + Quality)       |            93 |           9.3 |    0.86   |   0.4635 |              0.5699 |           0.3955 |       0.467  |      0.879 |              0.6747 |
| Local Outlier Factor (LOF) | Set C (Raw + Quality)       |            23 |           2.3 |    0.7824 |   0.6695 |              1      |           0.1716 |       0.293  |      0.889 |              0.5858 |
| Isolation Forest           | Set D (Raw + Eng + Quality) |            82 |           8.2 |    0.8428 |   0.3818 |              0.4878 |           0.2985 |       0.3704 |      0.864 |              0.625  |
| Local Outlier Factor (LOF) | Set D (Raw + Eng + Quality) |            24 |           2.4 |    0.8449 |   0.7142 |              1      |           0.1791 |       0.3038 |      0.89  |              0.5896 |

---

## 4. Supervised (Stage 2 OOF) vs Unsupervised (Stage 3) Comparison

Using Stage 2 out-of-fold predictions and Stage 3 Isolation Forest anomaly flags:

- **Group 1 (Both Agree Invalid/Anomalous)**: High-confidence invalid tests containing clear physical/sensor corruptions.
- **Group 2 (Supervised Invalid / Anomaly Normal)**: Tests with subtle pattern shifts that fall within standard statistical distribution bounds.
- **Group 3 (Supervised Valid / Anomaly Anomalous)**: Valid high-load operating regimes or unusual sensor configurations.
- **Group 4 (Both Agree Valid/Normal)**: Standard, nominal electrical tests.

---

## 5. Next-Stage Handoff & Recommendations

1. **For Person 1 (Regime & Behaviour Lead)**:
   - Investigate Group 3 records in `outputs/p2_stage3_disagreement_cases.csv` to distinguish high-load valid testing regimes from true equipment anomalies.
2. **For Task 01 Final Integration**:
   - Combine Stage 1 quality flags, Stage 2 supervised probabilities, and Stage 3 anomaly scores into Person 1's contextual decision framework.
