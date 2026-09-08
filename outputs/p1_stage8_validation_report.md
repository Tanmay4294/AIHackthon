# Person 1 - Stage 8: Validation & Threshold Selection Report

## A. Dataset Summary

- Training records: `1000`
- Valid count: `866` (`86.60%`)
- Invalid count: `134` (`13.40%`)
- Class imbalance: `Valid:Invalid = 866:134`

## B. CV Methodology

- 5-fold StratifiedKFold
- shuffle=True
- random_state=42
- Out-of-fold prediction generation only
- Threshold tuning performed exclusively on OOF probabilities
- Preprocessing and feature engineering kept inside fold-isolated pipelines
- `Test_ID`, `Validity_Label`, and `Reference_Parameter` are excluded from model features

## C. Threshold Analysis

Selected threshold: `0.40`

Selection rule:

Selected threshold 0.40 from 2 threshold(s) within 0.001 of the best Invalid F1 (0.9853). Primary criterion: Invalid F1. Tie-breaks favored higher Invalid recall, then higher Invalid precision, then fewer false positives and a deterministic lower-threshold choice.

Threshold sweep:

|   Threshold |   Num_Flagged |   Flag_Rate(%) |   Invalid_Precision |   Invalid_Recall |   Invalid_F1 |   Balanced_Accuracy |   Accuracy |   TP |   TN |   FP |   FN | Confusion_Matrix       |
|------------:|--------------:|---------------:|--------------------:|-----------------:|-------------:|--------------------:|-----------:|-----:|-----:|-----:|-----:|:-----------------------|
|        0.1  |           326 |           32.6 |              0.411  |           1      |       0.5826 |              0.8891 |      0.808 |  134 |  674 |  192 |    0 | [[674, 192], [0, 134]] |
|        0.15 |           229 |           22.9 |              0.5852 |           1      |       0.7383 |              0.9452 |      0.905 |  134 |  771 |   95 |    0 | [[771, 95], [0, 134]]  |
|        0.2  |           184 |           18.4 |              0.7283 |           1      |       0.8428 |              0.9711 |      0.95  |  134 |  816 |   50 |    0 | [[816, 50], [0, 134]]  |
|        0.25 |           157 |           15.7 |              0.8535 |           1      |       0.921  |              0.9867 |      0.977 |  134 |  843 |   23 |    0 | [[843, 23], [0, 134]]  |
|        0.3  |           148 |           14.8 |              0.9054 |           1      |       0.9504 |              0.9919 |      0.986 |  134 |  852 |   14 |    0 | [[852, 14], [0, 134]]  |
|        0.35 |           143 |           14.3 |              0.9371 |           1      |       0.9675 |              0.9948 |      0.991 |  134 |  857 |    9 |    0 | [[857, 9], [0, 134]]   |
|        0.4  |           138 |           13.8 |              0.971  |           1      |       0.9853 |              0.9977 |      0.996 |  134 |  862 |    4 |    0 | [[862, 4], [0, 134]]   |
|        0.45 |           136 |           13.6 |              0.9779 |           0.9925 |       0.9852 |              0.9945 |      0.996 |  133 |  863 |    3 |    1 | [[863, 3], [1, 133]]   |
|        0.5  |           132 |           13.2 |              0.9848 |           0.9701 |       0.9774 |              0.9839 |      0.994 |  130 |  864 |    2 |    4 | [[864, 2], [4, 130]]   |
|        0.55 |           131 |           13.1 |              0.9924 |           0.9701 |       0.9811 |              0.9845 |      0.995 |  130 |  865 |    1 |    4 | [[865, 1], [4, 130]]   |
|        0.6  |           124 |           12.4 |              1      |           0.9254 |       0.9612 |              0.9627 |      0.99  |  124 |  866 |    0 |   10 | [[866, 0], [10, 124]]  |
|        0.65 |           123 |           12.3 |              1      |           0.9179 |       0.9572 |              0.959  |      0.989 |  123 |  866 |    0 |   11 | [[866, 0], [11, 123]]  |
|        0.7  |           116 |           11.6 |              1      |           0.8657 |       0.928  |              0.9328 |      0.982 |  116 |  866 |    0 |   18 | [[866, 0], [18, 116]]  |
|        0.75 |           103 |           10.3 |              1      |           0.7687 |       0.8692 |              0.8843 |      0.969 |  103 |  866 |    0 |   31 | [[866, 0], [31, 103]]  |
|        0.8  |            85 |            8.5 |              1      |           0.6343 |       0.7763 |              0.8172 |      0.951 |   85 |  866 |    0 |   49 | [[866, 0], [49, 85]]   |
|        0.85 |            67 |            6.7 |              1      |           0.5    |       0.6667 |              0.75   |      0.933 |   67 |  866 |    0 |   67 | [[866, 0], [67, 67]]   |
|        0.9  |            51 |            5.1 |              1      |           0.3806 |       0.5514 |              0.6903 |      0.917 |   51 |  866 |    0 |   83 | [[866, 0], [83, 51]]   |

## D. Overall OOF Performance

- Invalid Precision: `0.9710`
- Invalid Recall: `1.0000`
- Invalid F1: `0.9853`
- Accuracy: `0.9960`
- Balanced Accuracy: `0.9977`
- ROC-AUC: `0.9999`
- PR-AUC: `0.9991`

### Confusion Matrix

| Actual       |   Predicted_Valid |   Predicted_Invalid |   Row_Total |
|:-------------|------------------:|--------------------:|------------:|
| True Valid   |               862 |                   4 |         866 |
| True Invalid |                 0 |                 134 |         134 |

### Named Confusion Counts

- True Valid: `862`
- False Invalid: `4`
- True Invalid: `134`
- Missed Invalid: `0`

## E. Fold-Level Performance

### Per Fold

|   Fold |   Fold_Size |   Invalid_Precision |   Invalid_Recall |   Invalid_F1 |   Balanced_Accuracy |   Accuracy |   TP |   TN |   FP |   FN |
|-------:|------------:|--------------------:|-----------------:|-------------:|--------------------:|-----------:|-----:|-----:|-----:|-----:|
|      1 |         200 |              0.8966 |                1 |       0.9455 |              0.9914 |      0.985 |   26 |  171 |    3 |    0 |
|      2 |         200 |              1      |                1 |       1      |              1      |      1     |   27 |  173 |    0 |    0 |
|      3 |         200 |              0.9643 |                1 |       0.9818 |              0.9971 |      0.995 |   27 |  172 |    1 |    0 |
|      4 |         200 |              1      |                1 |       1      |              1      |      1     |   27 |  173 |    0 |    0 |
|      5 |         200 |              1      |                1 |       1      |              1      |      1     |   27 |  173 |    0 |    0 |

### Stability Summary

| Metric            |   Mean |    Std |    Min |   Max |
|:------------------|-------:|-------:|-------:|------:|
| Invalid_Precision | 0.9722 | 0.0402 | 0.8966 |     1 |
| Invalid_Recall    | 1      | 0      | 1      |     1 |
| Invalid_F1        | 0.9855 | 0.0212 | 0.9455 |     1 |
| Balanced_Accuracy | 0.9977 | 0.0033 | 0.9914 |     1 |
| Accuracy          | 0.996  | 0.0058 | 0.985  |     1 |

## F. Misclassification Analysis

- False Positive count: `4`
- False Negative count: `0`

### False Positive Summary

| Test_ID   | Validity_Label   |   fold |   source_index |   oof_probability_invalid | selected_prediction   |   selected_prediction_binary | operating_regime   |   Load_Current_A |   Applied_Voltage_kV |   Ambient_Temperature_C |   Test_Duration_min |   Sensor_S1 |   Sensor_S2 |   Sensor_S3 |   Sensor_S4 | missing_any   | non_finite_value   | invalid_voltage   | invalid_current   | invalid_duration   | invalid_ambient_temp   | sensor_s2_negative   | sensor_zero_reading   | malformed_numeric   | data_quality_issue   |   p1_max_abs_residual |   p1_mean_abs_residual |   p1_consistency_index |   p1_sensor_disagreement_index |   p1_regime_cluster | high_current_flag   | high_voltage_flag   | high_temperature_flag   | long_duration_flag   | high_residual_flag   | sensor_disagreement_flag   | error_type   |
|:----------|:-----------------|-------:|---------------:|--------------------------:|:----------------------|-----------------------------:|:-------------------|-----------------:|---------------------:|------------------------:|--------------------:|------------:|------------:|------------:|------------:|:--------------|:-------------------|:------------------|:------------------|:-------------------|:-----------------------|:---------------------|:----------------------|:--------------------|:---------------------|----------------------:|-----------------------:|-----------------------:|-------------------------------:|--------------------:|:--------------------|:--------------------|:------------------------|:---------------------|:---------------------|:---------------------------|:-------------|
| TRN-0838  | Valid            |      1 |            301 |                  0.406173 | Invalid               |                            1 | High Voltage       |          79.8589 |              31.932  |                 32.6147 |             44.5479 |     20.6119 |     19.6283 |     25.9704 |     14.6559 | False         | False              | False             | False             | False              | False                  | False                | False                 | False               | False                |               33.3607 |                8.62212 |               0.103927 |                       0.559663 |                   3 | False               | True                | True                    | False                | True                 | False                      | FP           |
| TRN-0502  | Valid            |      1 |            691 |                  0.481854 | Invalid               |                            1 | High Voltage       |          15.7485 |              31.4814 |                 22.313  |             54.752  |     16.6856 |     13.4554 |     22.2083 |     63.0366 | False         | False              | False             | False             | False              | False                  | False                | False                 | False               | False                |               15.3845 |                3.91411 |               0.203496 |                       1.7188   |                   3 | False               | True                | False                   | False                | True                 | True                       | FP           |
| TRN-0705  | Valid            |      1 |            704 |                  0.538904 | Invalid               |                            1 | Standard           |          25.6949 |               8.2692 |                 31.0763 |             11.3228 |      5.3401 |      6.1135 |      6.6989 |     39.6231 | False         | False              | False             | False             | False              | False                  | False                | False                 | False               | False                |               11.2484 |                2.95701 |               0.252716 |                       2.37353  |                   4 | False               | False               | True                    | False                | True                 | True                       | FP           |
| TRN-0724  | Valid            |      3 |            920 |                  0.588373 | Invalid               |                            1 | Heavy Current      |         103.281  |              16.1649 |                 29.1169 |             27.8213 |     14.339  |     15.9315 |     17.0601 |     64.6338 | False         | False              | False             | False             | False              | False                  | False                | False                 | False               | False                |               16.3367 |                4.55846 |               0.179906 |                       1.79681  |                   2 | True                | False               | False                   | False                | True                 | True                       | FP           |

### False Negative Summary

_No records_

### Systematic Pattern Table

| Pattern_Category   | Pattern                   |   Count |   Rate(%) | Notes                                            | Error_Type   |
|:-------------------|:--------------------------|--------:|----------:|:-------------------------------------------------|:-------------|
| Operating Regime   | High Voltage              |       2 |        50 | 2 / 4 FP cases occur in this regime.             | FP           |
| Operating Regime   | Standard                  |       1 |        25 | 1 / 4 FP cases occur in this regime.             | FP           |
| Operating Regime   | Heavy Current             |       1 |        25 | 1 / 4 FP cases occur in this regime.             | FP           |
| Residual Severity  | p1_max_abs_residual >= 10 |       4 |       100 | Counts cases with large fold-isolated residuals. | FP           |
| Load               | Load_Current_A > 85       |       1 |        25 | High-current operating regime.                   | FP           |
| Voltage            | Applied_Voltage_kV > 25   |       2 |        50 | High-voltage operating regime.                   | FP           |
| FN Count           | None                      |       0 |         0 | No FN records at the selected threshold.         | FN           |

## G. Final Recommendation

- Final model: `Random_Forest`
- Final feature set: `Set_D_Full_Stage5`
- Final threshold: `0.40`
- Expected Invalid detection behavior: prioritize catching Invalid records first, with a very small number of false positives concentrated in unusual but physically plausible operating regimes.
- Integration readiness: Stable enough for integration
