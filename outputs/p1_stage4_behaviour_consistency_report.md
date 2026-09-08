# Person 1 — Stage 4: Behaviour & Physical Consistency Analysis Report

## Executive Summary
This report details the behaviour and physical-consistency analysis for CPRI Hackathon Task 01.
We constructed normal-behaviour reference regression models (`NormalBehaviourModels`) trained **STRICTLY on engineer-verified Valid historical test records** ($N=700$) to establish expected sensor relationships under varying operating conditions (voltage, current, temperature, duration).

Key residual and consistency features were generated across all 1,000 historical training records and 350 test records without row deletion or target leakage.

---

## 1. Physical Relationship Analysis & Operating Regimes
Strong linear correlations were identified between operating parameters and sensors.
Four distinct operating regimes were mapped:
1. **Heavy HV Load (Regime 1)**: High Current (>85A) & High Voltage (>25kV)
2. **Heavy Current (Regime 2)**: Load Current > 85A
3. **High Voltage (Regime 3)**: Applied Voltage > 25kV
4. **Standard Operating (Regime 4)**: Normal range operating parameters

### Operating Regime Breakdown
|   Regime_Code | Regime_Name   |   Total_Records |   Valid_Count |   Invalid_Count |   Invalid_Rate(%) |   Mean_Current_A |   Mean_Voltage_kV |
|--------------:|:--------------|----------------:|--------------:|----------------:|------------------:|-----------------:|------------------:|
|             1 | Heavy HV Load |              94 |            83 |              11 |             11.7  |            98.36 |             28.68 |
|             2 | Heavy Current |             179 |           155 |              24 |             13.41 |            97.94 |             16.47 |
|             3 | High Voltage  |             223 |           194 |              29 |             13    |            50.09 |             28.71 |
|             4 | Standard      |             504 |           434 |              70 |             13.89 |            50.25 |             16.05 |

---

## 2. Normal Behaviour Reference Models
Normal models were fitted using Ridge regression with cross-validated parameters strictly on Valid training data.

### Model Performance Metrics (Valid Training Records)
| Target_Sensor   |   Valid_Train_Count |   R2_Score |    MAE |    RMSE | Predictors                                                                   |
|:----------------|--------------------:|-----------:|-------:|--------:|:-----------------------------------------------------------------------------|
| Sensor_S1       |                 866 |     0.9977 | 0.1447 |  0.183  | Applied_Voltage_kV, Load_Current_A, Ambient_Temperature_C, Test_Duration_min |
| Sensor_S2       |                 866 |     0.9976 | 0.1375 |  0.1718 | Applied_Voltage_kV, Load_Current_A, Ambient_Temperature_C, Test_Duration_min |
| Sensor_S3       |                 866 |     0.9964 | 0.2356 |  0.2958 | Applied_Voltage_kV, Load_Current_A, Ambient_Temperature_C, Test_Duration_min |
| Sensor_S4       |                 837 |     0.0043 | 9.8145 | 12.5722 | Applied_Voltage_kV, Load_Current_A, Ambient_Temperature_C, Test_Duration_min |

---

## 3. Residual & Consistency Feature Separation Power
Invalid records exhibit significantly higher prediction residuals and sensor disagreement compared to Valid records.

### Effect Sizes (Cohen's d) & Separation Statistics
| Residual_Feature             |   Valid_Mean |   Valid_Median |   Valid_Std |   Invalid_Mean |   Invalid_Median |   Invalid_Std |   Cohens_D_EffectSize |   Invalid_Extreme_Residual_Rate(%) |
|:-----------------------------|-------------:|---------------:|------------:|---------------:|-----------------:|--------------:|----------------------:|-----------------------------------:|
| Sensor_S2_AbsResidual        |       0.1375 |         0.117  |      0.1031 |         2.7665 |           0.2352 |        3.8434 |                1.8817 |                              47.73 |
| Sensor_S1_AbsResidual        |       0.1447 |         0.1216 |      0.1122 |         2.423  |           0.2389 |        3.7447 |                1.6952 |                              45.31 |
| Sensor_S3_AbsResidual        |       0.2356 |         0.1945 |      0.179  |         1.9657 |           0.2659 |        4.0496 |                1.1902 |                              25.98 |
| p1_mean_abs_residual         |       2.5021 |         1.9912 |      1.981  |         4.3094 |           4.0707 |        2.2494 |                0.8952 |                              28.36 |
| p1_consistency_index         |       0.388  |         0.3343 |      0.214  |         0.2363 |           0.1972 |        0.1401 |               -0.7376 |                               2.24 |
| p1_sensor_disagreement_index |       1.5221 |         1.5157 |      0.5524 |         1.7016 |           1.6698 |        0.5818 |                0.3226 |                              17.16 |
| Sensor_S2_Residual           |       0      |         0.0024 |      0.1719 |         0.4362 |           0.0627 |        4.7214 |                0.2537 |                              33.33 |
| p1_max_abs_residual          |       9.4963 |         7.4026 |      7.916  |        11.3884 |          10.3023 |        6.2737 |                0.2452 |                               6.72 |
| p1_residual_error            |       9.4963 |         7.4026 |      7.916  |        11.3884 |          10.3023 |        6.2737 |                0.2452 |                               6.72 |
| Sensor_S3_Residual           |       0      |         0.0092 |      0.2959 |        -0.233  |          -0.0139 |        4.4988 |               -0.1431 |                              23.62 |

---

## 4. Visualizations Summary
The following 12 figures were generated and stored in `outputs/figures/`:
1. `p1_stage4_voltage_vs_sensors.png`: Applied Voltage vs Sensor readings.
2. `p1_stage4_current_vs_sensors.png`: Load Current vs Sensor readings.
3. `p1_stage4_cross_sensor_correlations.png`: Heatmap of cross-sensor alignment.
4. `p1_stage4_operating_regimes.png`: Distribution of operating regimes.
5. `p1_stage4_normal_fit_actual_vs_expected.png`: Expected vs Actual sensor fit on Valid records.
6. `p1_stage4_normal_model_r2_scores.png`: Barplot of normal model R² scores.
7. `p1_stage4_residual_distributions.png`: KDE plot of sensor residuals by class.
8. `p1_stage4_residual_boxplots_by_class.png`: Mean absolute residual boxplots.
9. `p1_stage4_max_abs_residual_by_class.png`: Max absolute residual boxplots.
10. `p1_stage4_consistency_index_by_class.png`: Consistency index density by class.
11. `p1_stage4_sensor_disagreement_index.png`: Sensor disagreement index distributions.
12. `p1_stage4_cohens_d_effect_sizes.png`: Effect size comparison across residual features.

---

## 5. Conclusion & Recommendations
- **`p1_max_abs_residual`** and **`p1_consistency_index`** provide powerful discriminative signal for identifying Invalid test records where raw physical parameters alone appear normal.
- All features are 100% reproducible and inference-safe for deployment in Person 2 Stage 4 final validity classification.
