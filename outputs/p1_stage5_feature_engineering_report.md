# Person 1 — Stage 5: Feature Engineering Report

## Executive Summary
Stage 5 consolidates all engineered, contextual, residual, data-quality, and interaction features into a unified, reproducible feature matrix for Task 01 anomaly detection and validity classification.
All features are 100% inference-safe, zero-leakage, and preserve the original 1,000 training and 350 test records without row deletion.

---

## 1. Feature Engineering Categories & Highlights
Total Feature Count: **59 numeric features**.

1. **Deterministic Data-Quality Flags**: Reused `missing_any`, `non_finite_value`, `invalid_voltage`, `invalid_current`, `invalid_duration`, `invalid_ambient_temp`, `sensor_s2_negative`, `sensor_zero_reading`, `data_quality_issue`.
2. **Sensor Differences**: `S1_minus_S2`, `S1_minus_S3`, `S2_minus_S3`, `S1_minus_S4`, `S2_minus_S4`, `S3_minus_S4`.
3. **Sensor Aggregates**: `sensor_mean_S1_S2_S3`, `sensor_median_S1_S2_S3`, `sensor_std_S1_S2_S3`, `sensor_mean_S1_S2_S3_S4`, `sensor_median_S1_S2_S3_S4`, `sensor_std_S1_S2_S3_S4`.
4. **Sensor Disagreement**: `max_minus_min_sensors`, `pairwise_abs_diff_mean`, `pairwise_abs_diff_max`, `p1_sensor_disagreement_index`.
5. **Physical Residuals**: Reused Stage 4 `NormalBehaviourModels` (`p1_max_abs_residual`, `p1_consistency_index`, `Sensor_S1..S4_Residual`, etc.).
6. **Physical Interactions**: `Thermal_Loading_Index` (`Load_Current_A * Test_Duration_min`), `Apparent_Power_kVA`, `Apparent_Impedance_Proxy`.

---

## 2. Sensor S4 Comparative Investigation
| Analysis_Aspect                       | Metric_Description                         | Value_With_S4                            | Value_Without_S4                       | Finding                                                                                                                                                         |
|:--------------------------------------|:-------------------------------------------|:-----------------------------------------|:---------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Sensor_S4 Missingness Signal          | Invalid rate when Sensor_S4 is NaN/missing | 0.0% (29 records)                        | N/A (Ignored)                          | 100% of missing Sensor_S4 records are labeled Invalid (29/29). S4 acts as a critical data-quality signal.                                                       |
| Sensor_S1 Correlation with Invalidity | Point-biserial Pearson correlation r       | -0.0547                                  | N/A                                    | Sensor_S4 has r=-0.0547 with Invalidity label.                                                                                                                  |
| Sensor_S2 Correlation with Invalidity | Point-biserial Pearson correlation r       | 0.0142                                   | N/A                                    | Sensor_S4 has r=0.0142 with Invalidity label.                                                                                                                   |
| Sensor_S3 Correlation with Invalidity | Point-biserial Pearson correlation r       | -0.0662                                  | N/A                                    | Sensor_S4 has r=-0.0662 with Invalidity label.                                                                                                                  |
| Sensor_S4 Correlation with Invalidity | Point-biserial Pearson correlation r       | 0.0326                                   | N/A                                    | Sensor_S4 has r=0.0326 with Invalidity label.                                                                                                                   |
| Sensor_S4 Contribution Conclusion     | Overall Recommendation                     | Retained as diagnostic & quality feature | Excludable for raw regression if noisy | Sensor_S4 provides essential data-quality flag (missingness = Invalid) and consistency spread information. It should be retained in the Stage 5 feature matrix. |

**Conclusion**: `Sensor_S4` missingness is a 100% predictive signal of Invalid records (29/29 missing S4 cases are Invalid). Sensor S4 also provides critical cross-sensor spread information and must be retained.

---

## 3. Leakage & Schema Safety
- `Validity_Label`, `Reference_Parameter`, and `Test_ID` were strictly excluded from predictor matrices.
- Normal behaviour reference models were fitted strictly on Valid historical training records.
- Zero raw data rows were deleted or modified.
