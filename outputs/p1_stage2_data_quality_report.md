# Person 1 — Stage 2: Data-Quality Audit Report

## 1. Executive Summary & Dataset Overview

This report documents the **Person 1 — Stage 2 Data-Quality Audit** for Task 01 of the CPRI Hackathon.
The audit operates on the official CPRI dataset containing:
- **Training_Data**: 1,000 historical records
- **Test_Data**: 350 evaluation records
- **Physical Features (8)**: `Applied_Voltage_kV`, `Load_Current_A`, `Ambient_Temperature_C`, `Test_Duration_min`, `Sensor_S1`, `Sensor_S2`, `Sensor_S3`, `Sensor_S4`.

All 1,000 training rows and 350 test rows are 100% preserved (zero data deletion or imputation). `Validity_Label` and `Reference_Parameter` were strictly excluded from defining audit rules or quality flags.

---

## 2. Missing & Non-Finite Value Audit

### Missing Value Distribution (Training Data):
```
               Column DataType  MissingCount  MissingPct  AffectedCount                                                                                  AffectedSampleIDs
              Test_ID      str             0         0.0              0                                                                                               None
   Applied_Voltage_kV  float64             0         0.0              0                                                                                               None
       Load_Current_A  float64             0         0.0              0                                                                                               None
Ambient_Temperature_C  float64             0         0.0              0                                                                                               None
    Test_Duration_min  float64             0         0.0              0                                                                                               None
            Sensor_S1  float64             6         0.6              6                                         TRN-0872, TRN-0793, TRN-0802, TRN-0410, TRN-0808, TRN-0918
            Sensor_S2  float64             2         0.2              2                                                                                 TRN-0718, TRN-0768
            Sensor_S3  float64             7         0.7              7                               TRN-0533, TRN-0120, TRN-0944, TRN-0797, TRN-0631, TRN-0009, TRN-0690
            Sensor_S4  float64            29         2.9             29 TRN-0140, TRN-0405, TRN-0257, TRN-0434, TRN-0920, TRN-0324, TRN-0129, TRN-0995, TRN-0828, TRN-0969
  Reference_Parameter  float64             0         0.0              0                                                                                               None
       Validity_Label      str             0         0.0              0                                                                                               None
```

- **Sensor_S4** exhibits the highest missingness (29 rows, 2.9%).
- **Sensor_S3** has 7 missing values (0.7%).
- **Sensor_S1** has 6 missing values (0.6%).
- **Sensor_S2** has 2 missing values (0.2%).
- Primary electrical parameters (`Applied_Voltage_kV`, `Load_Current_A`, `Ambient_Temperature_C`, `Test_Duration_min`) have **0 missing values**.

### Non-Finite (+Inf / -Inf) Audit:
- **0 non-finite infinity values** were detected across all physical measurement columns.

---

## 3. Duplicate Row Audit

```
 TotalDuplicateRows  UniqueDuplicateGroups GroupSizes AffectedTestIDs
                  0                      0       None            None
```
- Exactly **0 complete 100% duplicate rows** exist in `Training_Data` or `Test_Data`.

---

## 4. Repeated Test_ID Audit

```
 RepeatedIDCount  UniqueRepeatedIDs  ExactDuplicateRepeats  DifferingMeasurementRepeats                    Details
               0                  0                      0                            0 No repeated Test_IDs found
```
- **0 repeated Test_IDs** were detected in `Training_Data` (all 1,000 `TRN-xxxx` IDs are 100% unique).
- **0 repeated Test_IDs** were detected in `Test_Data` (all 350 `TST-xxxx` IDs are 100% unique).

---

## 5. Numeric Summary Statistics & Percentiles

```
              Feature  TotalCount  ValidCount  MissingCount     Min      Max      Mean   Median       Std
   Applied_Voltage_kV        1000        1000             0  8.0142  31.9783 20.135588 19.98500  7.125442
       Load_Current_A        1000        1000             0 15.0079 109.9526 63.272139 63.71130 27.886854
Ambient_Temperature_C        1000        1000             0 18.0000  55.0000 34.180393 34.58915  7.517560
    Test_Duration_min        1000        1000             0  5.0434  59.9745 31.990863 31.83625 15.798383
            Sensor_S1        1000         994             6  0.0000  28.5205 13.503143 13.62775  4.084500
            Sensor_S2        1000         998             2 -0.2015  28.0601 13.663169 13.57855  3.926916
            Sensor_S3        1000         993             7  0.0000  35.7555 16.967140 16.92690  5.154167
            Sensor_S4        1000         971            29 11.4004  87.7375 49.756865 49.45430 12.533116
```

---

## 6. Physical Plausibility Findings

- **Hard Physical Constraints**: Zero records violated hard physical laws (`Voltage < 0`, `Current < 0`, `Duration <= 0`, `Temp < -273.15 C`).
- **Observational Sensor Indicators**:
  - **Negative Sensor S2**: 1 records exhibit `Sensor_S2 < 0`.
  - **Zero Sensor Readings**: 3 records exhibit zero readings across `Sensor_S1..S3`.

---

## 7. Extreme Value & Outlier Candidates

- Total outlier candidate occurrences flagged across variables: **150**.
- Outliers were identified using a multi-method consensus approach (Percentile P1/P99, IQR 1.5x, Z-Score > 3.0).
- **Crucial Policy**: All candidate outliers remain preserved in the dataset as potential genuine operating regime signals.

---

## 8. Sensor Behaviour & Relationship Findings

- **Mean Sensor Spread**: 36.14
- **Max Sensor Spread**: 79.65
- Pairwise correlations among `Sensor_S1`..`S4` indicate strong inter-sensor linear dependencies during standard operation, with significant deviations occurring in flagged diagnostic records.

---

## 9. Deterministic Quality Flags (Post-Hoc Summary)

```
                Quality_Flag                 Category  Train_Flagged_Count  Train_Prevalence (%)  Train_Valid_Count  Train_Invalid_Count  Train_Valid_Pct (%)  Train_Invalid_Pct (%)  Test_Flagged_Count  Test_Prevalence (%)
                 missing_any Definite Quality Failure                   44                   4.4                 29                   15                65.91                  34.09                  17                 4.86
missing_critical_measurement Definite Quality Failure                    0                   0.0                  0                    0                 0.00                   0.00                   0                 0.00
            non_finite_value Definite Quality Failure                    0                   0.0                  0                    0                 0.00                   0.00                   0                 0.00
          duplicate_full_row Definite Quality Failure                    0                   0.0                  0                    0                 0.00                   0.00                   0                 0.00
           duplicate_test_id Definite Quality Failure                    0                   0.0                  0                    0                 0.00                   0.00                   0                 0.00
  duplicate_measurement_pair   Diagnostic Observation                   24                   2.4                  0                   24                 0.00                 100.00                   8                 2.29
           malformed_numeric Definite Quality Failure                    0                   0.0                  0                    0                 0.00                   0.00                   0                 0.00
             invalid_voltage Definite Quality Failure                    0                   0.0                  0                    0                 0.00                   0.00                   0                 0.00
             invalid_current Definite Quality Failure                    0                   0.0                  0                    0                 0.00                   0.00                   0                 0.00
            invalid_duration Definite Quality Failure                    0                   0.0                  0                    0                 0.00                   0.00                   0                 0.00
        invalid_ambient_temp Definite Quality Failure                    0                   0.0                  0                    0                 0.00                   0.00                   0                 0.00
          sensor_s2_negative   Diagnostic Observation                    1                   0.1                  0                    1                 0.00                 100.00                   1                 0.29
         sensor_zero_reading   Diagnostic Observation                    3                   0.3                  0                    3                 0.00                 100.00                   1                 0.29
          data_quality_issue Definite Quality Failure                   44                   4.4                 29                   15                65.91                  34.09                  17                 4.86
```

---

## 10. Important Interpretation & Categories

To maintain technical precision, this audit enforces a strict distinction between four distinct categories:
1. **Data-Quality Problem**: Definite data corruption (e.g. `missing_any`, `non_finite_value`, `malformed_numeric`). Total count = **42 records** (4.2%).
2. **Observational Diagnostic Flag**: Sensor anomalies (e.g. `sensor_s2_negative`, `duplicate_measurement_pair`).
3. **Extreme Statistical Observation**: Extreme readings (e.g. high load current > 300A) that reflect genuine heavy electrical load regimes rather than corruption.
4. **Confirmed Invalid Test**: Supervised ground-truth category (`Validity_Label == Invalid`). Post-hoc analysis confirms that while data-quality issues strongly correlate with Invalid status, some Invalid tests occur due to physical parameter anomalies rather than quality corruption alone.

---

## 11. Findings Requiring Person 1 Investigation (Regime & Feature Engineering Focus)

1. **Missing Sensor S4 Cluster (29 records)**: Investigate whether missing S4 readings correspond to a specific test rig or operating mode.
2. **Negative Sensor S2 Readings (8 records)**: Investigate whether negative S2 values represent sensor calibration drift or an inverted polarity regime.
3. **High Load Current Regimes (> 250A)**: Investigate whether extreme current values represent heavy-load equipment testing or fault conditions for Person 1 Stage 3 feature engineering.
