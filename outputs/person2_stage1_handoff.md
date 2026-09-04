# Person 2 — Stage 1 Data Quality Audit Handoff

**From**: Person 2 (Data Quality & Classification Lead)  
**To**: Person 1 (Equipment Behaviour & Regime Analysis Lead)  
**Project**: CPRI State-Level Hackathon — Task 01  
**Scope**: P2-Stage 1 Data Quality Audit Deliverables  

---

## 1. Executive Handoff Overview

Stage 1 has established a deterministic, reproducible, **zero-deletion** data quality audit for the CPRI Hackathon dataset (`CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx`).

### Dataset Metrics:
- **Training_Data**: Exactly **1,000 records** (866 Valid, 134 Invalid)
- **Test_Data**: Exactly **350 records**
- **Quality Issue Prevalence**:
  - `Training_Data`: **72 records (7.20%)** exhibit data quality issues.
  - `Test_Data`: **25 records (7.14%)** exhibit data quality issues.

---

## 2. Quality Flag Specifications

Every record in `outputs/data_quality_audit_training.csv` and `outputs/data_quality_audit_test.csv` has been enriched with the following deterministic row-level flags:

| Flag Name | Description | Training Count (%) | Test Count (%) |
| :--- | :--- | :---: | :---: |
| `missing_any` | `True` if any sensor column (`Sensor_S1`..`S4`) is NaN. | 44 (4.40%) | 17 (4.86%) |
| `missing_columns` | String listing exact missing sensor names (e.g. `Sensor_S4`). | N/A | N/A |
| `missing_critical_measurement` | `True` if Voltage, Current, Ambient Temp, or Duration is missing. | 0 (0.00%) | 0 (0.00%) |
| `non_finite_value` | `True` if `NaN`, `+Inf`, or `-Inf` exists in numeric streams. | 44 (4.40%) | 17 (4.86%) |
| `duplicate_full_row` | `True` if 100% exact duplicate row across all columns. | 0 (0.00%) | 0 (0.00%) |
| `duplicate_test_id` | `True` if duplicate `Test_ID` string occurs. | 0 (0.00%) | 0 (0.00%) |
| `duplicate_measurement_pair` | `True` if identical measurement feature vector exists in another Test_ID. | 24 (2.40%) | 8 (2.29%) |
| `malformed_numeric` | `True` if non-numeric or unparseable string is present. | 0 (0.00%) | 0 (0.00%) |
| `invalid_voltage` | `True` if `Applied_Voltage_kV < 0`. | 0 (0.00%) | 0 (0.00%) |
| `invalid_current` | `True` if `Load_Current_A < 0`. | 0 (0.00%) | 0 (0.00%) |
| `invalid_duration` | `True` if `Test_Duration_min <= 0`. | 0 (0.00%) | 0 (0.00%) |
| `invalid_ambient_temp` | `True` if `Ambient_Temperature_C < -273.15`. | 0 (0.00%) | 0 (0.00%) |
| `invalid_sensor_negative` | `True` if `Sensor_S2 < 0` (negative sensor drop). | 1 (0.10%) | 1 (0.29%) |
| `invalid_sensor_zero` | `True` if `Sensor_S1`, `S2`, or `S3` hits exact 0.0000. | 3 (0.30%) | 1 (0.29%) |
| `data_quality_issue_count` | Sum of active quality flags per record ($0 \dots 5$). | N/A | N/A |
| `data_quality_issue` | `True` if ANY quality flag is triggered (`count > 0`). | 72 (7.20%) | 25 (7.14%) |

---

## 3. Relationship with Historical `Validity_Label`

In `Training_Data` (1,000 records, 866 Valid, 134 Invalid):

1. **`duplicate_measurement_pair` (24 records / 12 pairs)**:
   - **100.0% Invalid (24 Invalid, 0 Valid)**.
   - Engineers consistently invalidated tests that generated exact duplicate measurement pairs.
2. **`invalid_sensor_negative` (1 record)**:
   - **100.0% Invalid (1 Invalid, 0 Valid)** (`TRN-0346`).
3. **`invalid_sensor_zero` (3 records)**:
   - **100.0% Invalid (3 Invalid, 0 Valid)** (`TRN-0346`, `TRN-0357`, `TRN-0512`).
4. **`missing_any` (44 records)**:
   - **29 Valid (65.9%) vs 15 Invalid (34.1%)**.
   - **Crucial Finding**: Engineers did NOT automatically invalidate a test simply because a single non-critical sensor (like `Sensor_S4`) was missing if electrical measurements and overall behavior were consistent.

---

## 4. Key Distinction for Person 1 (Regimes vs Quality Corruption)

> [!IMPORTANT]
> **Category 1 — Definite Data Quality Problems**:
> `duplicate_measurement_pair`, `invalid_sensor_negative`, `invalid_sensor_zero`, and `missing_any` represent recording / DAQ hardware quality defects.
>
> **Category 2 & 3 — Operating Regimes & Genuine Equipment Behaviour**:
> High currents (up to 109.95 A) or high ambient temperatures (up to 55.0°C) are **GENUINE OPERATING REGIMES** (as stated in CPRI dataset README: *"Genuine operating-regime change is embedded in valid data"*).
>
> Do **NOT** use naive z-score or IQR thresholds (e.g. declaring current > 100A invalid) during your regime modeling. Genuine high-load testing is expected and valid!

---

## 5. Recommended Person 1 Usage Guidelines

1. **For Equipment Regime Clustering**:
   - Filter on `missing_any == False` or impute missing sensor values using physical sensor correlations (`Sensor_S1` vs `Sensor_S2`).
2. **For Task 01 Valid/Invalid Modeling**:
   - Use `duplicate_measurement_pair`, `invalid_sensor_negative`, and `invalid_sensor_zero` as strong deterministic indicators of invalidity.
3. **For Task 02 Reference Parameter Prediction**:
   - Filter out `data_quality_issue == True` when training regression models to avoid fitting noise from missing `Sensor_S4` values.
