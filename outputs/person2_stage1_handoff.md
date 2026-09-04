# Person 2 — Stage 1 Data Quality Audit Handoff

**From**: Person 2 (Data Quality & Classification Lead)  
**To**: Person 1 (Equipment Behaviour & Regime Analysis Lead)  
**Project**: CPRI State-Level Hackathon — Task 01  
**Scope**: P2-Stage 1 Data Quality Audit Deliverables (Corrected Definitions)  

---

## 1. Executive Handoff Overview

Stage 1 has established a deterministic, reproducible, **zero-deletion** data quality audit for the official CPRI Hackathon dataset (`CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx`).

### Dataset Metrics:
- **Training_Data**: Exactly **1,000 records** (866 Valid, 134 Invalid)
- **Test_Data**: Exactly **350 records**
- **Definite Data-Quality Failures (`data_quality_issue`)**:
  - `Training_Data`: **44 records (4.40%)** (all due to missing values `missing_any`).
  - `Test_Data`: **17 records (4.86%)** (all due to missing values `missing_any`).
- **Observational / Diagnostic Flags**:
  - `duplicate_measurement_pair`: 24 records (2.40%) in Training, 8 records (2.29%) in Test.
  - `sensor_s2_negative`: 1 record (0.10%) in Training, 1 record (0.29%) in Test.
  - `sensor_zero_reading`: 3 records (0.30%) in Training, 1 record (0.29%) in Test.

---

## 2. Strict Flag Categorization

| Flag Name | Category | Description | Training Count (%) | Test Count (%) |
| :--- | :---: | :--- | :---: | :---: |
| `missing_any` | Definite Quality Failure | `True` if any sensor column (`Sensor_S1`..`S4`) is NaN. | 44 (4.40%) | 17 (4.86%) |
| `missing_columns` | Audit Detail | String listing exact missing sensor names (e.g. `Sensor_S4`). | N/A | N/A |
| `missing_critical_measurement` | Definite Quality Failure | `True` if Voltage, Current, Ambient Temp, or Duration is missing. | 0 (0.00%) | 0 (0.00%) |
| `non_finite_value` | Definite Quality Failure | `True` ONLY if numeric value is `+Inf` or `-Inf` (`np.isinf()`). (NaNs excluded). | 0 (0.00%) | 0 (0.00%) |
| `duplicate_full_row` | Definite Quality Failure | `True` if 100% exact duplicate row across all columns including Test_ID. | 0 (0.00%) | 0 (0.00%) |
| `duplicate_test_id` | Definite Quality Failure | `True` if duplicate `Test_ID` string occurs. | 0 (0.00%) | 0 (0.00%) |
| `malformed_numeric` | Definite Quality Failure | `True` if non-numeric or unparseable string is present. | 0 (0.00%) | 0 (0.00%) |
| `invalid_voltage` | Definite Quality Failure | `True` if `Applied_Voltage_kV < 0` (Hard physical voltage law). | 0 (0.00%) | 0 (0.00%) |
| `invalid_current` | Definite Quality Failure | `True` if `Load_Current_A < 0` (Hard physical current law). | 0 (0.00%) | 0 (0.00%) |
| `invalid_duration` | Definite Quality Failure | `True` if `Test_Duration_min <= 0` (Hard physical duration law). | 0 (0.00%) | 0 (0.00%) |
| `invalid_ambient_temp` | Definite Quality Failure | `True` if `Ambient_Temperature_C < -273.15` (Below absolute zero). | 0 (0.00%) | 0 (0.00%) |
| `duplicate_measurement_pair` | Diagnostic Observation | `True` if identical measurement feature vector exists in another Test_ID. | 24 (2.40%) | 8 (2.29%) |
| `sensor_s2_negative` | Diagnostic Observation | `True` if `Sensor_S2 < 0` (Observed negative sensor reading). | 1 (0.10%) | 1 (0.29%) |
| `sensor_zero_reading` | Diagnostic Observation | `True` if `Sensor_S1`, `S2`, or `S3` hits exact 0.0000. | 3 (0.30%) | 1 (0.29%) |
| `data_quality_issue` | Combined Failure Flag | `True` if ANY definite quality failure is triggered. | 44 (4.40%) | 17 (4.86%) |

---

## 3. Relationship with Historical `Validity_Label`

In `Training_Data` (1,000 records, 866 Valid, 134 Invalid):

1. **`duplicate_measurement_pair` (24 records / 12 pairs)**:
   - **100.0% Invalid (24 Invalid, 0 Valid)**.
   - *Observation*: While tracked as a diagnostic observation rather than a hard corruption law, engineers consistently invalidated tests that generated exact duplicate measurement pairs.
2. **`sensor_s2_negative` (1 record)**:
   - **100.0% Invalid (1 Invalid, 0 Valid)** (`TRN-0346`).
3. **`sensor_zero_reading` (3 records)**:
   - **100.0% Invalid (3 Invalid, 0 Valid)** (`TRN-0346`, `TRN-0357`, `TRN-0512`).
4. **`missing_any` (44 records)**:
   - **29 Valid (65.9%) vs 15 Invalid (34.1%)**.
   - *Crucial Finding*: Engineers did NOT automatically invalidate a test simply because a single non-critical sensor (like `Sensor_S4`) was missing if electrical measurements and overall behavior were consistent.

---

## 4. Key Distinction for Person 1 (Regimes vs Quality Corruption)

> [!IMPORTANT]
> **Category 1 — Definite Data Quality Problems**:
> `missing_any`, `non_finite_value`, `duplicate_full_row`, `duplicate_test_id`, `malformed_numeric`, and hard physical laws represent true data quality corruptions.
>
> **Category 2 & 3 — Operating Regimes & Genuine Equipment Behaviour**:
> High load currents (up to 109.95 A) or high ambient temperatures (up to 55.0°C) are **GENUINE OPERATING REGIMES** (as stated in CPRI dataset README: *"Genuine operating-regime change is embedded in valid data"*).
>
> Do **NOT** use naive z-score or IQR thresholds (e.g. declaring current > 100A invalid) during your regime modeling. Genuine high-load testing is expected and valid!

---

## 5. Recommended Person 1 Usage Guidelines

1. **For Equipment Regime Clustering**:
   - Filter on `missing_any == False` or impute missing sensor values using physical sensor correlations (`Sensor_S1` vs `Sensor_S2`).
2. **For Task 01 Valid/Invalid Modeling**:
   - Include diagnostic flags `duplicate_measurement_pair`, `sensor_s2_negative`, and `sensor_zero_reading` as valuable features for downstream classifier training.
3. **For Task 02 Reference Parameter Prediction**:
   - Filter out `data_quality_issue == True` when training regression models to avoid fitting noise from missing `Sensor_S4` values.
