# CPRI Hackathon — Task 01

## Project
The Central Power Research Institute (CPRI) performs repeated electrical testing on heavy power apparatus and sensors. This hackathon focuses on developing robust machine learning pipelines for screening electrical test records, identifying unreliable recordings, predicting key reference parameters, and generating automated diagnostic summaries.

## Task 01
The objective of **Task 01** is to identify abnormal/unreliable electrical tests and classify each test record as **Valid** or **Invalid**.

## Person 2
This section of the repository is owned by **Person 2** (Data Quality, Anomaly Detection & Classification Lead).  
Current active focus: **P2-Stage 1 — Data Quality Audit** (Stage 2 modeling has not been started).

## Checks Implemented in P2-Stage 1
1. **Missing Values**: Identifies rows containing `NaN`/`Null` entries across sensor columns (`Sensor_S1`..`S4`) (`missing_any`, `missing_columns`).
2. **Non-Finite Values**: Detects non-finite numerical representations (`non_finite_value`).
3. **Duplicate Complete Rows**: Detects exact 100% duplicate records across all columns (`duplicate_full_row`).
4. **Duplicate Test_IDs**: Audits whether `Test_ID` strings repeat (`duplicate_test_id`).
5. **Duplicate Measurement Pairs**: Detects records sharing identical feature vectors across different `Test_ID`s (`duplicate_measurement_pair`).
6. **Malformed Numeric Values**: Checks for strings or unparseable numeric values (`malformed_numeric`).
7. **Justified Physical Constraints**: Checks for genuine physical violations such as `Applied_Voltage_kV < 0`, `Load_Current_A < 0`, `Test_Duration_min <= 0`, `Sensor_S2 < 0`, and exact zero sensor drops (`invalid_sensor_zero`).

## Important Principle
**Unusual values are NOT automatically considered invalid.**  
As noted in the CPRI dataset specification, *genuine operating-regime changes are embedded in valid data*. High currents (up to 109.95 A) or high ambient temperatures (up to 55.0°C) represent valid high-load operating regimes. Stage 1 focuses strictly on Category 1 Data Quality corruption (missing values, DAQ drops, duplicate pairs) without using naive z-score or IQR cutoffs to discard valid high-load regimes.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute the Data Quality Audit pipeline
python src/run_audit_pipeline.py

# 3. Run automated pytest validation suite
pytest tests/test_quality_features.py -v
```

## Generated Outputs (`outputs/`)

- `outputs/data_quality_audit_training.csv`: Full `Training_Data` (1,000 records) enriched with deterministic row-level quality flags.
- `outputs/data_quality_audit_test.csv`: Full `Test_Data` (350 records) enriched with deterministic row-level quality flags.
- `outputs/data_quality_summary.csv`: Dataset-level audit summary table comparing flag prevalence and Valid/Invalid distributions.
- `outputs/person2_stage1_handoff.md`: Executive handoff documentation for Person 1 (Equipment Behaviour & Regime Analysis Lead).
- `outputs/figures/`: Diagnostic plots illustrating missing value distributions, flag counts, and `Validity_Label` associations.
