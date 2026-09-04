# CPRI Hackathon — Task 01

## Project
The Central Power Research Institute (CPRI) performs repeated electrical testing on heavy power apparatus and sensors. This hackathon focuses on developing robust machine learning pipelines for screening electrical test records, identifying unreliable recordings, predicting key reference parameters, and generating automated diagnostic summaries.

## Task 01
The objective of **Task 01** is to identify abnormal/unreliable electrical tests and classify each test record as **Valid** or **Invalid**.

## Person 2
This section of the repository is owned by **Person 2** (Data Quality, Anomaly Detection & Classification Lead).  
Current active focus: **P2-Stage 1 — Data Quality Audit** (Stage 2 modeling has not been started).

## Checks Implemented in P2-Stage 1
1. **Missing Values (`missing_any`)**: Identifies rows containing `NaN`/`Null` entries across measurement columns (`Sensor_S1`..`S4`).
2. **Non-Finite Values (`non_finite_value`)**: Detects ONLY actual `+Infinity` or `-Infinity` values (`np.isinf()`). NaNs are excluded and reported under missing values.
3. **Duplicate Complete Rows (`duplicate_full_row`)**: Detects exact 100% duplicate records across all columns including `Test_ID`.
4. **Duplicate Test_IDs (`duplicate_test_id`)**: Audits whether `Test_ID` strings repeat.
5. **Duplicate Measurement Pairs (`duplicate_measurement_pair`)**: Diagnostic observation flag detecting records sharing identical measurement feature vectors across different `Test_ID`s. Maintained for downstream analysis; NOT treated as automatic data corruption.
6. **Malformed Numeric Values (`malformed_numeric`)**: Checks for unparseable numeric strings or values.
7. **Justified Physical Constraints**: Enforces ONLY genuinely justified physical laws (e.g. `Applied_Voltage_kV < 0`, `Load_Current_A < 0`, `Test_Duration_min <= 0`, `Ambient_Temperature_C < -273.15`). Observed dataset ranges (e.g. 8 to 32 kV) are reported as statistics and NOT enforced as arbitrary physical invalidity laws.

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
