# CPRI Hackathon — Task 01

## Project
The Central Power Research Institute (CPRI) performs repeated electrical testing on heavy power apparatus and sensors. This hackathon focuses on developing robust machine learning pipelines for screening electrical test records, identifying unreliable recordings, predicting key reference parameters, and generating automated diagnostic summaries.

## Task 01
The objective of **Task 01** is to identify abnormal/unreliable electrical tests and classify each test record as **Valid** or **Invalid**.

## Person 2
## Person 2 Stages & Progress
This repository section is owned by **Person 2** (Data Quality, Anomaly Detection & Classification Lead).

- **Stage 1 (Data Quality Audit)**: Deterministic, zero-deletion quality auditing (`missing_any`, duplicate checks, physical constraints).
- **Stage 2 (Supervised Classification Baselines)**: 5-fold Stratified CV evaluation across 8 baseline configurations.
- **Stage 3 (Unsupervised Anomaly Detection)**: Independent Isolation Forest & LOF anomaly scoring, score direction standardization (higher = more anomalous), and 4-group agreement analysis.
- **Stage 4 (Final Validity Model)**: Final reproducible Valid vs Invalid classifier combining raw parameters, Stage 1 quality flags, and validated Person 2 fallback engineered features. Includes decision threshold optimization, false-positive/negative error analysis, and final inference on `Test_Data` (350 rows).

## Checks Implemented in P2-Stage 1
1. **Missing Values (`missing_any`)**: Identifies rows containing `NaN`/`Null` entries across measurement columns (`Sensor_S1`..`S4`).
2. **Non-Finite Values (`non_finite_value`)**: Detects ONLY actual `+Infinity` or `-Infinity` values (`np.isinf()`). NaNs are excluded and reported under missing values.
3. **Duplicate Complete Rows (`duplicate_full_row`)**: Detects exact 100% duplicate records across all columns including `Test_ID`.
4. **Duplicate Test_IDs (`duplicate_test_id`)**: Audits whether `Test_ID` strings repeat.
5. **Duplicate Measurement Pairs (`duplicate_measurement_pair`)**: Diagnostic observation flag detecting records sharing identical measurement feature vectors across different `Test_ID`s.
6. **Malformed Numeric Values (`malformed_numeric`)**: Checks for unparseable numeric strings or values.
7. **Justified Physical Constraints**: Enforces ONLY genuinely justified physical laws (e.g. `Applied_Voltage_kV < 0`, `Load_Current_A < 0`, `Test_Duration_min <= 0`, `Ambient_Temperature_C < -273.15`).

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute Stage 1 Audit pipeline
python src/run_audit_pipeline.py

# 3. Execute Stage 2 Baseline Supervised Classification pipeline
python src/run_stage2_baseline.py

# 4. Execute Stage 3 Unsupervised Anomaly Detection pipeline
python src/run_stage3_anomaly.py

# 5. Execute Stage 4 Final Validity Classifier pipeline
python src/run_stage4_final.py

# 6. Run complete automated pytest suite across all 4 stages
python -m pytest tests/ -v
```

## Generated Outputs (`outputs/`)

- `outputs/data_quality_audit_training.csv` & `data_quality_audit_test.csv` (Stage 1)
- `outputs/p2_stage2_baseline_comparison.csv` & `p2_stage2_oof_predictions.csv` (Stage 2)
- `outputs/p2_stage3_anomaly_scores_training.csv` & `p2_stage3_anomaly_scores_test.csv` (Stage 3)
- `outputs/p2_stage4_model_comparison.csv` & `p2_stage4_threshold_analysis.csv` (Stage 4)
- `outputs/p2_stage4_oof_predictions.csv` & `p2_stage4_final_test_predictions.csv` (Stage 4)
- `outputs/p2_stage4_false_positive_analysis.csv` & `p2_stage4_false_negative_analysis.csv` (Stage 4)
- `outputs/p2_stage4_report.md` & `outputs/p2_stage4_handoff.md` (Stage 4)
- `outputs/figures/`: Diagnostic plots for all 4 stages.
