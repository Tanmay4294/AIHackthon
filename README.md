# CPRI Hackathon — Task 01

## Project
The Central Power Research Institute (CPRI) performs repeated electrical testing on heavy power apparatus and sensors. This hackathon focuses on developing robust machine learning pipelines for screening electrical test records, identifying unreliable recordings, predicting key reference parameters, and generating automated diagnostic summaries.

## Task 01
The objective of **Task 01** is to identify abnormal/unreliable electrical tests and classify each test record as **Valid** or **Invalid**.

---

## Person 1 Stages & Progress
This section is owned by **Person 1** (Contextual, Regime & Feature Engineering Lead).

- **Stage 1 (Dataset & Environment Setup)**: Modular dataset loader (`src/dataset_loader.py`), programmatic Excel sheet inspection, schema separation (isolating physical features from targets and IDs), relative path resolution, setup notebook (`notebooks/person1_stage1_dataset_setup.ipynb`), and 18 automated unit tests.
- **Stage 2 (Data-Quality Audit)**: Comprehensive, reproducible, non-destructive audit layer (`src/data_quality_audit.py`). Directly reuses Person 2 Stage 1 quality flags (`quality_features.py`) and dataset loader (`dataset_loader.py`). Enforces zero data deletion (1,000 training and 350 test rows preserved) and zero target leakage. Generates 7 CSV reports, 8 figures in `outputs/figures/`, markdown report `outputs/p1_stage2_data_quality_report.md`, interactive notebook (`notebooks/person1_stage2_data_quality_audit.ipynb`), pipeline script (`src/run_data_quality_audit.py`), and 16 unit tests.
- **Stage 3 (Valid vs Invalid Historical Analysis)**: Rigorous exploratory analysis module (`src/stage3_historical_analysis.py`) investigating patterns distinguishing historical Valid (866 records, 86.6%) vs Invalid (134 records, 13.4%) tests. Evaluates class-wise statistics, point-biserial / Spearman correlations, Cohen's d effect sizes, sensor consistency, recurring invalid patterns, legitimate unusual operating regimes, representative record case studies, and ranked feature shortlists. Generates 5 CSV reports, 8 figures in `outputs/figures/`, markdown report `outputs/p1_stage3_valid_invalid_report.md`, interactive notebook (`notebooks/person1_stage3_valid_invalid_analysis.ipynb`), pipeline script (`src/run_stage3_historical_analysis.py`), and 14 unit tests.
- **Stage 4 (Behaviour & Physical Consistency Analysis)**: Normal-behaviour reference regression models (`src/stage4_behaviour.py`) trained **STRICTLY on Valid historical test records** ($N=700$) to predict expected sensor values under varying operating conditions. Generates residual and consistency features (`p1_max_abs_residual`, `p1_consistency_index`, `p1_sensor_disagreement_index`, `p1_regime_cluster`) without target leakage or row deletion. Generates 9 CSV reports, 12 figures in `outputs/figures/`, markdown report `outputs/p1_stage4_behaviour_consistency_report.md`, handoff report `outputs/p1_stage4_handoff.md`, notebook (`notebooks/person1_stage4_behaviour_consistency_analysis.ipynb`), pipeline script (`src/run_stage4_behaviour.py`), updated feature adapter (`src/person1_adapter.py`), and 15 unit tests.
- **Stage 5 (Feature Engineering)**: Consolidated, inference-safe feature engineering module (`src/stage5_features.py`) combining quality flags, sensor differences (`S1_minus_S2`, `S1_minus_S3`, `S2_minus_S3`), sensor aggregates (mean, median, std across S1..S3 & S1..S4), sensor disagreement metrics, physical residuals (`p1_max_abs_residual`), physical interactions (`Thermal_Loading_Index`, `Apparent_Power_kVA`, `Apparent_Impedance_Proxy`), and comparative `Sensor_S4` investigation. Generates feature dictionary (`outputs/p1_stage5_feature_dictionary.csv`), summary CSVs, markdown report `outputs/p1_stage5_feature_engineering_report.md`, handoff report `outputs/p1_stage5_handoff.md`, pipeline script (`src/run_stage5_features.py`), notebook (`notebooks/person1_stage5_feature_engineering.ipynb`), and 15 unit tests.
- **Stage 6 (Build Baseline Detectors)**: Comprehensive evaluation framework (`src/stage6_baselines.py`) implementing 6 baseline detectors: Deterministic Quality Rules, Robust Statistical IQR Outliers, Z-Score Outliers, Isolation Forest, LOF, and Residual/Consistency Thresholds. Includes Precision/Recall/F1/ROC-AUC evaluation, False Positive & False Negative analysis, and operating regime performance breakdown across Heavy HV Load, Heavy Current, High Voltage, and Standard regimes. Generates 6 CSV reports, 6 figures in `outputs/figures/`, report `outputs/p1_stage6_baseline_report.md`, handoff report `outputs/p1_stage6_handoff.md`, pipeline script (`src/run_stage6_baselines.py`), notebook (`notebooks/person1_stage6_baseline_detectors.ipynb`), and 15 unit tests (118 total tests passing 100%).

---

## Person 2 Stages & Progress
This section is owned by **Person 2** (Data Quality, Anomaly Detection & Classification Lead).

- **Stage 1 (Data Quality Audit)**: Deterministic, zero-deletion quality auditing (`missing_any`, duplicate checks, physical constraints).
- **Stage 2 (Supervised Classification Baselines)**: 5-fold Stratified CV evaluation across 8 baseline configurations.
- **Stage 3 (Unsupervised Anomaly Detection)**: Independent Isolation Forest & LOF anomaly scoring, score direction standardization (higher = more anomalous), and 4-group agreement analysis.
- **Stage 4 (Final Validity Model)**: Final reproducible Valid vs Invalid classifier combining raw parameters, Stage 1 quality flags, and validated Person 2 fallback engineered features. Includes decision threshold optimization, false-positive/negative error analysis, and final inference on `Test_Data` (350 rows).

---

## Checks Implemented in P2-Stage 1 & Reused in P1-Stage 2/3
1. **Missing Values (`missing_any`)**: Identifies rows containing `NaN`/`Null` entries across measurement columns (`Sensor_S1`..`S4`).
2. **Non-Finite Values (`non_finite_value`)**: Detects ONLY actual `+Infinity` or `-Infinity` values (`np.isinf()`). NaNs are excluded and reported under missing values.
3. **Duplicate Complete Rows (`duplicate_full_row`)**: Detects exact 100% duplicate records across all columns including `Test_ID`.
4. **Duplicate Test_IDs (`duplicate_test_id`)**: Audits whether `Test_ID` strings repeat.
5. **Duplicate Measurement Pairs (`duplicate_measurement_pair`)**: Diagnostic observation flag detecting records sharing identical measurement feature vectors across different `Test_ID`s.
6. **Malformed Numeric Values (`malformed_numeric`)**: Checks for unparseable numeric strings or values.
7. **Justified Physical Constraints**: Enforces ONLY genuinely justified physical laws (e.g. `Applied_Voltage_kV < 0`, `Load_Current_A < 0`, `Test_Duration_min <= 0`, `Ambient_Temperature_C < -273.15`).

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute Person 1 Stage 1 Setup Notebook
jupyter notebook notebooks/person1_stage1_dataset_setup.ipynb

# 3. Execute Person 1 Stage 2 Data-Quality Audit pipeline
python src/run_data_quality_audit.py

# 4. Execute Person 1 Stage 3 Valid vs Invalid Historical Analysis pipeline
python src/run_stage3_historical_analysis.py

# 5. Execute Person 1 Stage 4 Behaviour & Physical Consistency Analysis pipeline
python src/run_stage4_behaviour.py

# 6. Execute Person 1 Stage 5 Feature Engineering pipeline
python src/run_stage5_features.py

# 7. Execute Person 1 Stage 6 Baseline Detectors pipeline
python src/run_stage6_baselines.py

# 8. Execute Person 2 Stage 1 Audit pipeline
python src/run_audit_pipeline.py

# 9. Execute Person 2 Stage 2 Baseline Supervised Classification pipeline
python src/run_stage2_baseline.py

# 10. Execute Person 2 Stage 3 Unsupervised Anomaly Detection pipeline
python src/run_stage3_anomaly.py

# 11. Execute Person 2 Stage 4 Final Validity Classifier pipeline
python src/run_stage4_final.py

# 12. Run complete automated pytest suite across all test modules (118 tests)
python -m pytest tests/ -v
```

---

## Generated Outputs (`outputs/`)

- `outputs/p1_stage1_gap_analysis.md` (Person 1 Stage 1)
- `outputs/p1_stage2_gap_analysis.md` & `p1_stage2_data_quality_report.md` (Person 1 Stage 2)
- `outputs/p1_stage2_missing_value_report.csv` (Person 1 Stage 2)
- `outputs/p1_stage2_duplicate_report.csv` & `p1_stage2_repeated_test_id_report.csv` (Person 1 Stage 2)
- `outputs/p1_stage2_numeric_statistics.csv` & `p1_stage2_outlier_candidates.csv` (Person 1 Stage 2)
- `outputs/p1_stage2_quality_flags_training.csv` & `p1_stage2_quality_flags_test.csv` (Person 1 Stage 2)
- `outputs/p1_stage3_gap_analysis.md` & `p1_stage3_valid_invalid_report.md` (Person 1 Stage 3)
- `outputs/p1_stage3_class_statistics.csv` & `p1_stage3_feature_associations.csv` (Person 1 Stage 3)
- `outputs/p1_stage3_pattern_analysis.csv` & `p1_stage3_regime_analysis.csv` (Person 1 Stage 3)
- `outputs/p1_stage3_representative_records.csv` (Person 1 Stage 3)
- `outputs/p1_stage4_gap_analysis.md` & `p1_stage4_behaviour_consistency_report.md` (Person 1 Stage 4)
- `outputs/p1_stage4_relationship_analysis.csv` & `p1_stage4_regime_analysis.csv` (Person 1 Stage 4)
- `outputs/p1_stage4_normal_model_performance.csv` & `p1_stage4_class_residuals.csv` (Person 1 Stage 4)
- `outputs/p1_stage4_sensor_reliability.csv` & `p1_stage4_candidate_features.csv` (Person 1 Stage 4)
- `outputs/p1_stage4_representative_records.csv` (Person 1 Stage 4)
- `outputs/p1_stage4_training_residuals.csv` & `p1_stage4_test_residuals.csv` (Person 1 Stage 4)
- `outputs/p1_stage4_handoff.md` (Person 1 Stage 4)
- `outputs/p1_stage5_gap_analysis.md` & `p1_stage5_feature_engineering_report.md` (Person 1 Stage 5)
- `outputs/p1_stage5_feature_dictionary.csv` & `p1_stage5_feature_matrix_summary.csv` (Person 1 Stage 5)
- `outputs/p1_stage5_s4_investigation.csv` & `p1_stage5_handoff.md` (Person 1 Stage 5)
- `outputs/p1_stage6_baseline_comparison.csv` & `p1_stage6_baseline_predictions_training.csv` (Person 1 Stage 6)
- `outputs/p1_stage6_false_positive_analysis.csv` & `p1_stage6_false_negative_analysis.csv` (Person 1 Stage 6)
- `outputs/p1_stage6_regime_performance.csv` & `p1_stage6_threshold_analysis.csv` (Person 1 Stage 6)
- `outputs/p1_stage6_baseline_report.md` & `p1_stage6_handoff.md` (Person 1 Stage 6)
- `outputs/data_quality_audit_training.csv` & `data_quality_audit_test.csv` (Person 2 Stage 1)
- `outputs/p2_stage2_baseline_comparison.csv` & `p2_stage2_oof_predictions.csv` (Person 2 Stage 2)
- `outputs/p2_stage3_anomaly_scores_training.csv` & `p2_stage3_anomaly_scores_test.csv` (Person 2 Stage 3)
- `outputs/p2_stage4_model_comparison.csv` & `p2_stage4_threshold_analysis.csv` (Person 2 Stage 4)
- `outputs/p2_stage4_oof_predictions.csv` & `p2_stage4_final_test_predictions.csv` (Person 2 Stage 4)
- `outputs/p2_stage4_false_positive_analysis.csv` & `p2_stage4_false_negative_analysis.csv` (Person 2 Stage 4)
- `outputs/p2_stage4_report.md` & `outputs/p2_stage4_handoff.md` (Person 2 Stage 4)
- `outputs/figures/`: Diagnostic plots for all stages (including P1 Stage 2, Stage 3, Stage 4, Stage 6).
