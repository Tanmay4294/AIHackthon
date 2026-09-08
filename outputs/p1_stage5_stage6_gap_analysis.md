# Person 1 — Stage 5 & Stage 6 Gap Analysis

## Executive Summary
This document provides a gap analysis comparing Person 1 Stage 5 and Stage 6 requirements against the existing codebase (`AIHackthon`).
Existing Person 1 Stages 1–4 and Person 2 Stages 1–4 provide dataset loaders, quality flags, baseline classifiers, unsupervised anomaly models, and physical residual/consistency features.

---

## Gap Analysis Table

| Requirement | Already Implemented | Partially Implemented | Missing | Reusable Existing Module | Action Taken |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **1. Sensor Differences** (`S1_minus_S2`, `S1_minus_S3`, etc.) | ❌ | ❌ | ⚠️ Missing | None | Implemented `create_sensor_difference_features()` in `src/stage5_features.py` |
| **2. Sensor Aggregates** (mean, median, std across S1..S3 & S1..S4) | ❌ | 🟢 `Sensor_Mean` | ⚠️ Missing S1..S3 & median/std | `src/stage3_features.py` | Implemented `create_sensor_aggregate_features()` in `src/stage5_features.py` |
| **3. Sensor Disagreement** (`max_minus_min`, pairwise abs diffs) | 🟢 `Sensor_Spread` | ❌ | ⚠️ Missing pairwise abs diffs | `src/quality_features.py` | Implemented `create_sensor_disagreement_features()` in `src/stage5_features.py` |
| **4. Physical Residuals** (`p1_max_abs_residual`, `p1_consistency_index`) | 🟢 `NormalBehaviourModels` | ❌ | ❌ Complete | `src/stage4_behaviour.py` | Reused `NormalBehaviourModels.transform()` directly in `src/stage5_features.py` |
| **5. Deterministic Quality Flags** (`missing_any`, `non_finite_value`, etc.) | 🟢 `create_quality_flags` | ❌ | ❌ Complete | `src/quality_features.py` | Reused `create_quality_flags()` directly in `src/stage5_features.py` |
| **6. Physical Interactions** (`Thermal_Loading_Index`, `Apparent_Power_kVA`) | 🟢 `Apparent_Power_kVA` | ❌ | ⚠️ Missing thermal & impedance | `src/stage3_features.py` | Implemented `create_interaction_features()` in `src/stage5_features.py` |
| **7. Sensor S4 Investigation** | ❌ | ❌ | ⚠️ Missing comparative evaluation | `src/stage4_behaviour.py` | Implemented `compare_s4_information()` in `src/stage5_features.py` |
| **8. Baseline Anomaly Detectors** (Quality, IQR, Z-Score, IF, LOF, Residual) | 🟢 IF & LOF in P2 Stage 3 | ❌ | ⚠️ Missing Quality, IQR, Z-score, Residual baselines | `src/stage3_anomaly.py` | Implemented `src/stage6_baselines.py` and `src/run_stage6_baselines.py` |
