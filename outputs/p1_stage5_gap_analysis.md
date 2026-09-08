# Person 1 — Stage 5: Gap Analysis

## Executive Summary
This document provides a gap analysis comparing Person 1 Stage 5 requirements against the existing codebase (`AIHackthon`).
Existing Person 1 Stages 1–4 and Person 2 Stages 1–4 already provide dataset loaders, quality flags, baseline classifiers, unsupervised anomaly models, and physical residual/consistency features.
Stage 5 will consolidate and expand engineered features while preventing feature duplication.

---

## Gap Analysis Table

| Stage 5 Requirement | Already Implemented | Partially Implemented | Missing | Reusable Existing Module | Required New Implementation |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **1. Sensor Difference Features** (`S1_minus_S2`, `S1_minus_S3`, `S2_minus_S3`, `S4` differences) | ❌ | ❌ | ⚠️ Missing | None | Implement `create_sensor_difference_features()` in `src/stage5_features.py` |
| **2. Sensor Aggregate Features** (mean/median/std across S1..S3 & S1..S4) | ❌ | 🟢 `Sensor_Mean` in P2 | ⚠️ Missing S1..S3 & median/std | `src/stage3_features.py` | Implement `create_sensor_aggregate_features()` in `src/stage5_features.py` |
| **3. Sensor Disagreement Features** (`max_minus_min`, pairwise abs diffs, mean/max pairwise diffs) | 🟢 `Sensor_Spread`, `p1_sensor_disagreement_index` | ❌ | ⚠️ Missing pairwise abs diffs | `src/quality_features.py`, `src/stage4_behaviour.py` | Implement `create_sensor_disagreement_features()` in `src/stage5_features.py` |
| **4. Relative / Normalized Residual Features** (`p1_max_abs_residual`, `p1_consistency_index`, etc.) | 🟢 `NormalBehaviourModels` | ❌ | ❌ Complete | `src/stage4_behaviour.py` | Reuse `NormalBehaviourModels.transform()` directly in `src/stage5_features.py` |
| **5. Deterministic Data-Quality Features** (`missing_any`, `non_finite_value`, `invalid_voltage`, etc.) | 🟢 `create_quality_flags` | ❌ | ❌ Complete | `src/quality_features.py` | Reuse `create_quality_flags()` directly in `src/stage5_features.py` |
| **6. Interaction Features** (`Load_Current_A * Test_Duration_min`, `Apparent_Power_kVA`, etc.) | 🟢 `Apparent_Power_kVA` | ❌ | ⚠️ Missing thermal load & impedance ratio | `src/stage3_features.py` | Implement `create_interaction_features()` in `src/stage5_features.py` |
| **7. Sensor S4 Information Investigation** | ❌ | ❌ | ⚠️ Missing systematic evaluation | `src/stage4_behaviour.py` | Implement `compare_s4_information()` in `src/stage5_features.py` |
| **8. Stage 5 Feature Matrix Module** (`src/stage5_features.py`) | ❌ | ❌ | ⚠️ Missing unified Stage 5 module | `src/dataset_loader.py` | Implement `src/stage5_features.py` and `src/run_stage5_features.py` |
| **9. Feature Dictionary & Summary CSVs** | ❌ | ❌ | ⚠️ Missing dictionary & summary CSVs | None | Generate `p1_stage5_feature_dictionary.csv` & `p1_stage5_feature_matrix_summary.csv` |
| **10. Stage 5 Feature Engineering Report & Handoff** | ❌ | ❌ | ⚠️ Missing reports | None | Generate `outputs/p1_stage5_feature_engineering_report.md` & `outputs/p1_stage5_handoff.md` |

---

## Action Plan
1. Construct `src/stage5_features.py` bringing together deterministic quality flags, sensor differences, sensor aggregates, sensor disagreement features, Stage 4 physical residuals, and physical interactions.
2. Implement S4 comparative analysis (`compare_s4_information`) testing feature sets WITH vs WITHOUT `Sensor_S4`.
3. Create `src/run_stage5_features.py` to run the feature pipeline, export feature dictionary, matrix summaries, and reports.
