# Person 1 — Stage 7: Gap Analysis

## Executive Summary
This document provides a gap analysis comparing Person 1 Stage 7 requirements against the existing codebase (`AIHackthon`).
Existing Person 1 Stages 1–6 and Person 2 Stages 1–4 provide dataset loaders, quality flags, physical residual models, Stage 5 feature matrix (63 features), and Stage 6 baseline detectors.
Stage 7 will construct leakage-free 5-fold Stratified Cross-Validation pipelines across 5 feature sets and 3 supervised model architectures, optimize the decision threshold, evaluate feature importance, perform error analysis, compare supervised vs unsupervised signals, and generate complete outputs and 20 unit tests.

---

## Gap Analysis Table

| Requirement | Already Implemented | Partially Implemented | Missing | Reusable Existing Module | Required Action |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **1. Dataset Loader & Quality Flags** | 🟢 `load_training_data`, `create_quality_flags` | ❌ | ❌ Complete | `src/dataset_loader.py`, `src/quality_features.py` | Reuse directly |
| **2. P1 Stage 4 Physical Residual Features** | 🟢 `NormalBehaviourModels` | ❌ | ❌ Complete | `src/stage4_behaviour.py` | Reuse directly |
| **3. P1 Stage 5 Feature Engineering Matrix** | 🟢 `create_stage5_features` | ❌ | ❌ Complete | `src/stage5_features.py` | Reuse directly for Feature Sets A, B, C, D, E |
| **4. P1 Stage 6 / P2 Stage 3 Anomaly Score** | 🟢 `IsolationForestAnomalyDetector` | ❌ | ❌ Complete | `src/stage3_anomaly.py`, `src/stage6_baselines.py` | Incorporate Isolation Forest score into Feature Set E |
| **5. 5-Fold Stratified CV Pipeline** | 🟢 In P2 Stage 4 | ❌ | ⚠️ Missing for P1 Stage 7 models | `src/stage4_models.py` | Implement `src/stage7_ml_models.py` with 5-fold Stratified CV across Sets A–E |
| **6. Candidate Models** (Logistic Regression, Random Forest, HistGradientBoosting) | 🟢 In P2 Stage 4 | ❌ | ⚠️ Missing for P1 Stage 7 evaluation | `src/stage4_models.py` | Implement model wrappers in `src/stage7_ml_models.py` |
| **7. Decision Threshold Optimization** | 🟢 In P2 Stage 4 | ❌ | ⚠️ Missing for P1 Stage 7 winning model | `src/final_validity_model.py` | Implement OOF threshold evaluation (0.10–0.90) in `src/stage7_ml_models.py` |
| **8. Feature Importance / Permutation Importance** | ❌ | ❌ | ⚠️ Missing P1 Stage 7 importance | None | Implement `calculate_permutation_importance()` in `src/stage7_ml_models.py` |
| **9. False Positive & False Negative Analysis** | 🟢 In P1 Stage 6 & P2 Stage 4 | ❌ | ⚠️ Missing P1 Stage 7 ML FP/FN analysis | `src/stage6_baselines.py` | Implement `analyze_ml_errors()` in `src/stage7_ml_models.py` |
| **10. Stage 7 Pipeline Runner Script** | ❌ | ❌ | ⚠️ Missing runner script | `src/run_stage6_baselines.py` | Implement `src/run_stage7_ml_models.py` |
| **11. Markdown Reports & Handoff** | ❌ | ❌ | ⚠️ Missing Stage 7 reports | None | Generate `outputs/p1_stage7_report.md` & `outputs/p1_stage7_handoff.md` |
| **12. Interactive Notebook & Unit Tests** | ❌ | ❌ | ⚠️ Missing notebook & 20 tests | None | Create `notebooks/person1_stage7_ml_classification.ipynb` & `tests/test_person1_stage7_ml_models.py` |

---

## Action Plan
1. Implement `src/stage7_ml_models.py` providing 5 feature set definitions (A: Raw, B: Raw+Quality, C: Raw+Residual, D: Full Stage 5, E: Full Stage 5 + Anomaly Score), 3 model classifiers, 5-fold Stratified CV evaluation, threshold tuning, permutation importance, and error analysis.
2. Implement `src/run_stage7_ml_models.py` to execute the Stage 7 ML evaluation, export all 7 CSV reports, 7 diagnostic figures, and markdown reports.
3. Write `notebooks/person1_stage7_ml_classification.ipynb` and `tests/test_person1_stage7_ml_models.py` (20 unit tests).
4. Run full test suite (`python -m pytest tests/ -v`), update `README.md`, commit (`Implement P1 Stage 7 ML classification and anomaly models`), and push to `origin/main`.
