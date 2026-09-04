"""
Automated Validation Suite — P2-Stage 2 Supervised Classification Baseline
Verifies dataset integrity, target exclusion, constant feature handling, leakage-free pipelines,
5-fold Stratified CV, metric calculations, and reproducibility.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stage2_features import load_training_data_with_flags, prepare_stage2_feature_sets
from stage2_models import evaluate_rule_baseline, evaluate_model_cv, get_model_pipeline

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx')


@pytest.fixture
def stage2_data():
    if not os.path.exists(DATASET_PATH):
        pytest.skip(f"CPRI dataset file not found at {DATASET_PATH}")
    df_flagged = load_training_data_with_flags(DATASET_PATH)
    prep = prepare_stage2_feature_sets(df_flagged)
    return df_flagged, prep


def test_row_counts_and_class_distribution(stage2_data):
    df_flagged, prep = stage2_data
    
    # 1. Training dataset contains 1000 rows
    assert len(df_flagged) == 1000
    
    # 2. Valid + Invalid counts equal total rows (866 + 134 = 1000)
    valid_count = (df_flagged["Validity_Label"].str.strip() == "Valid").sum()
    invalid_count = (df_flagged["Validity_Label"].str.strip() == "Invalid").sum()
    
    assert valid_count == 866
    assert invalid_count == 134
    assert valid_count + invalid_count == 1000
    assert prep["y"].sum() == 134


def test_target_and_id_exclusion_guarantee(stage2_data):
    _, prep = stage2_data
    X_raw = prep["X_raw"]
    X_raw_quality = prep["X_raw_quality"]
    
    forbidden = ["Reference_Parameter", "Validity_Label", "Test_ID"]
    
    # 3, 4, 5: Excluded from feature sets A and B
    for col in forbidden:
        assert col not in X_raw.columns, f"Target/ID Leakage: {col} in Feature Set A!"
        assert col not in X_raw_quality.columns, f"Target/ID Leakage: {col} in Feature Set B!"


def test_constant_feature_detection_and_exclusion(stage2_data):
    _, prep = stage2_data
    
    # 7. Constant features detected and excluded
    const_feats = prep["constant_features"]
    assert len(const_feats) > 0, "Constant features should be detected in Training_Data!"
    
    # Confirm zero variance features like non_finite_value, duplicate_full_row are in const_feats
    assert "non_finite_value" in const_feats
    assert "duplicate_full_row" in const_feats
    assert "duplicate_test_id" in const_feats
    
    # Confirm none of the constant features exist in active feature matrices
    for c_col in const_feats.keys():
        assert c_col not in prep["X_raw"].columns
        assert c_col not in prep["X_raw_quality"].columns


def test_pipeline_leakage_isolation():
    """Verify SimpleImputer and StandardScaler fit ONLY on fold training portion."""
    # Synthetic data with missing values
    X_dummy = pd.DataFrame({
        "feat1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feat2": [10.0, 20.0, 30.0, 40.0, np.nan, 60.0, 70.0, 80.0, 90.0, 100.0]
    })
    y_dummy = pd.Series([0, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    
    pipe = get_model_pipeline(model_name="logistic_regression", config="balanced", random_state=42)
    
    # Fit pipeline
    pipe.fit(X_dummy.iloc[:5], y_dummy.iloc[:5])
    
    # Check that imputer median was calculated ONLY on training fold (first 5 rows)
    imputer = pipe.named_steps["imputer"]
    # For feat1, first 5 rows are [1.0, 2.0, NaN, 4.0, 5.0] -> median is 3.0
    assert imputer.statistics_[0] == 3.0


def test_5fold_stratified_cv_evaluation(stage2_data):
    _, prep = stage2_data
    X_raw = prep["X_raw"]
    y = prep["y"]
    
    # 8, 10, 11: Execute 5-fold Stratified CV on Logistic Regression
    res = evaluate_model_cv(X_raw, y, model_name="logistic_regression", feature_set_label="Raw", config="balanced")
    
    assert res["model_name"] == "Logistic Regression (Balanced)"
    assert len(res["oof_preds"]) == 1000
    assert len(res["oof_probs"]) == 1000
    assert "invalid_precision" in res
    assert "invalid_recall" in res
    assert "invalid_f1" in res
    assert "balanced_accuracy" in res
    assert res["confusion_matrix"].shape == (2, 2)


def test_all_8_baseline_configurations_produce_outputs(stage2_data):
    _, prep = stage2_data
    X_raw = prep["X_raw"]
    X_quality = prep["X_raw_quality"]
    y = prep["y"]
    
    configs = [
        ("Rule_Definite", lambda: evaluate_rule_baseline(X_quality, y, rule_type="definite_issue")),
        ("Rule_Dup", lambda: evaluate_rule_baseline(X_quality, y, rule_type="definite_plus_duplicate")),
        ("LR_Raw", lambda: evaluate_model_cv(X_raw, y, "logistic_regression", "Raw", config="balanced")),
        ("LR_Quality", lambda: evaluate_model_cv(X_quality, y, "logistic_regression", "Quality", config="balanced")),
        ("RF_Raw", lambda: evaluate_model_cv(X_raw, y, "random_forest", "Raw", config="balanced")),
        ("RF_Quality", lambda: evaluate_model_cv(X_quality, y, "random_forest", "Quality", config="balanced")),
        ("GB_Raw", lambda: evaluate_model_cv(X_raw, y, "gradient_boosting", "Raw")),
        ("GB_Quality", lambda: evaluate_model_cv(X_quality, y, "gradient_boosting", "Quality"))
    ]
    
    assert len(configs) == 8
    
    for name, fn in configs:
        res = fn()
        assert len(res["oof_preds"]) == 1000
        assert 0.0 <= res["invalid_f1"] <= 1.0


def test_reproducibility(stage2_data):
    _, prep = stage2_data
    X_raw = prep["X_raw"]
    y = prep["y"]
    
    run1 = evaluate_model_cv(X_raw, y, model_name="random_forest", feature_set_label="Raw", config="balanced", random_state=42)
    run2 = evaluate_model_cv(X_raw, y, model_name="random_forest", feature_set_label="Raw", config="balanced", random_state=42)
    
    np.testing.assert_array_equal(run1["oof_preds"], run2["oof_preds"])
    np.testing.assert_array_almost_equal(run1["oof_probs"], run2["oof_probs"])
    assert run1["invalid_f1"] == run2["invalid_f1"]
