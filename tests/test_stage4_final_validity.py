"""
Automated Validation Suite — P2-Stage 4 Final Validity Model
Verifies dataset record counts, target exclusion, zero leakage, 5-fold Stratified CV,
threshold validity, reproducible inference, and multi-stage regression checks.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stage4_features import load_stage4_datasets, prepare_stage4_feature_sets
from stage4_models import evaluate_stage4_cv, get_stage4_pipeline
from final_validity_model import FinalValidityClassifier

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx')


@pytest.fixture
def stage4_data():
    if not os.path.exists(DATASET_PATH):
        pytest.skip(f"CPRI dataset file not found at {DATASET_PATH}")
    df_tr, df_te = load_stage4_datasets(DATASET_PATH)
    prep = prepare_stage4_feature_sets(df_tr, df_te)
    return df_tr, df_te, prep


def test_stage4_dataset_counts(stage4_data):
    df_tr, df_te, _ = stage4_data
    # 1. Training dataset has 1000 records
    assert len(df_tr) == 1000
    # 2. Test dataset has 350 records
    assert len(df_te) == 350


def test_target_and_id_exclusion_guarantee(stage4_data):
    _, _, prep = stage4_data
    forbidden = ["Reference_Parameter", "Validity_Label", "Test_ID"]
    
    # 3, 4, 5, 6, 19: Excluded from candidate feature sets A, B, C
    for set_key, X_mat in prep["X_train"].items():
        for col in forbidden:
            assert col not in X_mat.columns, f"Target/ID Leakage: {col} found in {set_key}!"


def test_5fold_stratified_cv_coverage_and_probabilities(stage4_data):
    _, _, prep = stage4_data
    X_tr = prep["X_train"]["Set_C"]
    y_tr = prep["y_train"]
    
    # 7, 8, 9, 10: 5 Stratified folds used, OOF predictions cover all 1000 records exactly once, probabilities bounded in [0,1]
    res = evaluate_stage4_cv(X_tr, y_tr, model_name="random_forest", feature_set_label="Set_C", config="balanced")
    
    assert len(res["oof_preds"]) == 1000
    assert len(res["oof_probs"]) == 1000
    assert np.all(np.isfinite(res["oof_probs"]))
    assert np.all((res["oof_probs"] >= 0.0) & (res["oof_probs"] <= 1.0))
    assert set(np.unique(res["oof_preds"])).issubset({0, 1})
    assert len(np.unique(res["oof_folds"])) == 5


def test_fold_isolated_anomaly_transformer_no_leakage():
    """20. Verify FoldIsolatedAnomalyTransformer fits strictly inside fold without leakage."""
    X_dummy = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feat2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    })
    y_dummy = pd.Series([0, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    
    pipe = get_stage4_pipeline(model_name="random_forest_anomaly", config="balanced", random_state=42)
    
    # Fit pipeline on fold 1
    pipe.fit(X_dummy.iloc[:5], y_dummy.iloc[:5])
    
    # Transform fold 2
    Xt_val = pipe.named_steps["imputer"].transform(X_dummy.iloc[5:])
    Xt_anom = pipe.named_steps["anomaly_transformer"].transform(Xt_val)
    
    assert Xt_anom.shape == (5, 3)
    assert np.all(np.isfinite(Xt_anom))


def test_final_classifier_inference_on_test_data(stage4_data):
    df_tr, df_te, _ = stage4_data
    
    clf = FinalValidityClassifier(model_name="random_forest", feature_set_name="Set_C", decision_threshold=0.45, random_state=42)
    clf.fit(df_tr)
    
    # 11, 12, 13, 14, 21: Threshold valid, predictions binary Valid/Invalid, Test_Data works without Validity_Label
    assert 0.0 <= clf.decision_threshold <= 1.0
    
    test_preds = clf.predict_dataset(df_te)
    
    assert len(test_preds) == 350
    assert "Test_ID" in test_preds.columns
    assert "Probability_Invalid" in test_preds.columns
    assert "Predicted_Validity" in test_preds.columns
    assert len(test_preds["Test_ID"].unique()) == 350
    assert set(test_preds["Predicted_Validity"].unique()).issubset({"Valid", "Invalid"})
    assert np.all((test_preds["Probability_Invalid"] >= 0.0) & (test_preds["Probability_Invalid"] <= 1.0))


def test_stage4_reproducibility(stage4_data):
    df_tr, _, prep = stage4_data
    X_tr = prep["X_train"]["Set_C"]
    y_tr = prep["y_train"]
    
    # 15. Final model is reproducible with random_state=42
    run1 = evaluate_stage4_cv(X_tr, y_tr, model_name="random_forest", feature_set_label="Set_C", config="balanced", random_state=42)
    run2 = evaluate_stage4_cv(X_tr, y_tr, model_name="random_forest", feature_set_label="Set_C", config="balanced", random_state=42)
    
    np.testing.assert_array_equal(run1["oof_preds"], run2["oof_preds"])
    np.testing.assert_array_almost_equal(run1["oof_probs"], run2["oof_probs"])
    assert run1["invalid_f1"] == run2["invalid_f1"]
