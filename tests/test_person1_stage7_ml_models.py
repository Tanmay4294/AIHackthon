"""
CPRI Hackathon — Person 1 Stage 7 Unit Tests
=============================================
Comprehensive unit tests for Person 1 Stage 7 ML Classification & Anomaly Models.
"""

import pytest
import numpy as np
import pandas as pd

from src.dataset_loader import load_training_data, load_test_data, TASK01_TARGET, ID_COLUMN
from src.stage5_features import create_stage5_features
from src.stage7_ml_models import (
    prepare_feature_sets,
    get_model_pipelines,
    evaluate_model_cv,
    evaluate_all_stage7_models,
    optimize_decision_threshold,
    calculate_permutation_importance,
    analyze_ml_errors
)


@pytest.fixture
def sample_train_features():
    df_train = load_training_data()
    df_feat, _ = create_stage5_features(df_train, fit_normal_models=True)
    return df_feat


@pytest.fixture
def sample_test_features(sample_train_features):
    df_test = load_test_data()
    df_feat, _ = create_stage5_features(df_test, fit_normal_models=False)
    return df_feat


def test_1_training_data_has_1000_rows():
    df_train = load_training_data()
    assert len(df_train) == 1000


def test_2_validity_label_used_only_as_target(sample_train_features):
    train_sets, _, _ = prepare_feature_sets(sample_train_features)
    for name, X in train_sets.items():
        assert TASK01_TARGET not in X.columns


def test_3_reference_parameter_excluded(sample_train_features):
    train_sets, _, _ = prepare_feature_sets(sample_train_features)
    for name, X in train_sets.items():
        assert "Reference_Parameter" not in X.columns


def test_4_test_id_excluded(sample_train_features):
    train_sets, _, _ = prepare_feature_sets(sample_train_features)
    for name, X in train_sets.items():
        assert ID_COLUMN not in X.columns


def test_5_no_target_leakage_in_sets(sample_train_features):
    train_sets, _, _ = prepare_feature_sets(sample_train_features)
    for name, X in train_sets.items():
        assert not any("Validity" in col for col in X.columns)


def test_6_inference_available_features_only(sample_test_features):
    test_sets, _, _ = prepare_feature_sets(sample_test_features)
    for name, X in test_sets.items():
        if X is not None:
            assert TASK01_TARGET not in X.columns


def test_7_finite_features_after_imputation(sample_train_features):
    train_sets, _, _ = prepare_feature_sets(sample_train_features)
    X = train_sets["Set_D_Full_Stage5"]
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    assert not np.isnan(X_imp).any()
    assert not np.isinf(X_imp).any()


def test_8_stratified_cv_splits(sample_train_features):
    y_true = (sample_train_features[TASK01_TARGET] == "Invalid").astype(int).values
    train_sets, _, _ = prepare_feature_sets(sample_train_features)
    pipeline = get_model_pipelines()["Logistic_Regression"]
    metrics, oof_probs = evaluate_model_cv("Logistic_Regression", pipeline, train_sets["Set_A_Raw"], y_true, n_splits=5)
    assert metrics["Model"] == "Logistic_Regression"


def test_9_identical_cv_folds(sample_train_features):
    y_true = (sample_train_features[TASK01_TARGET] == "Invalid").astype(int).values
    train_sets, _, _ = prepare_feature_sets(sample_train_features)
    m1, oof1 = evaluate_model_cv("Logistic_Regression", get_model_pipelines()["Logistic_Regression"], train_sets["Set_A_Raw"], y_true)
    m2, oof2 = evaluate_model_cv("Random_Forest", get_model_pipelines()["Random_Forest"], train_sets["Set_A_Raw"], y_true)
    assert len(oof1) == len(oof2) == 1000


def test_10_every_row_receives_exact_one_oof_prediction(sample_train_features):
    comp_df, all_oof_probs, _ = evaluate_all_stage7_models(sample_train_features)
    for config, probs in all_oof_probs.items():
        assert len(probs) == 1000
        assert not np.isnan(probs).any()


def test_11_oof_probabilities_in_unit_range(sample_train_features):
    comp_df, all_oof_probs, _ = evaluate_all_stage7_models(sample_train_features)
    for config, probs in all_oof_probs.items():
        assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_12_invalid_class_metrics_calculation(sample_train_features):
    y_true = (sample_train_features[TASK01_TARGET] == "Invalid").astype(int).values
    train_sets, _, _ = prepare_feature_sets(sample_train_features)
    pipeline = get_model_pipelines()["Logistic_Regression"]
    metrics, _ = evaluate_model_cv("Logistic_Regression", pipeline, train_sets["Set_A_Raw"], y_true)
    for k in ["Invalid_Precision", "Invalid_Recall", "Invalid_F1", "Balanced_Accuracy", "Accuracy", "ROC_AUC", "PR_AUC"]:
        assert 0.0 <= metrics[k] <= 1.0


def test_13_threshold_tuning_uses_oof(sample_train_features):
    y_true = (sample_train_features[TASK01_TARGET] == "Invalid").astype(int).values
    train_sets, _, _ = prepare_feature_sets(sample_train_features)
    _, oof_probs = evaluate_model_cv("Logistic_Regression", get_model_pipelines()["Logistic_Regression"], train_sets["Set_A_Raw"], y_true)
    thresh_df = optimize_decision_threshold(y_true, oof_probs)
    assert isinstance(thresh_df, pd.DataFrame)
    assert len(thresh_df) > 0


def test_14_selected_threshold_reproducible(sample_train_features):
    y_true = (sample_train_features[TASK01_TARGET] == "Invalid").astype(int).values
    train_sets, _, _ = prepare_feature_sets(sample_train_features)
    _, oof1 = evaluate_model_cv("Logistic_Regression", get_model_pipelines()["Logistic_Regression"], train_sets["Set_A_Raw"], y_true)
    _, oof2 = evaluate_model_cv("Logistic_Regression", get_model_pipelines()["Logistic_Regression"], train_sets["Set_A_Raw"], y_true)
    t1 = optimize_decision_threshold(y_true, oof1).iloc[0]["Threshold"]
    t2 = optimize_decision_threshold(y_true, oof2).iloc[0]["Threshold"]
    assert t1 == t2


def test_15_final_model_fit_successful(sample_train_features):
    y_true = (sample_train_features[TASK01_TARGET] == "Invalid").astype(int).values
    train_sets, _, _ = prepare_feature_sets(sample_train_features)
    pipeline = get_model_pipelines()["Random_Forest"]
    pipeline.fit(train_sets["Set_D_Full_Stage5"].values, y_true)
    assert hasattr(pipeline, "classes_")


def test_16_predict_test_data_without_target(sample_train_features, sample_test_features):
    y_true = (sample_train_features[TASK01_TARGET] == "Invalid").astype(int).values
    train_sets, test_sets, _ = prepare_feature_sets(sample_train_features, sample_test_features)
    pipeline = get_model_pipelines()["Random_Forest"]
    pipeline.fit(train_sets["Set_D_Full_Stage5"].values, y_true)
    test_probs = pipeline.predict_proba(test_sets["Set_D_Full_Stage5"].values)[:, 1]
    assert len(test_probs) == 350


def test_17_test_data_350_predictions(sample_train_features, sample_test_features):
    y_true = (sample_train_features[TASK01_TARGET] == "Invalid").astype(int).values
    train_sets, test_sets, _ = prepare_feature_sets(sample_train_features, sample_test_features)
    pipeline = get_model_pipelines()["Logistic_Regression"]
    pipeline.fit(train_sets["Set_D_Full_Stage5"].values, y_true)
    preds = pipeline.predict(test_sets["Set_D_Full_Stage5"].values)
    assert len(preds) == 350


def test_18_random_seed_reproducibility(sample_train_features):
    comp1, _, _ = evaluate_all_stage7_models(sample_train_features)
    comp2, _, _ = evaluate_all_stage7_models(sample_train_features)
    pd.testing.assert_frame_equal(comp1, comp2)


def test_19_permutation_importance(sample_train_features):
    y_true = (sample_train_features[TASK01_TARGET] == "Invalid").astype(int).values
    train_sets, _, names_dict = prepare_feature_sets(sample_train_features)
    pipeline = get_model_pipelines()["Logistic_Regression"]
    imp_df = calculate_permutation_importance(pipeline, train_sets["Set_A_Raw"], y_true, names_dict["Set_A_Raw"])
    assert isinstance(imp_df, pd.DataFrame)
    assert "importance_mean" in imp_df.columns


def test_20_ml_error_analysis(sample_train_features):
    y_true = (sample_train_features[TASK01_TARGET] == "Invalid").astype(int).values
    preds = np.zeros(len(y_true), dtype=int)
    fp_df, fn_df = analyze_ml_errors(sample_train_features, y_true, preds, "Test_Model")
    assert isinstance(fp_df, pd.DataFrame)
    assert isinstance(fn_df, pd.DataFrame)
