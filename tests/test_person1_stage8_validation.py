"""
CPRI Hackathon -- Person 1 Stage 8 Validation Tests
====================================================

Verifies leakage-free OOF generation, threshold selection, fold stability,
and Stage 8 reporting artifacts.
"""

from pathlib import Path
import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.dataset_loader import load_training_data, TASK01_TARGET, ID_COLUMN, TASK02_TARGET
from src.stage8_validation import (
    generate_oof_predictions,
    evaluate_thresholds,
    select_threshold,
    evaluate_fold_stability,
    analyze_misclassifications,
    get_threshold_candidates,
    prepare_fold_feature_matrices,
)


def _first_stratified_fold(df_train):
    y_true = (df_train[TASK01_TARGET].astype(str).str.strip() == "Invalid").astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return next(skf.split(df_train, y_true))


@pytest.fixture(scope="module")
def stage8_artifacts():
    df_train = load_training_data()
    oof_df = generate_oof_predictions(df_train)
    threshold_df = evaluate_thresholds(oof_df)
    selection = select_threshold(threshold_df)
    fold_metrics_df, fold_summary_df = evaluate_fold_stability(oof_df, selection["selected_threshold"])
    fp_df, fn_df, summary_df = analyze_misclassifications(oof_df, selection["selected_threshold"])
    return {
        "train_df": df_train,
        "oof_df": oof_df,
        "threshold_df": threshold_df,
        "selection": selection,
        "fold_metrics_df": fold_metrics_df,
        "fold_summary_df": fold_summary_df,
        "fp_df": fp_df,
        "fn_df": fn_df,
        "summary_df": summary_df,
    }


def test_1_oof_predictions_contain_1000_records(stage8_artifacts):
    oof_df = stage8_artifacts["oof_df"]
    assert len(oof_df) == 1000


def test_2_every_training_record_receives_exactly_one_oof_prediction(stage8_artifacts):
    oof_df = stage8_artifacts["oof_df"]
    assert oof_df[ID_COLUMN].nunique() == 1000
    assert oof_df["source_index"].is_unique
    assert (oof_df[ID_COLUMN].value_counts() == 1).all()


def test_3_test_id_excluded_from_model_features(stage8_artifacts):
    df_train = stage8_artifacts["train_df"]
    train_idx, val_idx = _first_stratified_fold(df_train)
    fold = prepare_fold_feature_matrices(
        df_train.iloc[train_idx],
        df_train.iloc[val_idx],
        feature_set_name="Set_D_Full_Stage5",
    )
    x_train = fold["X_train"]
    assert ID_COLUMN not in x_train.columns


def test_4_validity_label_excluded_from_x(stage8_artifacts):
    df_train = stage8_artifacts["train_df"]
    train_idx, val_idx = _first_stratified_fold(df_train)
    fold = prepare_fold_feature_matrices(
        df_train.iloc[train_idx],
        df_train.iloc[val_idx],
        feature_set_name="Set_D_Full_Stage5",
    )
    x_train = fold["X_train"]
    assert TASK01_TARGET not in x_train.columns


def test_5_reference_parameter_excluded_from_x(stage8_artifacts):
    df_train = stage8_artifacts["train_df"]
    train_idx, val_idx = _first_stratified_fold(df_train)
    fold = prepare_fold_feature_matrices(
        df_train.iloc[train_idx],
        df_train.iloc[val_idx],
        feature_set_name="Set_D_Full_Stage5",
    )
    x_train = fold["X_train"]
    assert TASK02_TARGET not in x_train.columns


def test_6_oof_probabilities_are_finite(stage8_artifacts):
    probs = stage8_artifacts["oof_df"]["oof_probability_invalid"].to_numpy()
    assert np.isfinite(probs).all()


def test_7_oof_probabilities_within_unit_interval(stage8_artifacts):
    probs = stage8_artifacts["oof_df"]["oof_probability_invalid"].to_numpy()
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_8_threshold_candidates_are_exactly_requested_grid():
    expected = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]
    assert get_threshold_candidates() == expected


def test_9_selected_threshold_within_candidate_range(stage8_artifacts):
    candidates = get_threshold_candidates()
    selected = stage8_artifacts["selection"]["selected_threshold"]
    assert selected in candidates


def test_10_confusion_matrix_accounts_for_all_1000_records(stage8_artifacts):
    threshold = stage8_artifacts["selection"]["selected_threshold"]
    probs = stage8_artifacts["oof_df"]["oof_probability_invalid"].to_numpy()
    y_true = (stage8_artifacts["oof_df"][TASK01_TARGET].astype(str).str.strip() == "Invalid").astype(int).to_numpy()
    y_pred = (probs >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    assert tp + tn + fp + fn == 1000


def test_11_fp_plus_tn_equals_total_valid_records(stage8_artifacts):
    oof_df = stage8_artifacts["oof_df"]
    threshold = stage8_artifacts["selection"]["selected_threshold"]
    probs = oof_df["oof_probability_invalid"].to_numpy()
    y_true = (oof_df[TASK01_TARGET].astype(str).str.strip() == "Invalid").astype(int).to_numpy()
    y_pred = (probs >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    total_valid = int((oof_df[TASK01_TARGET].astype(str).str.strip() == "Valid").sum())
    assert fp + tn == total_valid


def test_12_tp_plus_fn_equals_total_invalid_records(stage8_artifacts):
    oof_df = stage8_artifacts["oof_df"]
    threshold = stage8_artifacts["selection"]["selected_threshold"]
    probs = oof_df["oof_probability_invalid"].to_numpy()
    y_true = (oof_df[TASK01_TARGET].astype(str).str.strip() == "Invalid").astype(int).to_numpy()
    y_pred = (probs >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total_invalid = int((oof_df[TASK01_TARGET].astype(str).str.strip() == "Invalid").sum())
    assert tp + fn == total_invalid


def test_13_fold_metrics_contain_all_5_folds(stage8_artifacts):
    fold_metrics_df = stage8_artifacts["fold_metrics_df"]
    assert set(fold_metrics_df["Fold"]) == {1, 2, 3, 4, 5}
    assert len(fold_metrics_df) == 5


def test_14_fold_metrics_are_reproducible_with_random_state_42(stage8_artifacts):
    df_train = stage8_artifacts["train_df"]
    oof_1 = generate_oof_predictions(df_train, random_state=42)
    oof_2 = generate_oof_predictions(df_train, random_state=42)
    thr_1 = evaluate_thresholds(oof_1)
    thr_2 = evaluate_thresholds(oof_2)
    sel_1 = select_threshold(thr_1)["selected_threshold"]
    sel_2 = select_threshold(thr_2)["selected_threshold"]
    fold_1, _ = evaluate_fold_stability(oof_1, sel_1)
    fold_2, _ = evaluate_fold_stability(oof_2, sel_2)
    pd.testing.assert_frame_equal(fold_1, fold_2)


def test_15_test_data_is_never_used_for_threshold_tuning():
    import src.stage8_validation as stage8_validation

    source = inspect.getsource(stage8_validation)
    assert "load_test_data" not in source
    assert "Test_Data" not in source


def test_16_existing_stage1_to_stage7_tests_remain_present():
    tests_dir = Path(__file__).resolve().parent
    expected = [
        tests_dir / "test_person1_stage1_dataset_setup.py",
        tests_dir / "test_person1_stage2_data_quality.py",
        tests_dir / "test_person1_stage3_historical_analysis.py",
        tests_dir / "test_person1_stage4_behaviour.py",
        tests_dir / "test_person1_stage5_feature_engineering.py",
        tests_dir / "test_person1_stage6_baselines.py",
        tests_dir / "test_person1_stage7_ml_models.py",
        tests_dir / "test_person1_stage7_leakage_audit.py",
    ]
    assert all(path.exists() for path in expected)
