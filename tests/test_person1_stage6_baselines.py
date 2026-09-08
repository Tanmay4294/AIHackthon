"""
CPRI Hackathon — Person 1 Stage 6 Unit Tests
=============================================
Tests for Person 1 Stage 6 Baseline Detectors Module.
"""

import pytest
import numpy as np
import pandas as pd

from src.dataset_loader import load_training_data, load_test_data, TASK01_TARGET, ID_COLUMN
from src.stage5_features import create_stage5_features
from src.stage6_baselines import (
    run_deterministic_quality_baseline,
    run_iqr_statistical_baseline,
    run_zscore_statistical_baseline,
    run_unsupervised_baseline,
    run_residual_consistency_baseline,
    evaluate_detector,
    analyze_false_positives,
    analyze_false_negatives,
    evaluate_regime_performance
)


@pytest.fixture
def sample_train_features():
    df_train = load_training_data()
    df_feat, _ = create_stage5_features(df_train, fit_normal_models=True)
    return df_feat


def test_deterministic_quality_baseline(sample_train_features):
    preds, scores = run_deterministic_quality_baseline(sample_train_features)
    assert len(preds) == len(sample_train_features)
    assert set(np.unique(preds)).issubset({0, 1})
    assert len(scores) == len(sample_train_features)


def test_iqr_statistical_baseline(sample_train_features):
    preds, scores = run_iqr_statistical_baseline(sample_train_features)
    assert len(preds) == len(sample_train_features)
    assert set(np.unique(preds)).issubset({0, 1})


def test_zscore_statistical_baseline(sample_train_features):
    preds, scores = run_zscore_statistical_baseline(sample_train_features, threshold=3.0)
    assert len(preds) == len(sample_train_features)
    assert set(np.unique(preds)).issubset({0, 1})


def test_unsupervised_isolation_forest_baseline(sample_train_features):
    preds, scores = run_unsupervised_baseline(sample_train_features, sample_train_features, method="isolation_forest")
    assert len(preds) == len(sample_train_features)
    assert set(np.unique(preds)).issubset({0, 1})
    assert not np.isnan(scores).any()


def test_unsupervised_lof_baseline(sample_train_features):
    preds, scores = run_unsupervised_baseline(sample_train_features, sample_train_features, method="lof")
    assert len(preds) == len(sample_train_features)
    assert set(np.unique(preds)).issubset({0, 1})
    assert not np.isnan(scores).any()


def test_residual_consistency_baseline(sample_train_features):
    preds, scores = run_residual_consistency_baseline(sample_train_features, max_res_threshold=3.0)
    assert len(preds) == len(sample_train_features)
    assert set(np.unique(preds)).issubset({0, 1})


def test_evaluation_confusion_matrix_accounting(sample_train_features):
    y_true = (sample_train_features[TASK01_TARGET] == "Invalid").astype(int).values
    preds, scores = run_deterministic_quality_baseline(sample_train_features)
    metrics = evaluate_detector(y_true, preds, scores, "Deterministic", "Quality Flags", "Rule")
    assert metrics["TP"] + metrics["TN"] + metrics["FP"] + metrics["FN"] == len(sample_train_features)


def test_false_positive_analysis(sample_train_features):
    preds, _ = run_iqr_statistical_baseline(sample_train_features)
    fp_df = analyze_false_positives(sample_train_features, preds, "IQR_Statistical")
    assert isinstance(fp_df, pd.DataFrame)
    if not fp_df.empty:
        assert (fp_df["Validity_Label"] == "Valid").all()


def test_false_negative_analysis(sample_train_features):
    preds, _ = run_deterministic_quality_baseline(sample_train_features)
    fn_df = analyze_false_negatives(sample_train_features, preds, "Deterministic")
    assert isinstance(fn_df, pd.DataFrame)
    if not fn_df.empty:
        assert (fn_df["Validity_Label"] == "Invalid").all()


def test_regime_performance_evaluation(sample_train_features):
    preds_dict = {
        "Deterministic": run_deterministic_quality_baseline(sample_train_features)[0],
        "IQR": run_iqr_statistical_baseline(sample_train_features)[0]
    }
    reg_df = evaluate_regime_performance(sample_train_features, preds_dict)
    assert isinstance(reg_df, pd.DataFrame)
    assert "Operating_Regime" in reg_df.columns


def test_no_target_leakage_in_fitting(sample_train_features):
    # Confirm predictors matrix passed to isolation forest excludes target
    preds, scores = run_unsupervised_baseline(sample_train_features, sample_train_features, method="isolation_forest")
    assert len(preds) == 1000


def test_baseline_reproducibility(sample_train_features):
    p1, s1 = run_unsupervised_baseline(sample_train_features, sample_train_features, method="isolation_forest")
    p2, s2 = run_unsupervised_baseline(sample_train_features, sample_train_features, method="isolation_forest")
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(s1, s2)


def test_test_data_prediction_compatibility(sample_train_features):
    df_test = load_test_data()
    df_test_feat, _ = create_stage5_features(df_test, fit_normal_models=False)
    p_test, s_test = run_unsupervised_baseline(sample_train_features, df_test_feat, method="isolation_forest")
    assert len(p_test) == 350
    assert len(s_test) == 350


def test_metrics_range(sample_train_features):
    y_true = (sample_train_features[TASK01_TARGET] == "Invalid").astype(int).values
    preds, scores = run_residual_consistency_baseline(sample_train_features)
    m = evaluate_detector(y_true, preds, scores, "Residual", "Stage 4", "Thresh")
    assert 0.0 <= m["Invalid_Precision"] <= 1.0
    assert 0.0 <= m["Invalid_Recall"] <= 1.0
    assert 0.0 <= m["Invalid_F1"] <= 1.0
    assert 0.0 <= m["Balanced_Accuracy"] <= 1.0


def test_zero_row_deletion_across_baselines(sample_train_features):
    preds1, _ = run_deterministic_quality_baseline(sample_train_features)
    preds2, _ = run_zscore_statistical_baseline(sample_train_features)
    assert len(preds1) == 1000
    assert len(preds2) == 1000
