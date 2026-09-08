"""
CPRI Hackathon — Person 1 Stage 5 Unit Tests
=============================================
Tests for Person 1 Stage 5 Feature Engineering Module.
"""

import pytest
import numpy as np
import pandas as pd

from src.dataset_loader import load_training_data, load_test_data, TASK01_TARGET, ID_COLUMN
from src.stage5_features import (
    create_sensor_difference_features,
    create_sensor_aggregate_features,
    create_sensor_disagreement_features,
    create_interaction_features,
    create_stage5_features,
    prepare_stage5_feature_matrix,
    validate_stage5_schema,
    compare_s4_information,
    build_feature_dictionary
)


@pytest.fixture
def sample_train_data():
    return load_training_data()


@pytest.fixture
def sample_test_data():
    return load_test_data()


def test_row_count_preservation(sample_train_data, sample_test_data):
    df_train_feat, _ = create_stage5_features(sample_train_data, fit_normal_models=True)
    df_test_feat, _ = create_stage5_features(sample_test_data, fit_normal_models=False)
    assert len(df_train_feat) == 1000
    assert len(df_test_feat) == 350


def test_feature_generation_on_both_datasets(sample_train_data, sample_test_data):
    df_train_feat, models = create_stage5_features(sample_train_data, fit_normal_models=True)
    df_test_feat, _ = create_stage5_features(sample_test_data, fit_normal_models=False, normal_models=models)
    assert "S1_minus_S2" in df_train_feat.columns
    assert "S1_minus_S2" in df_test_feat.columns
    assert "sensor_mean_S1_S2_S3" in df_train_feat.columns
    assert "sensor_mean_S1_S2_S3" in df_test_feat.columns


def test_no_target_in_feature_matrix(sample_train_data):
    df_train_feat, _ = create_stage5_features(sample_train_data, fit_normal_models=True)
    X = prepare_stage5_feature_matrix(df_train_feat)
    assert TASK01_TARGET not in X.columns
    assert "Reference_Parameter" not in X.columns
    assert ID_COLUMN not in X.columns


def test_no_duplicate_feature_names(sample_train_data):
    df_train_feat, _ = create_stage5_features(sample_train_data, fit_normal_models=True)
    assert len(df_train_feat.columns) == len(set(df_train_feat.columns))


def test_sensor_difference_features(sample_train_data):
    df_diff = create_sensor_difference_features(sample_train_data)
    assert "S1_minus_S2" in df_diff.columns
    assert "S1_minus_S3" in df_diff.columns
    assert "S2_minus_S3" in df_diff.columns


def test_sensor_aggregate_features(sample_train_data):
    df_agg = create_sensor_aggregate_features(sample_train_data)
    assert "sensor_mean_S1_S2_S3" in df_agg.columns
    assert "sensor_median_S1_S2_S3" in df_agg.columns
    assert "sensor_std_S1_S2_S3" in df_agg.columns


def test_sensor_disagreement_features(sample_train_data):
    df_dis = create_sensor_disagreement_features(sample_train_data)
    assert "max_minus_min_sensors" in df_dis.columns
    assert "pairwise_abs_diff_mean" in df_dis.columns
    assert "pairwise_abs_diff_max" in df_dis.columns


def test_interaction_features(sample_train_data):
    df_int = create_interaction_features(sample_train_data)
    assert "Apparent_Power_kVA" in df_int.columns
    assert "Thermal_Loading_Index" in df_int.columns
    assert "Apparent_Impedance_Proxy" in df_int.columns


def test_reproducibility(sample_train_data):
    df1, _ = create_stage5_features(sample_train_data, fit_normal_models=True)
    df2, _ = create_stage5_features(sample_train_data, fit_normal_models=True)
    pd.testing.assert_frame_equal(df1, df2)


def test_quality_flags_inference_safe(sample_test_data):
    df_test_feat, _ = create_stage5_features(sample_test_data, fit_normal_models=False)
    assert "missing_any" in df_test_feat.columns
    assert "non_finite_value" in df_test_feat.columns


def test_residuals_without_target_at_inference(sample_test_data, sample_train_data):
    df_train_feat, models = create_stage5_features(sample_train_data, fit_normal_models=True)
    df_test_feat, _ = create_stage5_features(sample_test_data, fit_normal_models=False, normal_models=models)
    assert "p1_max_abs_residual" in df_test_feat.columns
    assert "p1_consistency_index" in df_test_feat.columns


def test_compare_s4_information(sample_train_data):
    s4_df = compare_s4_information(sample_train_data)
    assert isinstance(s4_df, pd.DataFrame)
    assert not s4_df.empty


def test_build_feature_dictionary(sample_train_data):
    df_feat, _ = create_stage5_features(sample_train_data, fit_normal_models=True)
    fdict = build_feature_dictionary(df_feat)
    assert isinstance(fdict, pd.DataFrame)
    assert "feature_name" in fdict.columns
    assert "recommended_status" in fdict.columns


def test_schema_validation(sample_train_data):
    df_feat, _ = create_stage5_features(sample_train_data, fit_normal_models=True)
    features = list(df_feat.columns[:5])
    assert validate_stage5_schema(df_feat, features) is True


def test_no_raw_data_mutation(sample_train_data):
    orig_copy = sample_train_data.copy()
    _ = create_stage5_features(sample_train_data, fit_normal_models=True)
    pd.testing.assert_frame_equal(sample_train_data, orig_copy)
