"""
CPRI Hackathon — Person 1 Stage 4 Unit Tests
=============================================
Tests for Person 1 Stage 4 Behaviour & Physical Consistency Module.
"""

import pytest
import numpy as np
import pandas as pd

from src.dataset_loader import load_training_data, load_test_data, TASK01_TARGET, ID_COLUMN
from src.stage4_behaviour import (
    analyze_relationships,
    discover_operating_regimes,
    NormalBehaviourModels,
    analyze_residuals_by_class,
    analyze_sensor_reliability,
    analyze_legitimate_unusual_regimes_residuals,
    run_stage4_behaviour
)
from src.person1_adapter import get_person1_features, has_person1_features


@pytest.fixture
def sample_train_data():
    return load_training_data()


@pytest.fixture
def sample_test_data():
    return load_test_data()


def test_relationship_analysis_output_structure(sample_train_data):
    rel_df = analyze_relationships(sample_train_data)
    assert isinstance(rel_df, pd.DataFrame)
    assert not rel_df.empty
    expected_cols = ["Category", "Variable_1", "Variable_2", "Sample_Count", "Pearson_r", "Spearman_rho", "Relationship_Type"]
    for col in expected_cols:
        assert col in rel_df.columns


def test_discover_operating_regimes(sample_train_data):
    df_regime, regime_summary = discover_operating_regimes(sample_train_data)
    assert len(df_regime) == len(sample_train_data)
    assert "Operating_Regime_Code" in df_regime.columns
    assert isinstance(regime_summary, pd.DataFrame)
    assert not regime_summary.empty
    assert "Regime_Code" in regime_summary.columns


def test_normal_models_fit_valid_only(sample_train_data):
    models = NormalBehaviourModels()
    models.fit(sample_train_data)
    assert models.is_fitted
    assert len(models.models) > 0
    # Confirm trained record count in model performance matches Valid count
    n_valid = int((sample_train_data[TASK01_TARGET] == "Valid").sum())
    for perf in models.model_performance:
        assert 0 < perf["Valid_Train_Count"] <= n_valid


def test_normal_models_no_target_leakage(sample_train_data):
    models = NormalBehaviourModels()
    models.fit(sample_train_data)
    for perf in models.model_performance:
        predictors = perf["Predictors"].split(", ")
        assert TASK01_TARGET not in predictors
        assert "Reference_Parameter" not in predictors
        assert ID_COLUMN not in predictors


def test_normal_models_transform_output(sample_train_data):
    models = NormalBehaviourModels()
    models.fit(sample_train_data)
    df_res = models.transform(sample_train_data)
    assert len(df_res) == len(sample_train_data)
    assert "Sensor_S1_Expected" in df_res.columns
    assert "Sensor_S1_Residual" in df_res.columns
    assert "p1_max_abs_residual" in df_res.columns
    assert "p1_consistency_index" in df_res.columns


def test_transform_test_data_without_labels(sample_test_data, sample_train_data):
    models = NormalBehaviourModels()
    models.fit(sample_train_data)
    df_res_test = models.transform(sample_test_data)
    assert len(df_res_test) == len(sample_test_data)
    assert "p1_max_abs_residual" in df_res_test.columns
    assert TASK01_TARGET not in sample_test_data.columns


def test_residual_statistics_by_class(sample_train_data):
    models = NormalBehaviourModels()
    models.fit(sample_train_data)
    df_res = models.transform(sample_train_data)
    res_stats = analyze_residuals_by_class(df_res)
    assert isinstance(res_stats, pd.DataFrame)
    assert not res_stats.empty
    assert "Cohens_D_EffectSize" in res_stats.columns


def test_sensor_reliability_analysis(sample_train_data):
    models = NormalBehaviourModels()
    models.fit(sample_train_data)
    df_res = models.transform(sample_train_data)
    perf_df = pd.DataFrame(models.model_performance)
    reliability_df = analyze_sensor_reliability(df_res, perf_df)
    assert isinstance(reliability_df, pd.DataFrame)
    assert "Reliability_Rank" in reliability_df.columns


def test_legitimate_unusual_regimes_residuals(sample_train_data):
    models = NormalBehaviourModels()
    models.fit(sample_train_data)
    df_res = models.transform(sample_train_data)
    rep_df = analyze_legitimate_unusual_regimes_residuals(df_res)
    assert isinstance(rep_df, pd.DataFrame)


def test_run_stage4_behaviour_end_to_end(sample_train_data, sample_test_data):
    results = run_stage4_behaviour(sample_train_data, sample_test_data)
    expected_keys = [
        "relationship_analysis", "regime_analysis", "model_performance",
        "residual_statistics", "sensor_reliability", "candidate_features",
        "representative_records", "residuals_train", "residuals_test", "fitted_models"
    ]
    for k in expected_keys:
        assert k in results


def test_zero_row_deletion(sample_train_data, sample_test_data):
    models = NormalBehaviourModels()
    models.fit(sample_train_data)
    df_train_transformed = models.transform(sample_train_data)
    df_test_transformed = models.transform(sample_test_data)
    assert len(df_train_transformed) == 1000
    assert len(df_test_transformed) == 350


def test_test_id_exclusion(sample_train_data):
    models = NormalBehaviourModels()
    models.fit(sample_train_data)
    assert ID_COLUMN not in models.feature_names


def test_person1_adapter_integration(sample_train_data):
    models = NormalBehaviourModels()
    models.fit(sample_train_data)
    df_res = models.transform(sample_train_data)
    assert has_person1_features(df_res) is True
    p1_feats, is_genuine, names = get_person1_features(df_res)
    assert is_genuine is True
    assert "p1_max_abs_residual" in names


def test_no_inf_or_nan_in_residuals(sample_train_data):
    models = NormalBehaviourModels()
    models.fit(sample_train_data)
    df_res = models.transform(sample_train_data)
    res_stats = analyze_residuals_by_class(df_res)
    assert not res_stats["Cohens_D_EffectSize"].isna().any()
    assert not np.isinf(res_stats["Cohens_D_EffectSize"]).any()


def test_reproducibility(sample_train_data):
    m1 = NormalBehaviourModels().fit(sample_train_data)
    r1 = m1.transform(sample_train_data)

    m2 = NormalBehaviourModels().fit(sample_train_data)
    r2 = m2.transform(sample_train_data)

    pd.testing.assert_frame_equal(r1, r2)
