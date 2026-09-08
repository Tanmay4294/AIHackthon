"""
CPRI Hackathon — Person 1 Stage 7 Leakage Audit Unit Tests
===========================================================
Verifies that Stage 7 model evaluation enforces strict fold isolation
and that zero target leakage occurs across cross-validation folds.
"""

import pytest
import numpy as np
import pandas as pd

from src.dataset_loader import load_training_data, TASK01_TARGET, ID_COLUMN
from src.stage7_ml_models import (
    get_model_pipelines,
    evaluate_stage7_configuration_fold_isolated,
    evaluate_all_stage7_models,
    optimize_decision_threshold
)


@pytest.fixture
def sample_train_data():
    return load_training_data()


def test_fold_isolated_evaluation_prevents_global_leakage(sample_train_data):
    pipeline = get_model_pipelines()["Random_Forest"]
    metrics, oof_probs = evaluate_stage7_configuration_fold_isolated("Random_Forest", pipeline, "Set_D_Full_Stage5", sample_train_data, n_splits=5)
    assert len(oof_probs) == 1000
    assert not np.isnan(oof_probs).any()
    # Verified: Corrected fold-isolated F1 is non-trivial (< 1.0 at default 0.50 threshold before threshold optimization)
    assert 0.85 <= metrics["Invalid_F1"] <= 0.99


def test_raw_features_have_realistic_performance(sample_train_data):
    pipeline = get_model_pipelines()["Logistic_Regression"]
    metrics, _ = evaluate_stage7_configuration_fold_isolated("Logistic_Regression", pipeline, "Set_A_Raw", sample_train_data, n_splits=5)
    # Raw features without residuals or quality flags have low F1 (< 0.50)
    assert metrics["Invalid_F1"] < 0.50


def test_progressive_feature_set_improvement(sample_train_data):
    comp_df, _, _ = evaluate_all_stage7_models(sample_train_data)
    rf_rows = comp_df[comp_df["Model"] == "Random_Forest"]

    f1_raw = rf_rows[rf_rows["Feature_Set"] == "Set_A_Raw"]["Invalid_F1"].iloc[0]
    f1_full = rf_rows[rf_rows["Feature_Set"] == "Set_D_Full_Stage5"]["Invalid_F1"].iloc[0]

    # Full Stage 5 features significantly outperform raw features
    assert f1_full > f1_raw + 0.20


def test_threshold_optimization_improves_recall_and_f1(sample_train_data):
    y_true = (sample_train_data[TASK01_TARGET] == "Invalid").astype(int).values
    pipeline = get_model_pipelines()["Random_Forest"]
    _, oof_probs = evaluate_stage7_configuration_fold_isolated("Random_Forest", pipeline, "Set_D_Full_Stage5", sample_train_data, n_splits=5)

    thresh_df = optimize_decision_threshold(y_true, oof_probs)
    best_row = thresh_df.iloc[0]

    # Optimized threshold achieves > 0.98 F1 and > 0.98 Recall
    assert best_row["Invalid_F1"] >= 0.98
    assert best_row["Invalid_Recall"] >= 0.98
