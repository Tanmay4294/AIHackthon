"""
Automated Validation Suite — P2-Stage 3 Unsupervised Anomaly Detection
Verifies dataset integrity, target leakage exclusion, score finiteness and direction,
5-fold/OOF agreement group accounting, test data independence, and reproducibility.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stage3_features import load_stage3_datasets, prepare_stage3_feature_sets
from stage3_anomaly import IsolationForestAnomalyDetector, LOFAnomalyDetector, evaluate_unsupervised_anomaly

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx')


@pytest.fixture
def stage3_data():
    if not os.path.exists(DATASET_PATH):
        pytest.skip(f"CPRI dataset file not found at {DATASET_PATH}")
    df_tr, df_te = load_stage3_datasets(DATASET_PATH)
    prep = prepare_stage3_feature_sets(df_tr, df_te)
    return df_tr, df_te, prep


def test_dataset_record_counts(stage3_data):
    df_tr, df_te, _ = stage3_data
    # 1. Training dataset has 1000 records
    assert len(df_tr) == 1000
    # 2. Test dataset has 350 records
    assert len(df_te) == 350


def test_target_and_id_exclusion_guarantee(stage3_data):
    _, _, prep = stage3_data
    forbidden = ["Reference_Parameter", "Validity_Label", "Test_ID"]
    
    # 3, 4, 5: Excluded from feature sets A, B, C, D
    for set_key, X_mat in prep["X_train"].items():
        for col in forbidden:
            assert col not in X_mat.columns, f"Target/ID Leakage: {col} in {set_key}!"


def test_anomaly_score_finiteness_and_flags(stage3_data):
    _, _, prep = stage3_data
    X_tr = prep["X_train"]["Set_A"]
    X_te = prep["X_test"]["Set_A"]
    
    detector = IsolationForestAnomalyDetector(n_estimators=100, random_state=42)
    detector.fit(X_tr)
    
    tr_scores = detector.score_samples(X_tr)
    tr_flags = detector.predict(X_tr)
    te_scores = detector.score_samples(X_te)
    te_flags = detector.predict(X_te)
    
    # 6. All anomaly scores are finite
    assert np.all(np.isfinite(tr_scores))
    assert np.all(np.isfinite(te_scores))
    
    # 8. Anomaly flags contain only valid binary values (0 or 1)
    assert set(np.unique(tr_flags)).issubset({0, 1})
    assert set(np.unique(te_flags)).issubset({0, 1})
    
    # 9, 10. All 1000 training records and 350 test records receive scores
    assert len(tr_scores) == 1000
    assert len(te_scores) == 350


def test_anomaly_score_direction_convention(stage3_data):
    _, _, prep = stage3_data
    X_tr = prep["X_train"]["Set_A"]
    
    detector = IsolationForestAnomalyDetector(n_estimators=100, random_state=42)
    detector.fit(X_tr)
    scores = detector.score_samples(X_tr)
    
    # 7. Higher anomaly score means more anomalous
    # Create extreme synthetic outlier
    X_outlier = X_tr.iloc[[0]].copy()
    for col in X_outlier.columns:
        X_outlier[col] = X_outlier[col] * 1000.0  # extreme shift
        
    outlier_score = detector.score_samples(X_outlier)[0]
    median_normal_score = np.median(scores)
    
    assert outlier_score > median_normal_score, "Higher score must represent greater anomaly!"


def test_test_data_independence_from_validity_label(stage3_data):
    _, df_te, prep = stage3_data
    # 11. Test data processing operates independently without requiring Validity_Label
    assert "Validity_Label" not in df_te.columns
    
    X_tr = prep["X_train"]["Set_B"]
    X_te = prep["X_test"]["Set_B"]
    
    lof = LOFAnomalyDetector(n_neighbors=15)
    lof.fit(X_tr)
    
    scores = lof.score_samples(X_te)
    flags = lof.predict(X_te)
    
    assert len(scores) == 350
    assert len(flags) == 350


def test_agreement_group_accounting(stage3_data):
    df_tr, _, prep = stage3_data
    X_tr = prep["X_train"]["Set_D"]
    y_tr = prep["y_train"]
    
    detector = IsolationForestAnomalyDetector(n_estimators=100, random_state=42)
    detector.fit(X_tr)
    anom_flags = detector.predict(X_tr)
    
    # Dummy supervised binary predictions
    sup_flags = (y_tr == 1).astype(int).to_numpy()
    
    g1 = np.sum((sup_flags == 1) & (anom_flags == 1))
    g2 = np.sum((sup_flags == 1) & (anom_flags == 0))
    g3 = np.sum((sup_flags == 0) & (anom_flags == 1))
    g4 = np.sum((sup_flags == 0) & (anom_flags == 0))
    
    # 12. Agreement groups account for 100% of historical records
    assert g1 + g2 + g3 + g4 == 1000


def test_stage3_reproducibility(stage3_data):
    _, _, prep = stage3_data
    X_tr = prep["X_train"]["Set_B"]
    
    run1 = IsolationForestAnomalyDetector(n_estimators=100, random_state=42).fit(X_tr).score_samples(X_tr)
    run2 = IsolationForestAnomalyDetector(n_estimators=100, random_state=42).fit(X_tr).score_samples(X_tr)
    
    # 13. Reproducibility with random_state=42
    np.testing.assert_array_equal(run1, run2)
