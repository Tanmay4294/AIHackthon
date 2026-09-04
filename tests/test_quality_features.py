"""
Automated Validation Suite — P2-Stage 1 Data Quality Audit (Corrected)
Tests quality_features.py on official CPRI dataset and verifies all corrected audit rules.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quality_features import (
    audit_missing_values,
    audit_non_finite_values,
    audit_duplicate_rows,
    audit_duplicate_test_ids,
    audit_duplicate_measurement_pairs,
    audit_numeric_columns,
    audit_physical_constraints,
    create_quality_flags,
    generate_quality_summary,
    FEATURE_COLUMNS
)

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx')


@pytest.fixture
def cpri_data():
    if not os.path.exists(DATASET_PATH):
        pytest.skip(f"CPRI dataset file not found at {DATASET_PATH}")
    excel = pd.ExcelFile(DATASET_PATH)
    df_train = pd.read_excel(excel, sheet_name='Training_Data')
    df_test = pd.read_excel(excel, sheet_name='Test_Data')
    return df_train, df_test


def test_row_counts_and_preservation(cpri_data):
    df_train, df_test = cpri_data
    
    trn_len_orig = len(df_train)
    tst_len_orig = len(df_test)
    
    df_train_flagged = create_quality_flags(df_train)
    df_test_flagged = create_quality_flags(df_test)
    
    # Check 1: Row counts unchanged
    assert len(df_train_flagged) == trn_len_orig == 1000
    assert len(df_test_flagged) == tst_len_orig == 350
    
    # Check 2: Original columns preserved without alteration
    for col in df_train.columns:
        pd.testing.assert_series_equal(df_train_flagged[col], df_train[col])
        
    for col in df_test.columns:
        pd.testing.assert_series_equal(df_test_flagged[col], df_test[col])


def test_determinism_and_reproducibility(cpri_data):
    df_train, _ = cpri_data
    
    run1 = create_quality_flags(df_train)
    run2 = create_quality_flags(df_train)
    
    # Identical outputs on repeated executions
    pd.testing.assert_frame_equal(run1, run2)


def test_non_finite_infinity_isolation():
    """Verify non_finite_value detects +Inf/-Inf ONLY and ignores NaN."""
    dummy_df = pd.DataFrame({
        "Applied_Voltage_kV": [10.0, np.nan, np.inf, -np.inf, 20.0],
        "Load_Current_A": [50.0, 60.0, 70.0, 80.0, 90.0]
    })
    non_fin = audit_non_finite_values(dummy_df, feature_cols=["Applied_Voltage_kV", "Load_Current_A"])
    
    # Row 0: False (10.0)
    # Row 1: False (NaN - missing_any, NOT non_finite)
    # Row 2: True (+Inf)
    # Row 3: True (-Inf)
    # Row 4: False (20.0)
    assert list(non_fin) == [False, False, True, True, False]


def test_test_data_independence_from_validity_label(cpri_data):
    _, df_test = cpri_data
    assert "Validity_Label" not in df_test.columns
    
    df_test_flagged = create_quality_flags(df_test)
    assert "data_quality_issue" in df_test_flagged.columns
    assert len(df_test_flagged) == 350


def test_cpri_specific_data_quality_counts(cpri_data):
    df_train, df_test = cpri_data
    
    df_trn_f = create_quality_flags(df_train)
    df_tst_f = create_quality_flags(df_test)
    
    # Missing values
    assert df_trn_f['missing_any'].sum() == 44
    assert df_tst_f['missing_any'].sum() == 17
    
    # Actual +Inf/-Inf ONLY (0 in CPRI dataset)
    assert df_trn_f['non_finite_value'].sum() == 0
    assert df_tst_f['non_finite_value'].sum() == 0

    # Duplicate full rows (0 in CPRI dataset)
    assert df_trn_f['duplicate_full_row'].sum() == 0
    assert df_tst_f['duplicate_full_row'].sum() == 0

    # Duplicate Test_IDs (0 in CPRI dataset)
    assert df_trn_f['duplicate_test_id'].sum() == 0
    assert df_tst_f['duplicate_test_id'].sum() == 0

    # Hard Physical Laws (0 violations in CPRI dataset)
    assert df_trn_f['invalid_voltage'].sum() == 0
    assert df_trn_f['invalid_current'].sum() == 0
    assert df_trn_f['invalid_duration'].sum() == 0
    assert df_trn_f['invalid_ambient_temp'].sum() == 0
    
    # Diagnostic Observations
    assert df_trn_f['duplicate_measurement_pair'].sum() == 24
    assert df_tst_f['duplicate_measurement_pair'].sum() == 8
    
    assert df_trn_f['sensor_s2_negative'].sum() == 1
    assert df_tst_f['sensor_s2_negative'].sum() == 1
    
    assert df_trn_f['sensor_zero_reading'].sum() == 3
    assert df_tst_f['sensor_zero_reading'].sum() == 1

    # Combined Definite Data Quality Failure (44 in Training, 17 in Test)
    assert df_trn_f['data_quality_issue'].sum() == 44
    assert df_tst_f['data_quality_issue'].sum() == 17
