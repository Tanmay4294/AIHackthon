"""
Automated Validation Suite — P2-Stage 1 Data Quality Audit
Tests quality_features.py on official CPRI dataset and dummy data.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add src to path
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
    assert len(df_train_flagged) == trn_len_orig
    assert len(df_test_flagged) == tst_len_orig
    
    # Check 2: Original columns preserved without alteration
    for col in df_train.columns:
        pd.testing.assert_series_equal(df_train_flagged[col], df_train[col])
        
    for col in df_test.columns:
        pd.testing.assert_series_equal(df_test_flagged[col], df_test[col])


def test_determinism_and_reproducibility(cpri_data):
    df_train, df_test = cpri_data
    
    run1 = create_quality_flags(df_train)
    run2 = create_quality_flags(df_train)
    
    # Identical outputs on repeated executions
    pd.testing.assert_frame_equal(run1, run2)


def test_test_data_independence_from_validity_label(cpri_data):
    _, df_test = cpri_data
    assert "Validity_Label" not in df_test.columns
    
    # Must run without error on Test_Data
    df_test_flagged = create_quality_flags(df_test)
    assert "data_quality_issue" in df_test_flagged.columns
    assert len(df_test_flagged) == 350


def test_cpri_specific_data_quality_counts(cpri_data):
    df_train, df_test = cpri_data
    
    df_trn_f = create_quality_flags(df_train)
    df_tst_f = create_quality_flags(df_test)
    
    # Verify exact empirical counts obtained from CPRI dataset
    assert df_trn_f['missing_any'].sum() == 44
    assert df_tst_f['missing_any'].sum() == 17
    
    assert df_trn_f['duplicate_measurement_pair'].sum() == 24
    assert df_tst_f['duplicate_measurement_pair'].sum() == 8
    
    assert df_trn_f['invalid_sensor_negative'].sum() == 1
    assert df_tst_f['invalid_sensor_negative'].sum() == 1
    
    assert df_trn_f['invalid_sensor_zero'].sum() == 3
    assert df_tst_f['invalid_sensor_zero'].sum() == 1

    assert df_trn_f['data_quality_issue'].sum() == 72
    assert df_tst_f['data_quality_issue'].sum() == 25
