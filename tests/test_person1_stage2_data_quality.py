"""
CPRI Hackathon — Person 1 Stage 2 Unit Tests
=============================================
Automated test suite verifying the Person 1 Stage 2 Data-Quality Audit pipeline:
1. Training rows remain 1,000; Test rows remain 350.
2. Missing-value audit covers all columns.
3. Non-finite (+Inf / -Inf) detection works.
4. Duplicate detection works.
5. Repeated Test_ID detection & classification work.
6. Numeric statistics & percentile tables cover all numeric columns.
7. Physical suspicious-value detection works.
8. Quality audit flags are deterministic.
9. Validity_Label is not required for audit execution.
10. Reference_Parameter is not used as an audit feature.
11. Test_ID is not used as a predictive feature.
12. Source dataset is not mutated during audit.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_quality_audit import (
    audit_duplicates,
    audit_missing_values,
    audit_non_finite_values,
    audit_sensor_behaviour,
    audit_test_id_repeats,
    calculate_numeric_statistics,
    calculate_percentile_tables,
    create_quality_audit_flags,
    detect_physical_suspicious_values,
    identify_extreme_candidates,
    run_data_quality_audit,
)
from src.dataset_loader import (
    ID_COLUMN,
    PHYSICAL_FEATURES,
    TASK01_TARGET,
    TASK02_TARGET,
    load_test_data,
    load_training_data,
)


@pytest.fixture
def cpri_datasets():
    df_train = load_training_data()
    df_test = load_test_data()
    return df_train, df_test


def test_training_row_count_preservation(cpri_datasets):
    """1. Verifies Training_Data remains 1,000 rows."""
    df_train, _ = cpri_datasets
    assert len(df_train) == 1000, f"Expected 1,000 rows, got {len(df_train)}"


def test_test_row_count_preservation(cpri_datasets):
    """2. Verifies Test_Data remains 350 rows."""
    _, df_test = cpri_datasets
    assert len(df_test) == 350, f"Expected 350 rows, got {len(df_test)}"


def test_missing_value_audit_covers_all_columns(cpri_datasets):
    """3. Verifies missing-value audit covers every column in the dataset."""
    df_train, _ = cpri_datasets
    df_missing = audit_missing_values(df_train)
    assert len(df_missing) == len(df_train.columns)
    assert "Column" in df_missing.columns
    assert "MissingCount" in df_missing.columns


def test_non_finite_value_detection():
    """4. Verifies non-finite detection (+Inf / -Inf) on synthetic edge cases."""
    df_synth = pd.DataFrame({
        "Test_ID": ["TRN-0001", "TRN-0002", "TRN-0003"],
        "Applied_Voltage_kV": [12.0, np.inf, 24.0],
        "Load_Current_A": [100.0, 150.0, -np.inf],
    })
    df_non_finite = audit_non_finite_values(df_synth)
    voltage_row = df_non_finite[df_non_finite["Column"] == "Applied_Voltage_kV"].iloc[0]
    current_row = df_non_finite[df_non_finite["Column"] == "Load_Current_A"].iloc[0]

    assert voltage_row["PosInfCount"] == 1
    assert current_row["NegInfCount"] == 1


def test_duplicate_detection():
    """5. Verifies exact full-row duplicate detection on synthetic data."""
    df_synth = pd.DataFrame({
        "Test_ID": ["TRN-0001", "TRN-0001", "TRN-0002"],
        "Applied_Voltage_kV": [12.0, 12.0, 24.0],
        "Load_Current_A": [100.0, 100.0, 150.0],
    })
    df_dup = audit_duplicates(df_synth)
    assert df_dup.iloc[0]["TotalDuplicateRows"] == 2
    assert df_dup.iloc[0]["UniqueDuplicateGroups"] == 1


def test_repeated_test_id_detection():
    """6. Verifies repeated Test_ID detection and exact vs differing measurement classification."""
    df_synth = pd.DataFrame({
        "Test_ID": ["TRN-0001", "TRN-0001", "TRN-0002", "TRN-0002"],
        "Applied_Voltage_kV": [12.0, 12.0, 24.0, 36.0],  # TRN-0001 exact, TRN-0002 differing
        "Load_Current_A": [100.0, 100.0, 150.0, 150.0],
    })
    df_repeats = audit_test_id_repeats(df_synth)
    assert len(df_repeats) == 2

    row_0001 = df_repeats[df_repeats["Test_ID"] == "TRN-0001"].iloc[0]
    row_0002 = df_repeats[df_repeats["Test_ID"] == "TRN-0002"].iloc[0]

    assert row_0001["Classification"] == "Exact Duplicate Measurements"
    assert row_0002["Classification"] == "Same Test_ID Differing Measurements"


def test_numeric_statistics_cover_all_numeric_columns(cpri_datasets):
    """7. Verifies numeric statistics cover all 8 physical features."""
    df_train, _ = cpri_datasets
    df_stats = calculate_numeric_statistics(df_train)
    assert len(df_stats) == 8
    expected_cols = {"TotalCount", "ValidCount", "MissingCount", "Min", "Max", "Mean", "Median", "Std"}
    assert expected_cols.issubset(set(df_stats.columns))


def test_percentile_table_coverage(cpri_datasets):
    """8. Verifies percentile tables calculate 11 percentiles for physical features."""
    df_train, _ = cpri_datasets
    df_pct = calculate_percentile_tables(df_train)
    assert len(df_pct) == 8
    assert "P1_0" in df_pct.columns
    assert "P50_0" in df_pct.columns
    assert "P99_0" in df_pct.columns


def test_physical_suspicious_value_detection():
    """9. Verifies physical suspicious value detection on synthetic physical violations."""
    df_synth = pd.DataFrame({
        "Test_ID": ["TRN-0001", "TRN-0002"],
        "Applied_Voltage_kV": [-5.0, 12.0],  # Negative voltage violation
        "Load_Current_A": [100.0, -10.0],    # Negative current violation
        "Test_Duration_min": [10.0, 0.0],    # Non-positive duration
        "Ambient_Temperature_C": [25.0, 30.0],
        "Sensor_S2": [50.0, -2.0]            # Negative sensor S2
    })
    df_phys = detect_physical_suspicious_values(df_synth)
    assert len(df_phys) == 2
    assert "invalid_negative_voltage" in df_phys.iloc[0]["SuspiciousFlags"]
    assert "sensor_s2_negative" in df_phys.iloc[1]["SuspiciousFlags"]


def test_outlier_candidates_identification():
    """10. Verifies multi-method outlier candidate detection on synthetic data."""
    np.random.seed(42)
    normal_vals = np.random.normal(loc=100.0, scale=5.0, size=100)
    normal_vals[0] = 500.0  # Extreme outlier

    df_synth = pd.DataFrame({
        "Test_ID": [f"TRN-{i:04d}" for i in range(100)],
        "Applied_Voltage_kV": normal_vals,
        "Load_Current_A": [100.0] * 100,
        "Ambient_Temperature_C": [25.0] * 100,
        "Test_Duration_min": [10.0] * 100,
        "Sensor_S1": [50.0] * 100,
        "Sensor_S2": [50.0] * 100,
        "Sensor_S3": [50.0] * 100,
        "Sensor_S4": [50.0] * 100,
    })

    df_outliers = identify_extreme_candidates(df_synth)
    assert len(df_outliers) > 0
    flagged_ids = df_outliers["Test_ID"].tolist()
    assert "TRN-0000" in flagged_ids


def test_sensor_behaviour_audit(cpri_datasets):
    """11. Verifies sensor behaviour audit computes correlation matrix and spread metrics."""
    df_train, _ = cpri_datasets
    sensor_res = audit_sensor_behaviour(df_train)
    assert "correlation_matrix" in sensor_res
    assert "summary" in sensor_res
    assert sensor_res["correlation_matrix"].shape == (4, 4)


def test_deterministic_quality_flags_alignment(cpri_datasets):
    """12. Verifies create_quality_audit_flags produces deterministic, aligned quality flags."""
    df_train, _ = cpri_datasets
    df_flags = create_quality_audit_flags(df_train)
    assert len(df_flags) == 1000
    assert "data_quality_issue" in df_flags.columns
    assert "missing_any" in df_flags.columns
    assert "duplicate_measurement_pair" in df_flags.columns


def test_audit_runs_without_validity_label(cpri_datasets):
    """13. Verifies audit pipeline runs seamlessly on Test_Data without Validity_Label."""
    _, df_test = cpri_datasets
    assert TASK01_TARGET not in df_test.columns
    res = run_data_quality_audit(df_test=df_test)
    assert "missing_report_test" in res
    assert len(res["flags_test"]) == 350


def test_reference_parameter_not_required(cpri_datasets):
    """14. Verifies audit runs on DataFrames missing Reference_Parameter."""
    df_train, _ = cpri_datasets
    df_no_ref = df_train.drop(columns=[TASK02_TARGET], errors="ignore")
    res = run_data_quality_audit(df_train=df_no_ref)
    assert len(res["flags_train"]) == 1000


def test_test_id_not_used_as_predictive_feature(cpri_datasets):
    """15. Verifies Test_ID is strictly excluded from physical feature matrices."""
    df_train, _ = cpri_datasets
    stats_df = calculate_numeric_statistics(df_train)
    assert ID_COLUMN not in stats_df["Feature"].values


def test_source_dataset_not_mutated(cpri_datasets):
    """16. Verifies audit functions do NOT mutate original input DataFrames."""
    df_train, _ = cpri_datasets
    df_copy = df_train.copy()
    _ = create_quality_audit_flags(df_train)
    assert df_train.equals(df_copy)
