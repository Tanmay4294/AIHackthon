"""
CPRI Hackathon — Person 1 Stage 1 Unit Tests
=============================================
Automated test suite verifying dataset loading, workbook integrity, sheet schemas,
`Test_ID` uniqueness/formats, schema separation rules, relative path resolution,
and non-destructive behavior.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.dataset_loader import (
    DATASET_PATH,
    PHYSICAL_FEATURES,
    ID_COLUMN,
    TASK01_TARGET,
    TASK02_TARGET,
    get_dataset_path,
    load_sheet,
    load_training_data,
    load_test_data,
    load_readme,
    load_sample_submission,
    get_feature_schema,
    extract_physical_feature_matrix,
)
from src.quality_features import create_quality_flags, generate_quality_summary


def test_workbook_file_exists():
    """1. Verifies dataset Excel file exists at expected path."""
    path = get_dataset_path()
    assert path.exists(), f"Excel file missing at {path}"
    assert path.is_file(), f"Path {path} is not a file"


def test_all_4_sheets_present():
    """2. Verifies all 4 expected sheets exist in the Excel workbook."""
    path = get_dataset_path()
    excel_file = pd.ExcelFile(path)
    expected_sheets = {"README", "Training_Data", "Test_Data", "Sample_Submission"}
    actual_sheets = set(excel_file.sheet_names)
    assert expected_sheets.issubset(actual_sheets), f"Missing sheets: {expected_sheets - actual_sheets}"


def test_training_row_count():
    """3. Verifies Training_Data has exactly 1,000 rows."""
    df_train = load_training_data()
    assert len(df_train) == 1000, f"Expected 1,000 training rows, got {len(df_train)}"


def test_test_row_count():
    """4. Verifies Test_Data has exactly 350 rows."""
    df_test = load_test_data()
    assert len(df_test) == 350, f"Expected 350 test rows, got {len(df_test)}"


def test_physical_features_schema():
    """5. Verifies physical input feature names and count (8 features)."""
    schema = get_feature_schema()
    assert len(schema["physical_features"]) == 8
    expected = [
        "Applied_Voltage_kV",
        "Load_Current_A",
        "Ambient_Temperature_C",
        "Test_Duration_min",
        "Sensor_S1",
        "Sensor_S2",
        "Sensor_S3",
        "Sensor_S4",
    ]
    assert schema["physical_features"] == expected


def test_test_id_present_in_training_and_test():
    """6. Verifies Test_ID column is present in both Training and Test datasets."""
    df_train = load_training_data()
    df_test = load_test_data()
    assert ID_COLUMN in df_train.columns
    assert ID_COLUMN in df_test.columns


def test_training_test_id_uniqueness():
    """7. Verifies Test_ID values in Training dataset are 100% unique and follow TRN- format."""
    df_train = load_training_data()
    assert df_train[ID_COLUMN].nunique() == 1000
    assert df_train[ID_COLUMN].astype(str).str.startswith("TRN-").all()


def test_test_test_id_uniqueness():
    """8. Verifies Test_ID values in Test dataset are 100% unique and follow TST- format."""
    df_test = load_test_data()
    assert df_test[ID_COLUMN].nunique() == 350
    assert df_test[ID_COLUMN].astype(str).str.startswith("TST-").all()


def test_validity_label_present_in_training():
    """9. Verifies Validity_Label exists in Training_Data."""
    df_train = load_training_data()
    assert TASK01_TARGET in df_train.columns
    assert set(df_train[TASK01_TARGET].unique()) == {"Valid", "Invalid"}


def test_validity_label_absent_in_test():
    """10. Verifies Validity_Label is absent in Test_Data."""
    df_test = load_test_data()
    assert TASK01_TARGET not in df_test.columns


def test_reference_parameter_present_in_training():
    """11. Verifies Reference_Parameter exists in Training_Data."""
    df_train = load_training_data()
    assert TASK02_TARGET in df_train.columns


def test_extract_physical_feature_matrix_training():
    """12. Verifies extract_physical_feature_matrix extracts 8 physical features."""
    df_train = load_training_data()
    X_train = extract_physical_feature_matrix(df_train)
    assert X_train.shape == (1000, 8)
    assert list(X_train.columns) == PHYSICAL_FEATURES


def test_test_id_excluded_from_feature_matrix():
    """13. Verifies Test_ID is strictly excluded from feature matrix X."""
    df_train = load_training_data()
    X_train = extract_physical_feature_matrix(df_train)
    assert ID_COLUMN not in X_train.columns


def test_targets_excluded_from_feature_matrix():
    """14. Verifies Validity_Label and Reference_Parameter are excluded from feature matrix X."""
    df_train = load_training_data()
    X_train = extract_physical_feature_matrix(df_train)
    assert TASK01_TARGET not in X_train.columns
    assert TASK02_TARGET not in X_train.columns


def test_non_destructive_data_loading():
    """15. Verifies data loading is read-only and non-destructive."""
    df1 = load_training_data()
    df2 = load_training_data()
    assert df1.equals(df2)


def test_relative_path_resolution():
    """16. Verifies get_dataset_path resolves via relative path inside data directory."""
    path = get_dataset_path()
    assert "data" in path.parts
    assert path.name == "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"


def test_person2_quality_features_compatibility():
    """17. Verifies Person 2 quality_features module functions seamlessly with loaded dataset."""
    df_train = load_training_data()
    df_q = create_quality_flags(df_train)
    assert len(df_q) == 1000
    assert "data_quality_issue" in df_q.columns
    assert "missing_any" in df_q.columns


def test_sample_submission_schema():
    """18. Verifies Sample_Submission has 350 rows and expected submission columns."""
    df_sub = load_sample_submission()
    assert len(df_sub) == 350
    assert list(df_sub.columns) == [ID_COLUMN, "Predicted_Reference_Parameter", TASK01_TARGET]
