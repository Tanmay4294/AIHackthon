"""
CPRI Hackathon — Task 01 — Person 2
P2-STAGE 1 — DATA QUALITY AUDIT MODULE

This module provides reusable, deterministic, row-level data quality auditing functions
for the CPRI Screening Dataset (Training_Data and Test_Data).

Guarantees:
- Zero row deletion (original datasets preserved)
- No arbitrary thresholds / no confusion between operating regime shifts and quality defects
- Deterministic and 100% reproducible flags
- Works with or without Validity_Label
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

# Standard CPRI Feature Columns
FEATURE_COLUMNS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4"
]

CRITICAL_MEASUREMENTS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min"
]


def audit_missing_values(df: pd.DataFrame, feature_cols: List[str] = FEATURE_COLUMNS) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Identifies rows containing missing (NaN/Null/None) values.
    
    Returns:
        missing_any: Boolean Series, True if any feature column is NaN
        missing_columns: String Series listing comma-separated missing column names per row
        missing_critical: Boolean Series, True if any critical measurement is missing
    """
    missing_mask = df[feature_cols].isna()
    missing_any = missing_mask.any(axis=1)
    
    # Identify specific missing column names per row
    def get_missing_cols(row):
        cols = [col for col in feature_cols if pd.isna(row[col])]
        return ",".join(cols) if cols else "None"
        
    missing_columns = df[feature_cols].apply(get_missing_cols, axis=1)
    
    # Check critical measurements
    avail_critical = [c for c in CRITICAL_MEASUREMENTS if c in df.columns]
    missing_critical = df[avail_critical].isna().any(axis=1) if avail_critical else pd.Series(False, index=df.index)
    
    return missing_any, missing_columns, missing_critical


def audit_non_finite_values(df: pd.DataFrame, feature_cols: List[str] = FEATURE_COLUMNS) -> pd.Series:
    """
    Detects NaN, +Infinity, or -Infinity values across numeric features.
    """
    non_finite = pd.Series(False, index=df.index)
    for col in feature_cols:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').to_numpy()
            non_finite |= ~np.isfinite(vals)
    return non_finite


def audit_duplicate_rows(df: pd.DataFrame) -> pd.Series:
    """
    Identifies exact 100% duplicate rows across all columns.
    """
    return df.duplicated(keep='first')


def audit_duplicate_test_ids(df: pd.DataFrame, id_col: str = "Test_ID") -> pd.Series:
    """
    Identifies whether Test_ID occurs more than once.
    """
    if id_col not in df.columns:
        return pd.Series(False, index=df.index)
    return df.duplicated(subset=[id_col], keep=False)


def audit_duplicate_measurement_pairs(df: pd.DataFrame, feature_cols: List[str] = FEATURE_COLUMNS) -> pd.Series:
    """
    Identifies records sharing identical measurement feature vectors across different Test_IDs.
    (Matches README note: 'duplicate measurement pairs')
    """
    avail_cols = [c for c in feature_cols if c in df.columns]
    return df.duplicated(subset=avail_cols, keep=False)


def audit_numeric_columns(df: pd.DataFrame, feature_cols: List[str] = FEATURE_COLUMNS) -> pd.Series:
    """
    Checks for malformed numeric strings or values that cannot safely be parsed as floats.
    """
    malformed = pd.Series(False, index=df.index)
    for col in feature_cols:
        if col in df.columns:
            # Try parsing to numeric; if original was non-null but coercion produces NaN, it's malformed
            parsed = pd.to_numeric(df[col], errors='coerce')
            malformed |= (df[col].notna() & parsed.isna())
    return malformed


def audit_physical_constraints(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Verifies physically justified constraints based on CPRI test rig setup properties.
    
    Constraints applied:
      - invalid_voltage: Applied_Voltage_kV < 0 (Electrical voltage magnitude must be >= 0)
      - invalid_current: Load_Current_A < 0 (Load current magnitude must be >= 0)
      - invalid_duration: Test_Duration_min <= 0 (Test duration must be strictly positive)
      - invalid_ambient_temp: Ambient_Temperature_C < -273.15 (Below absolute zero)
      - invalid_sensor_negative: Sensor_S2 < 0 (Sensor_S2 negative sensor drop)
      - invalid_sensor_zero: Sensor_S1 == 0 or Sensor_S2 == 0 or Sensor_S3 == 0 (Unusual exact zero sensor drops)
      
    Note: High sensor readings (e.g. S1 > 20) are NOT flagged as invalid because they represent
    genuine operating regime shifts under high load.
    """
    flags = {}
    
    # 1. Electrical & Operating constraints
    if "Applied_Voltage_kV" in df.columns:
        flags["invalid_voltage"] = (pd.to_numeric(df["Applied_Voltage_kV"], errors='coerce') < 0)
    else:
        flags["invalid_voltage"] = pd.Series(False, index=df.index)
        
    if "Load_Current_A" in df.columns:
        flags["invalid_current"] = (pd.to_numeric(df["Load_Current_A"], errors='coerce') < 0)
    else:
        flags["invalid_current"] = pd.Series(False, index=df.index)

    if "Test_Duration_min" in df.columns:
        flags["invalid_duration"] = (pd.to_numeric(df["Test_Duration_min"], errors='coerce') <= 0)
    else:
        flags["invalid_duration"] = pd.Series(False, index=df.index)

    if "Ambient_Temperature_C" in df.columns:
        flags["invalid_ambient_temp"] = (pd.to_numeric(df["Ambient_Temperature_C"], errors='coerce') < -273.15)
    else:
        flags["invalid_ambient_temp"] = pd.Series(False, index=df.index)

    # 2. Sensor zero/negative drops
    if "Sensor_S2" in df.columns:
        flags["invalid_sensor_negative"] = (pd.to_numeric(df["Sensor_S2"], errors='coerce') < 0)
    else:
        flags["invalid_sensor_negative"] = pd.Series(False, index=df.index)

    s1_zero = (pd.to_numeric(df["Sensor_S1"], errors='coerce') == 0) if "Sensor_S1" in df.columns else pd.Series(False, index=df.index)
    s2_zero = (pd.to_numeric(df["Sensor_S2"], errors='coerce') == 0) if "Sensor_S2" in df.columns else pd.Series(False, index=df.index)
    s3_zero = (pd.to_numeric(df["Sensor_S3"], errors='coerce') == 0) if "Sensor_S3" in df.columns else pd.Series(False, index=df.index)
    flags["invalid_sensor_zero"] = (s1_zero | s2_zero | s3_zero)

    return flags


def create_quality_flags(df: pd.DataFrame, feature_cols: List[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """
    Primary pipeline function that constructs all deterministic row-level quality flags.
    
    Preserves 100% of input rows and original columns.
    
    Args:
        df: Raw CPRI input DataFrame (Training_Data or Test_Data).
        feature_cols: List of measurement feature columns.
        
    Returns:
        DataFrame enriched with quality flag columns:
          - missing_any
          - missing_columns
          - missing_critical_measurement
          - non_finite_value
          - duplicate_full_row
          - duplicate_test_id
          - duplicate_measurement_pair
          - malformed_numeric
          - invalid_voltage
          - invalid_current
          - invalid_duration
          - invalid_sensor_negative
          - invalid_sensor_zero
          - data_quality_issue_count
          - data_quality_issue
    """
    df_out = df.copy()
    
    # A. Missing Values
    missing_any, missing_cols, missing_crit = audit_missing_values(df, feature_cols=feature_cols)
    df_out["missing_any"] = missing_any
    df_out["missing_columns"] = missing_cols
    df_out["missing_critical_measurement"] = missing_crit
    
    # B. Non-finite values
    df_out["non_finite_value"] = audit_non_finite_values(df, feature_cols=feature_cols)
    
    # C. Duplicate complete rows
    df_out["duplicate_full_row"] = audit_duplicate_rows(df)
    
    # D. Duplicate Test_IDs
    df_out["duplicate_test_id"] = audit_duplicate_test_ids(df, id_col="Test_ID")
    
    # E. Duplicate measurement pairs
    df_out["duplicate_measurement_pair"] = audit_duplicate_measurement_pairs(df, feature_cols=feature_cols)
    
    # F. Malformed numeric values
    df_out["malformed_numeric"] = audit_numeric_columns(df, feature_cols=feature_cols)
    
    # G. Physical constraints
    physical_flags = audit_physical_constraints(df)
    for pflag_name, pflag_series in physical_flags.items():
        df_out[pflag_name] = pflag_series

    # List of primary quality indicator flags
    primary_indicator_flags = [
        "missing_any",
        "non_finite_value",
        "duplicate_full_row",
        "duplicate_test_id",
        "duplicate_measurement_pair",
        "malformed_numeric",
        "invalid_voltage",
        "invalid_current",
        "invalid_duration",
        "invalid_ambient_temp",
        "invalid_sensor_negative",
        "invalid_sensor_zero"
    ]
    
    # Count of data quality issues per record
    df_out["data_quality_issue_count"] = df_out[primary_indicator_flags].astype(int).sum(axis=1)
    
    # Combined binary flag
    df_out["data_quality_issue"] = (df_out["data_quality_issue_count"] > 0)
    
    return df_out


def generate_quality_summary(df_train_flagged: pd.DataFrame, df_test_flagged: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generates dataset-level audit summary table for Training_Data and Test_Data.
    Include Valid / Invalid breakdown for Training_Data.
    """
    flag_cols = [
        "missing_any",
        "missing_critical_measurement",
        "non_finite_value",
        "duplicate_full_row",
        "duplicate_test_id",
        "duplicate_measurement_pair",
        "malformed_numeric",
        "invalid_voltage",
        "invalid_current",
        "invalid_duration",
        "invalid_sensor_negative",
        "invalid_sensor_zero",
        "data_quality_issue"
    ]
    
    rows = []
    
    n_train = len(df_train_flagged)
    n_test = len(df_test_flagged) if df_test_flagged is not None else 0
    has_validity = "Validity_Label" in df_train_flagged.columns
    
    for fcol in flag_cols:
        if fcol not in df_train_flagged.columns:
            continue
            
        trn_cnt = int(df_train_flagged[fcol].sum())
        trn_pct = round(trn_cnt / n_train * 100.0, 2) if n_train > 0 else 0.0
        
        row_dict = {
            "Quality_Flag": fcol,
            "Train_Flagged_Count": trn_cnt,
            "Train_Prevalence (%)": trn_pct
        }
        
        if has_validity:
            valid_cnt = int((df_train_flagged[fcol] & (df_train_flagged["Validity_Label"] == "Valid")).sum())
            invalid_cnt = int((df_train_flagged[fcol] & (df_train_flagged["Validity_Label"] == "Invalid")).sum())
            row_dict["Train_Valid_Count"] = valid_cnt
            row_dict["Train_Invalid_Count"] = invalid_cnt
            row_dict["Train_Valid_Pct (%)"] = round(valid_cnt / trn_cnt * 100.0, 2) if trn_cnt > 0 else 0.0
            row_dict["Train_Invalid_Pct (%)"] = round(invalid_cnt / trn_cnt * 100.0, 2) if trn_cnt > 0 else 0.0

        if df_test_flagged is not None and fcol in df_test_flagged.columns:
            tst_cnt = int(df_test_flagged[fcol].sum())
            tst_pct = round(tst_cnt / n_test * 100.0, 2) if n_test > 0 else 0.0
            row_dict["Test_Flagged_Count"] = tst_cnt
            row_dict["Test_Prevalence (%)"] = tst_pct

        rows.append(row_dict)

    return pd.DataFrame(rows)
