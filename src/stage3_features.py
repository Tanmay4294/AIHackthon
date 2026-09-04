"""
CPRI Hackathon — Task 01 — Person 2
P2-STAGE 3 — ANOMALY FEATURE ENGINEERING & PREPARATION MODULE

Provides functions to load training and test datasets with Stage 1 quality flags,
construct physically justified engineered features, build 4 feature sets (A, B, C, D),
and detect and exclude constant features.

Guarantees:
- Strict exclusion of Reference_Parameter, Test_ID, and Validity_Label from feature matrices.
- Safe division and missing/zero handling in engineered features to prevent inf/NaN values.
- Reuses existing src/quality_features.py output flags without modification.
"""

from typing import List, Tuple, Dict, Any
import os
import pandas as pd
import numpy as np

from quality_features import (
    create_quality_flags,
    FEATURE_COLUMNS
)
from stage2_features import (
    RAW_INPUT_COLUMNS,
    EXCLUDED_COLUMNS,
    detect_constant_features
)


def load_stage3_datasets(dataset_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads Training_Data (1,000 rows) and Test_Data (350 rows) from CPRI Excel file
    and enriches both with existing Stage 1 quality flags.
    
    Returns:
        df_train_flagged: DataFrame of Training_Data (1,000 rows)
        df_test_flagged: DataFrame of Test_Data (350 rows)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"CPRI dataset file not found at: {dataset_path}")
        
    excel = pd.ExcelFile(dataset_path)
    df_train_raw = pd.read_excel(excel, sheet_name='Training_Data')
    df_test_raw = pd.read_excel(excel, sheet_name='Test_Data')
    
    df_train_flagged = create_quality_flags(df_train_raw, feature_cols=FEATURE_COLUMNS)
    df_test_flagged = create_quality_flags(df_test_raw, feature_cols=FEATURE_COLUMNS)
    
    return df_train_flagged, df_test_flagged


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs physically justified engineered features from raw operating and sensor parameters.
    
    Features created:
    - Apparent_Power_kVA: Applied_Voltage_kV * Load_Current_A
    - Sensor_Spread: max(S1..S4) - min(S1..S4)
    - Sensor_Mean: mean(S1..S4)
    - Sensor_S1_S2_Ratio: Sensor_S1 / (Sensor_S2 + 1e-5) (safely handles zero)
    - Sensor_S3_S4_Ratio: Sensor_S3 / (Sensor_S4 + 1e-5) (safely handles zero)
    """
    df_eng = pd.DataFrame(index=df.index)
    
    # 1. Apparent Power (Electrical Loading Indicator)
    if "Applied_Voltage_kV" in df.columns and "Load_Current_A" in df.columns:
        df_eng["Apparent_Power_kVA"] = df["Applied_Voltage_kV"] * df["Load_Current_A"]
        
    # Sensor columns
    sensor_cols = [c for c in ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"] if c in df.columns]
    
    if len(sensor_cols) == 4:
        sensor_matrix = df[sensor_cols]
        # 2. Sensor Spread (Max - Min across sensors)
        df_eng["Sensor_Spread"] = sensor_matrix.max(axis=1) - sensor_matrix.min(axis=1)
        # 3. Sensor Mean (Average sensor magnitude)
        df_eng["Sensor_Mean"] = sensor_matrix.mean(axis=1)
        
    # 4. Sensor S1/S2 Ratio
    if "Sensor_S1" in df.columns and "Sensor_S2" in df.columns:
        s1 = df["Sensor_S1"].fillna(0.0)
        s2 = df["Sensor_S2"].fillna(0.0)
        df_eng["Sensor_S1_S2_Ratio"] = (s1 + 1e-5) / (s2.abs() + 1e-5)
        
    # 5. Sensor S3/S4 Ratio
    if "Sensor_S3" in df.columns and "Sensor_S4" in df.columns:
        s3 = df["Sensor_S3"].fillna(0.0)
        s4 = df["Sensor_S4"].fillna(0.0)
        df_eng["Sensor_S3_S4_Ratio"] = (s3 + 1e-5) / (s4.abs() + 1e-5)
        
    # Clean up any potential inf or NaN in engineered features
    df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
    return df_eng


def prepare_stage3_feature_sets(df_flagged: pd.DataFrame, df_test_flagged: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Constructs Feature Sets A, B, C, D for Training_Data and Test_Data.
    Detects and excludes constant features based on Training_Data.
    
    Returns:
        Dict containing feature matrices for training and test data across sets A, B, C, D,
        along with constant feature logs and target vector y (if available).
    """
    df_eng_train = create_engineered_features(df_flagged)
    
    # Raw physical variables (Set A)
    raw_cols = [c for c in RAW_INPUT_COLUMNS if c in df_flagged.columns]
    
    # Engineered variables (Set B addition)
    eng_cols = list(df_eng_train.columns)
    
    # Active Stage 1 Quality Flags (Set C addition)
    quality_candidates = [
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
        "invalid_ambient_temp",
        "sensor_s2_negative",
        "sensor_zero_reading",
        "data_quality_issue"
    ]
    avail_quality = [c for c in quality_candidates if c in df_flagged.columns]
    
    # Combine raw + engineered + quality into single dataframe for constant analysis
    df_combined_train = pd.concat([df_flagged[raw_cols], df_eng_train, df_flagged[avail_quality]], axis=1)
    
    # Detect constant features in training data
    active_cols, constant_cols, const_details = detect_constant_features(df_combined_train, list(df_combined_train.columns))
    
    # Filter active columns for each feature set
    set_a_cols = [c for c in raw_cols if c in active_cols]
    set_b_cols = [c for c in (raw_cols + eng_cols) if c in active_cols]
    set_c_cols = [c for c in (raw_cols + avail_quality) if c in active_cols]
    set_d_cols = [c for c in (raw_cols + eng_cols + avail_quality) if c in active_cols]
    
    # Construct training feature matrices
    X_train_dict = {
        "Set_A": df_combined_train[set_a_cols].copy(),
        "Set_B": df_combined_train[set_b_cols].copy(),
        "Set_C": df_combined_train[set_c_cols].copy(),
        "Set_D": df_combined_train[set_d_cols].copy()
    }
    
    # Target encoding if available (Training_Data)
    y_train = None
    if "Validity_Label" in df_flagged.columns:
        y_train = (df_flagged["Validity_Label"].str.strip() == "Invalid").astype(int)
        
    # Construct test feature matrices if test data provided
    X_test_dict = {}
    if df_test_flagged is not None:
        df_eng_test = create_engineered_features(df_test_flagged)
        df_combined_test = pd.concat([df_test_flagged[raw_cols], df_eng_test, df_test_flagged[avail_quality]], axis=1)
        
        X_test_dict = {
            "Set_A": df_combined_test[set_a_cols].copy(),
            "Set_B": df_combined_test[set_b_cols].copy(),
            "Set_C": df_combined_test[set_c_cols].copy(),
            "Set_D": df_combined_test[set_d_cols].copy()
        }
        
    # Safety assertions against leakage
    for set_key, X_mat in X_train_dict.items():
        for exc in EXCLUDED_COLUMNS:
            assert exc not in X_mat.columns, f"LEAKAGE ERROR: {exc} found in {set_key} feature matrix!"
            
    return {
        "X_train": X_train_dict,
        "X_test": X_test_dict,
        "y_train": y_train,
        "constant_features": const_details,
        "feature_set_cols": {
            "Set_A": set_a_cols,
            "Set_B": set_b_cols,
            "Set_C": set_c_cols,
            "Set_D": set_d_cols
        },
        "test_ids_train": df_flagged["Test_ID"] if "Test_ID" in df_flagged.columns else pd.Series(index=df_flagged.index),
        "test_ids_test": df_test_flagged["Test_ID"] if df_test_flagged is not None and "Test_ID" in df_test_flagged.columns else None
    }
