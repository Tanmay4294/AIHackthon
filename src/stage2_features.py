"""
CPRI Hackathon — Task 01 — Person 2
P2-STAGE 2 — FEATURE MATRIX & DATA PREPARATION MODULE

Provides functions to load training data, encode ground truth target labels (Invalid=1, Valid=0),
construct Feature Set A (Raw Inputs) and Feature Set B (Raw + Stage 1 Quality Flags),
and explicitly detect, log, and exclude constant features.

Guarantees:
- Strict exclusion of Reference_Parameter, Test_ID, and Validity_Label from feature matrices.
- Explicit detection and reporting of constant features with zero variance.
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

# Raw physical operating parameters & sensors (Feature Set A)
RAW_INPUT_COLUMNS = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4"
]

EXCLUDED_COLUMNS = [
    "Test_ID",
    "Reference_Parameter",
    "Validity_Label",
    "missing_columns"  # string column
]


def load_training_data_with_flags(dataset_path: str) -> pd.DataFrame:
    """
    Loads Training_Data from CPRI Excel file and applies existing Stage 1 quality flags.
    
    Returns:
        df_flagged: DataFrame (1,000 rows) with original data + Stage 1 quality flags.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"CPRI dataset file not found at: {dataset_path}")
        
    excel = pd.ExcelFile(dataset_path)
    df_train_raw = pd.read_excel(excel, sheet_name='Training_Data')
    
    # Enrich with existing Stage 1 quality flags
    df_flagged = create_quality_flags(df_train_raw, feature_cols=FEATURE_COLUMNS)
    return df_flagged


def detect_constant_features(df: pd.DataFrame, candidate_cols: List[str]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Detects constant features (zero variance across training records).
    
    Returns:
        active_cols: List of non-constant feature column names.
        constant_cols: List of constant feature column names.
        constant_details: Dict mapping constant column name to its constant value.
    """
    active_cols = []
    constant_cols = []
    constant_details = {}
    
    for col in candidate_cols:
        if col not in df.columns:
            continue
        series = df[col]
        # Ignore NaNs when checking uniqueness if numeric
        unique_vals = series.dropna().unique()
        if len(unique_vals) <= 1:
            constant_cols.append(col)
            val = unique_vals[0] if len(unique_vals) == 1 else "All NaN"
            constant_details[col] = val
        else:
            active_cols.append(col)
            
    return active_cols, constant_cols, constant_details


def prepare_stage2_feature_sets(df_flagged: pd.DataFrame) -> Dict[str, Any]:
    """
    Constructs Feature Set A (Raw Inputs) and Feature Set B (Raw + Stage 1 Quality Flags).
    Target y: Invalid = 1, Valid = 0.
    
    Returns a dictionary containing:
        - X_raw: DataFrame of Raw active features
        - X_raw_quality: DataFrame of Raw + Quality active features
        - y: pd.Series of binary targets (1 for Invalid, 0 for Valid)
        - raw_cols_active: List of active raw feature names
        - quality_cols_active: List of active quality feature names
        - constant_features: Dict of detected constant features and their values
        - test_ids: pd.Series of Test_ID strings
    """
    # 1. Target Encoding
    if "Validity_Label" not in df_flagged.columns:
        raise ValueError("Training data missing 'Validity_Label' column!")
        
    y = (df_flagged["Validity_Label"].str.strip() == "Invalid").astype(int)
    
    # 2. Raw Input Features
    avail_raw = [c for c in RAW_INPUT_COLUMNS if c in df_flagged.columns]
    raw_active, raw_constant, raw_const_details = detect_constant_features(df_flagged, avail_raw)
    
    # 3. Quality Feature Candidates (all boolean/numeric quality flags from Stage 1)
    stage1_quality_candidates = [
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
    
    avail_quality = [c for c in stage1_quality_candidates if c in df_flagged.columns]
    quality_active, quality_constant, quality_const_details = detect_constant_features(df_flagged, avail_quality)
    
    # Combine constant details
    constant_features = {**raw_const_details, **quality_const_details}
    
    # Construct feature matrices
    X_raw = df_flagged[raw_active].copy()
    
    all_b_cols = raw_active + quality_active
    X_raw_quality = df_flagged[all_b_cols].copy()
    
    # Safety Assertions against Target & ID Leakage
    for col in EXCLUDED_COLUMNS:
        assert col not in X_raw.columns, f"LEAKAGE ERROR: {col} found in Feature Set A!"
        assert col not in X_raw_quality.columns, f"LEAKAGE ERROR: {col} found in Feature Set B!"
        
    return {
        "X_raw": X_raw,
        "X_raw_quality": X_raw_quality,
        "y": y,
        "raw_cols_active": raw_active,
        "quality_cols_active": quality_active,
        "raw_cols_constant": raw_constant,
        "quality_cols_constant": quality_constant,
        "constant_features": constant_features,
        "test_ids": df_flagged["Test_ID"] if "Test_ID" in df_flagged.columns else pd.Series(index=df_flagged.index)
    }
