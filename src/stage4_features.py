"""
CPRI Hackathon — Task 01 — Person 2
P2-STAGE 4 — FEATURE MATRIX PREPARATION MODULE

Constructs candidate feature sets (Set A, Set B, Set C) for Stage 4 model evaluation,
integrates the Person 1 adapter interface, and explicitly detects and excludes constant features.

Guarantees:
- Strict exclusion of Reference_Parameter, Test_ID, and Validity_Label from feature matrices.
- Direct reuse of existing Stage 1 create_quality_flags() without code modification.
- Explicit detection and removal of constant zero-variance features.
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
from person1_adapter import get_person1_features, has_person1_features


def load_stage4_datasets(dataset_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads Training_Data (1,000 rows) and Test_Data (350 rows) from CPRI Excel file
    and enriches both with existing Stage 1 quality flags.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"CPRI dataset file not found at: {dataset_path}")
        
    excel = pd.ExcelFile(dataset_path)
    df_train_raw = pd.read_excel(excel, sheet_name='Training_Data')
    df_test_raw = pd.read_excel(excel, sheet_name='Test_Data')
    
    df_train_flagged = create_quality_flags(df_train_raw, feature_cols=FEATURE_COLUMNS)
    df_test_flagged = create_quality_flags(df_test_raw, feature_cols=FEATURE_COLUMNS)
    
    return df_train_flagged, df_test_flagged


def prepare_stage4_feature_sets(df_flagged: pd.DataFrame, df_test_flagged: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Constructs Candidate Feature Sets A, B, C for Training_Data and Test_Data.
    Detects and excludes constant zero-variance features based on Training_Data.
    
    Returns:
        Dict containing training/test feature matrices, constant feature logs,
        Person 1 availability status, and binary targets y.
    """
    # Extract contextual / Person 1 features via adapter
    df_ctx_train, is_genuine_p1, ctx_cols = get_person1_features(df_flagged)
    
    # Raw physical variables
    raw_cols = [c for c in RAW_INPUT_COLUMNS if c in df_flagged.columns]
    
    # Active Stage 1 Quality Flags
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
    
    # Combine all candidate features for constant detection
    df_combined_train = pd.concat([df_flagged[raw_cols], df_ctx_train, df_flagged[avail_quality]], axis=1)
    
    # Detect constant features in training data
    active_cols, constant_cols, const_details = detect_constant_features(df_combined_train, list(df_combined_train.columns))
    
    # Filter active columns for candidate sets
    set_a_cols = [c for c in (raw_cols + avail_quality) if c in active_cols]
    set_b_cols = [c for c in (raw_cols + ctx_cols) if c in active_cols]
    set_c_cols = [c for c in (raw_cols + avail_quality + ctx_cols) if c in active_cols]
    
    # Training feature matrices
    X_train_dict = {
        "Set_A": df_combined_train[set_a_cols].copy(),
        "Set_B": df_combined_train[set_b_cols].copy(),
        "Set_C": df_combined_train[set_c_cols].copy()
    }
    
    # Target binary encoding: Invalid = 1, Valid = 0
    y_train = None
    if "Validity_Label" in df_flagged.columns:
        y_train = (df_flagged["Validity_Label"].str.strip() == "Invalid").astype(int)
        
    # Test feature matrices if test data provided
    X_test_dict = {}
    if df_test_flagged is not None:
        df_ctx_test, _, _ = get_person1_features(df_test_flagged)
        df_combined_test = pd.concat([df_test_flagged[raw_cols], df_ctx_test, df_test_flagged[avail_quality]], axis=1)
        
        X_test_dict = {
            "Set_A": df_combined_test[set_a_cols].copy(),
            "Set_B": df_combined_test[set_b_cols].copy(),
            "Set_C": df_combined_test[set_c_cols].copy()
        }
        
    # Safety Assertions against Target & ID Leakage
    for set_key, X_mat in X_train_dict.items():
        for exc in EXCLUDED_COLUMNS:
            assert exc not in X_mat.columns, f"LEAKAGE ERROR: {exc} found in {set_key} feature matrix!"
            
    return {
        "X_train": X_train_dict,
        "X_test": X_test_dict,
        "y_train": y_train,
        "is_genuine_person1": is_genuine_p1,
        "contextual_cols": ctx_cols,
        "constant_features": const_details,
        "feature_set_cols": {
            "Set_A": set_a_cols,
            "Set_B": set_b_cols,
            "Set_C": set_c_cols
        },
        "test_ids_train": df_flagged["Test_ID"] if "Test_ID" in df_flagged.columns else pd.Series(index=df_flagged.index),
        "test_ids_test": df_test_flagged["Test_ID"] if df_test_flagged is not None and "Test_ID" in df_test_flagged.columns else None
    }
