"""
CPRI Hackathon — Task 01 — Person 2 / Person 1 Feature Adapter Interface

Provides a clean interface for integrating Person 1's contextual, residual, consistency,
and regime features into Person 2's workflow.

If Person 1 feature files/columns are present in the DataFrame, this module extracts them directly.
If Person 1 feature columns are NOT present, this module falls back to generating validated
Person 2 engineered/contextual features (Apparent_Power_kVA, Sensor_Spread, Sensor_Mean,
Sensor_S1_S2_Ratio, Sensor_S3_S4_Ratio) and clearly labels them as fallback features.
"""

from typing import List, Tuple, Dict, Any
import os
import pandas as pd
import numpy as np


def has_person1_features(df: pd.DataFrame) -> bool:
    """
    Checks if genuine Person 1 feature columns are present in the DataFrame.
    Person 1 features typically include residual, consistency, or regime indicators.
    """
    person1_indicators = [
        "p1_residual_error",
        "p1_regime_cluster",
        "p1_consistency_index",
        "p1_regime_abnormality",
        "p1_max_abs_residual",
        "p1_mean_abs_residual",
        "p1_sensor_disagreement_index"
    ]
    return any(col in df.columns for col in person1_indicators)


def get_person1_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool, List[str]]:
    """
    Extracts or generates regime/contextual features for the input DataFrame.
    
    Returns:
        df_features: DataFrame containing contextual/regime features.
        is_genuine_person1: Boolean flag (True if genuine Person 1 features present, False if fallback used).
        feature_names: List of generated/extracted feature names.
    """
    if has_person1_features(df):
        person1_cols = [c for c in df.columns if c.startswith("p1_") or "Residual" in c or "Expected" in c]
        return df[person1_cols].copy(), True, person1_cols
        
    # Fallback to Person 2 Engineered / Contextual Features
    df_fallback = pd.DataFrame(index=df.index)
    
    # 1. Apparent Power (Electrical Loading Magnitude)
    if "Applied_Voltage_kV" in df.columns and "Load_Current_A" in df.columns:
        df_fallback["Apparent_Power_kVA"] = df["Applied_Voltage_kV"] * df["Load_Current_A"]
        
    # Sensor columns
    sensor_cols = [c for c in ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"] if c in df.columns]
    
    if len(sensor_cols) == 4:
        sensor_matrix = df[sensor_cols]
        # 2. Sensor Spread (Max - Min across sensors)
        df_fallback["Sensor_Spread"] = sensor_matrix.max(axis=1) - sensor_matrix.min(axis=1)
        # 3. Sensor Mean (Average sensor magnitude)
        df_fallback["Sensor_Mean"] = sensor_matrix.mean(axis=1)
        
    # 4. Sensor S1/S2 Ratio (Safely handle zero denominator)
    if "Sensor_S1" in df.columns and "Sensor_S2" in df.columns:
        s1 = df["Sensor_S1"].fillna(0.0)
        s2 = df["Sensor_S2"].fillna(0.0)
        df_fallback["Sensor_S1_S2_Ratio"] = (s1 + 1e-5) / (s2.abs() + 1e-5)
        
    # 5. Sensor S3/S4 Ratio (Safely handle zero denominator)
    if "Sensor_S3" in df.columns and "Sensor_S4" in df.columns:
        s3 = df["Sensor_S3"].fillna(0.0)
        s4 = df["Sensor_S4"].fillna(0.0)
        df_fallback["Sensor_S3_S4_Ratio"] = (s3 + 1e-5) / (s4.abs() + 1e-5)
        
    # Clean inf/NaN
    df_fallback = df_fallback.replace([np.inf, -np.inf], np.nan)
    return df_fallback, False, list(df_fallback.columns)
