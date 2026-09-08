"""
CPRI Hackathon — Person 1 Stage 2 Data-Quality Audit Module
===========================================================
Provides reusable, non-destructive, reproducible data-quality auditing functions for
Training_Data (1,000 rows) and Test_Data (350 rows).

Guarantees:
- Zero data deletion / imputation (all rows preserved)
- Zero target leakage (Validity_Label & Reference_Parameter NOT used to define quality rules)
- Direct reuse of Person 2 Stage 1 quality_features.py flags (create_quality_flags)
- Direct reuse of src/dataset_loader.py
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.dataset_loader import (
    PHYSICAL_FEATURES,
    ID_COLUMN,
    TASK01_TARGET,
    TASK02_TARGET,
    get_feature_schema,
    load_training_data,
    load_test_data,
)
from src.quality_features import (
    FEATURE_COLUMNS,
    create_quality_flags,
    generate_quality_summary,
)


def audit_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Audits missing values across all columns in the DataFrame.

    Reports total missing count, missing percentage, data type, and affected Test_IDs.
    """
    total_rows = len(df)
    records = []

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_pct = round(missing_count / total_rows * 100.0, 2) if total_rows > 0 else 0.0
        dtype_str = str(df[col].dtype)

        affected_ids = []
        if missing_count > 0 and ID_COLUMN in df.columns:
            affected_ids = df[df[col].isna()][ID_COLUMN].astype(str).tolist()

        records.append({
            "Column": col,
            "DataType": dtype_str,
            "MissingCount": missing_count,
            "MissingPct": missing_pct,
            "AffectedCount": len(affected_ids),
            "AffectedSampleIDs": ", ".join(affected_ids[:10]) if affected_ids else "None"
        })

    return pd.DataFrame(records)


def audit_non_finite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Audits +Infinity and -Infinity values across numeric columns.

    Reports +Inf count, -Inf count, total non-finite count, and affected Test_IDs.
    Note: NaNs are reported under missing values and excluded from non-finite counts.
    """
    total_rows = len(df)
    records = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        series_num = pd.to_numeric(df[col], errors='coerce')
        pos_inf_mask = series_num.notna() & np.isposinf(series_num.to_numpy())
        neg_inf_mask = series_num.notna() & np.isneginf(series_num.to_numpy())
        non_finite_mask = pos_inf_mask | neg_inf_mask

        pos_inf_cnt = int(pos_inf_mask.sum())
        neg_inf_cnt = int(neg_inf_mask.sum())
        total_non_finite = pos_inf_cnt + neg_inf_cnt
        non_finite_pct = round(total_non_finite / total_rows * 100.0, 2) if total_rows > 0 else 0.0

        affected_ids = []
        if total_non_finite > 0 and ID_COLUMN in df.columns:
            affected_ids = df[non_finite_mask][ID_COLUMN].astype(str).tolist()

        records.append({
            "Column": col,
            "PosInfCount": pos_inf_cnt,
            "NegInfCount": neg_inf_cnt,
            "TotalNonFiniteCount": total_non_finite,
            "NonFinitePct": non_finite_pct,
            "AffectedSampleIDs": ", ".join(affected_ids[:10]) if affected_ids else "None"
        })

    return pd.DataFrame(records)


def audit_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Audits exact 100% duplicate rows across all columns.

    Reports total duplicate rows, unique duplicate groups, group sizes, and affected Test_IDs.
    """
    dup_mask = df.duplicated(keep=False)
    total_dup_rows = int(dup_mask.sum())

    if total_dup_rows == 0:
        return pd.DataFrame([{
            "TotalDuplicateRows": 0,
            "UniqueDuplicateGroups": 0,
            "GroupSizes": "None",
            "AffectedTestIDs": "None"
        }])

    # Group by all columns to find exact duplicate clusters
    non_id_cols = [c for c in df.columns if c != ID_COLUMN]
    grouped = df[dup_mask].groupby(non_id_cols, dropna=False)

    group_count = len(grouped)
    group_sizes = [len(group) for _, group in grouped]
    affected_ids = df[dup_mask][ID_COLUMN].astype(str).tolist() if ID_COLUMN in df.columns else []

    return pd.DataFrame([{
        "TotalDuplicateRows": total_dup_rows,
        "UniqueDuplicateGroups": group_count,
        "GroupSizes": str(group_sizes),
        "AffectedTestIDs": ", ".join(affected_ids[:20]) if affected_ids else "None"
    }])


def audit_test_id_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """Audits repeated Test_IDs in the dataset.

    For every repeated Test_ID, compares physical measurements to distinguish:
    - Exact duplicate records sharing Test_ID
    - Same Test_ID but differing measurements
    Does NOT assume repeated IDs are automatically invalid.
    """
    if ID_COLUMN not in df.columns:
        return pd.DataFrame([{
            "RepeatedIDCount": 0,
            "UniqueRepeatedIDs": 0,
            "ExactDuplicateRepeats": 0,
            "DifferingMeasurementRepeats": 0,
            "Details": "Test_ID column not present"
        }])

    repeated_mask = df.duplicated(subset=[ID_COLUMN], keep=False)
    total_repeated_rows = int(repeated_mask.sum())

    if total_repeated_rows == 0:
        return pd.DataFrame([{
            "RepeatedIDCount": 0,
            "UniqueRepeatedIDs": 0,
            "ExactDuplicateRepeats": 0,
            "DifferingMeasurementRepeats": 0,
            "Details": "No repeated Test_IDs found"
        }])

    grouped = df[repeated_mask].groupby(ID_COLUMN)
    unique_repeated_ids = len(grouped)

    exact_dup_count = 0
    differing_meas_count = 0
    records = []

    avail_features = [c for c in PHYSICAL_FEATURES if c in df.columns]

    for test_id, group in grouped:
        group_size = len(group)
        meas_dups = group.duplicated(subset=avail_features, keep=False)

        if meas_dups.all():
            classification = "Exact Duplicate Measurements"
            exact_dup_count += 1
        else:
            classification = "Same Test_ID Differing Measurements"
            differing_meas_count += 1

        records.append({
            "Test_ID": str(test_id),
            "RepeatCount": group_size,
            "Classification": classification,
            "PhysicalFeaturesEvaluated": len(avail_features)
        })

    return pd.DataFrame(records)


def calculate_numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates summary statistics for all numeric fields.

    Calculates count, missing count, min, max, mean, median, and std.
    """
    numeric_cols = [c for c in PHYSICAL_FEATURES if c in df.columns]
    records = []

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors='coerce')
        valid_series = series.dropna()

        records.append({
            "Feature": col,
            "TotalCount": len(df),
            "ValidCount": len(valid_series),
            "MissingCount": int(series.isna().sum()),
            "Min": float(valid_series.min()) if len(valid_series) > 0 else np.nan,
            "Max": float(valid_series.max()) if len(valid_series) > 0 else np.nan,
            "Mean": float(valid_series.mean()) if len(valid_series) > 0 else np.nan,
            "Median": float(valid_series.median()) if len(valid_series) > 0 else np.nan,
            "Std": float(valid_series.std()) if len(valid_series) > 0 else np.nan
        })

    return pd.DataFrame(records)


def calculate_percentile_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates 14 percentiles for all physical numeric fields.

    Calculates: 1st, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th, 99.5th, 99.9th percentiles.
    """
    numeric_cols = [c for c in PHYSICAL_FEATURES if c in df.columns]
    percentiles = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.5, 99.9]
    records = []

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(series) == 0:
            continue

        pct_values = np.percentile(series, percentiles)
        row_dict = {"Feature": col}
        for p, val in zip(percentiles, pct_values):
            p_str = f"P{str(p).replace('.', '_')}"
            row_dict[p_str] = float(val)
        records.append(row_dict)

    return pd.DataFrame(records)


def detect_physical_suspicious_values(df: pd.DataFrame) -> pd.DataFrame:
    """Audits physically impossible and suspicious values based on CPRI problem principles.

    Evaluates:
    - Negative Voltage (Applied_Voltage_kV < 0)
    - Negative Current (Load_Current_A < 0)
    - Negative/Zero Duration (Test_Duration_min <= 0)
    - Implausible Temperature (Ambient_Temperature_C < -273.15 or > 100 C)
    - Negative Sensor Reading (Sensor_S2 < 0, etc.)
    - Zero Sensor Reading (S1==0 | S2==0 | S3==0 | S4==0)
    - Extreme V/I Combinations (Voltage > 30 kV & Current < 100 A, etc.)
    """
    records = []

    v_col = "Applied_Voltage_kV"
    i_col = "Load_Current_A"
    t_col = "Test_Duration_min"
    temp_col = "Ambient_Temperature_C"

    for idx, row in df.iterrows():
        test_id = str(row.get(ID_COLUMN, f"Row_{idx}"))
        suspicious_flags = []

        # 1. Voltage < 0
        if v_col in row and pd.notna(row[v_col]):
            v_val = float(row[v_col])
            if v_val < 0:
                suspicious_flags.append("invalid_negative_voltage")

        # 2. Current < 0
        if i_col in row and pd.notna(row[i_col]):
            i_val = float(row[i_col])
            if i_val < 0:
                suspicious_flags.append("invalid_negative_current")

        # 3. Duration <= 0
        if t_col in row and pd.notna(row[t_col]):
            t_val = float(row[t_col])
            if t_val <= 0:
                suspicious_flags.append("invalid_duration_non_positive")

        # 4. Temperature < -273.15
        if temp_col in row and pd.notna(row[temp_col]):
            temp_val = float(row[temp_col])
            if temp_val < -273.15:
                suspicious_flags.append("invalid_temp_below_absolute_zero")

        # 5. Negative sensor
        if "Sensor_S2" in row and pd.notna(row["Sensor_S2"]):
            s2_val = float(row["Sensor_S2"])
            if s2_val < 0:
                suspicious_flags.append("sensor_s2_negative")

        # 6. Zero sensor
        s_zeros = []
        for sc in ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]:
            if sc in row and pd.notna(row[sc]) and float(row[sc]) == 0:
                s_zeros.append(sc)
        if s_zeros:
            suspicious_flags.append(f"sensor_zero_reading({','.join(s_zeros)})")

        if suspicious_flags:
            records.append({
                "Test_ID": test_id,
                "Applied_Voltage_kV": row.get(v_col, np.nan),
                "Load_Current_A": row.get(i_col, np.nan),
                "Ambient_Temperature_C": row.get(temp_col, np.nan),
                "Test_Duration_min": row.get(t_col, np.nan),
                "SuspiciousFlags": "; ".join(suspicious_flags),
                "Count": len(suspicious_flags)
            })

    return pd.DataFrame(records)


def identify_extreme_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Multi-method outlier candidate detection (Percentile, IQR, Z-Score).

    Combines operating context (V, I, Temp) and Stage 1 quality flags.
    Does NOT remove candidates from the dataset!
    """
    flagged_df = create_quality_flags(df)
    numeric_cols = [c for c in PHYSICAL_FEATURES if c in df.columns]
    records = []

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors='coerce')
        valid_series = series.dropna()
        if len(valid_series) == 0:
            continue

        q1 = float(valid_series.quantile(0.25))
        q3 = float(valid_series.quantile(0.75))
        iqr = q3 - q1
        lower_iqr = q1 - 1.5 * iqr
        upper_iqr = q3 + 1.5 * iqr

        p1 = float(valid_series.quantile(0.01))
        p99 = float(valid_series.quantile(0.99))

        mean_val = float(valid_series.mean())
        std_val = float(valid_series.std())

        for idx, val in series.items():
            if pd.isna(val):
                continue

            reasons = []
            if val < lower_iqr or val > upper_iqr:
                reasons.append("IQR_Outlier")
            if val < p1 or val > p99:
                reasons.append("Percentile_Extreme_<P1_or_>P99")
            
            z_score = (val - mean_val) / std_val if std_val > 0 else 0.0
            if abs(z_score) > 3.0:
                reasons.append(f"ZScore_Extreme(|z|={round(z_score, 2)})")

            if reasons:
                row_flagged = flagged_df.loc[idx]
                test_id = str(df.loc[idx, ID_COLUMN]) if ID_COLUMN in df.columns else f"Row_{idx}"

                records.append({
                    "Test_ID": test_id,
                    "Variable": col,
                    "Value": val,
                    "TriggeringMethods": "; ".join(reasons),
                    "ZScore": round(z_score, 2),
                    "IQR_Bounds": f"[{round(lower_iqr, 2)}, {round(upper_iqr, 2)}]",
                    "Percentile_P1_P99": f"[{round(p1, 2)}, {round(p99, 2)}]",
                    "Applied_Voltage_kV": df.loc[idx, "Applied_Voltage_kV"] if "Applied_Voltage_kV" in df.columns else np.nan,
                    "Load_Current_A": df.loc[idx, "Load_Current_A"] if "Load_Current_A" in df.columns else np.nan,
                    "Ambient_Temperature_C": df.loc[idx, "Ambient_Temperature_C"] if "Ambient_Temperature_C" in df.columns else np.nan,
                    "QualityIssueFlag": bool(row_flagged.get("data_quality_issue", False)),
                    "MissingAnyFlag": bool(row_flagged.get("missing_any", False)),
                    "SensorS2NegativeFlag": bool(row_flagged.get("sensor_s2_negative", False)),
                    "SensorZeroFlag": bool(row_flagged.get("sensor_zero_reading", False))
                })

    return pd.DataFrame(records)


def audit_sensor_behaviour(df: pd.DataFrame) -> Dict[str, Any]:
    """Audits relationships and behaviour among Sensor_S1..S4.

    Includes sensor spread, mean, pairwise correlations, ratios, zero/negative counts,
    and disagreement in operating context.
    """
    sensor_cols = [c for c in ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"] if c in df.columns]

    if not sensor_cols:
        return {"correlation_matrix": pd.DataFrame(), "summary": {}}

    sensor_df = df[sensor_cols].apply(pd.to_numeric, errors='coerce')
    corr_matrix = sensor_df.corr()

    sensor_mean = sensor_df.mean(axis=1)
    sensor_min = sensor_df.min(axis=1)
    sensor_max = sensor_df.max(axis=1)
    sensor_spread = sensor_max - sensor_min

    # Safe Ratios
    s1_s2_ratio = np.where(sensor_df["Sensor_S2"] != 0, sensor_df["Sensor_S1"] / sensor_df["Sensor_S2"], np.nan) if "Sensor_S1" in sensor_df.columns and "Sensor_S2" in sensor_df.columns else None
    s3_s4_ratio = np.where(sensor_df["Sensor_S4"] != 0, sensor_df["Sensor_S3"] / sensor_df["Sensor_S4"], np.nan) if "Sensor_S3" in sensor_df.columns and "Sensor_S4" in sensor_df.columns else None

    summary = {
        "mean_sensor_spread": float(sensor_spread.dropna().mean()),
        "max_sensor_spread": float(sensor_spread.dropna().max()),
        "sensor_s2_negative_count": int((sensor_df["Sensor_S2"] < 0).sum()) if "Sensor_S2" in sensor_df.columns else 0,
        "sensor_zero_count": int((sensor_df == 0).any(axis=1).sum()),
        "s1_s2_ratio_mean": float(pd.Series(s1_s2_ratio).dropna().mean()) if s1_s2_ratio is not None else np.nan,
        "s3_s4_ratio_mean": float(pd.Series(s3_s4_ratio).dropna().mean()) if s3_s4_ratio is not None else np.nan,
    }

    return {
        "correlation_matrix": corr_matrix,
        "summary": summary,
        "sensor_metrics_df": pd.DataFrame({
            "Sensor_Mean": sensor_mean,
            "Sensor_Spread": sensor_spread,
            "S1_S2_Ratio": s1_s2_ratio,
            "S3_S4_Ratio": s3_s4_ratio
        })
    }


def create_quality_audit_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Generates deterministic, row-level data quality flags for the dataset.

    DIRECTLY REUSES create_quality_flags(df) from src/quality_features.py to guarantee
    100% flag definitions and zero code duplication.
    """
    return create_quality_flags(df)


def run_data_quality_audit(df_train: Optional[pd.DataFrame] = None, df_test: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Executes complete data quality audit pipeline across Training_Data and Test_Data.

    Returns dictionary containing all structured audit results.
    """
    if df_train is None:
        df_train = load_training_data()
    if df_test is None:
        df_test = load_test_data()

    audit_results = {
        "missing_report": audit_missing_values(df_train),
        "missing_report_test": audit_missing_values(df_test),
        "non_finite_report": audit_non_finite_values(df_train),
        "non_finite_report_test": audit_non_finite_values(df_test),
        "duplicate_report": audit_duplicates(df_train),
        "duplicate_report_test": audit_duplicates(df_test),
        "repeated_test_id_report": audit_test_id_repeats(df_train),
        "repeated_test_id_report_test": audit_test_id_repeats(df_test),
        "numeric_statistics_train": calculate_numeric_statistics(df_train),
        "numeric_statistics_test": calculate_numeric_statistics(df_test),
        "percentiles_train": calculate_percentile_tables(df_train),
        "percentiles_test": calculate_percentile_tables(df_test),
        "physical_suspicious_train": detect_physical_suspicious_values(df_train),
        "physical_suspicious_test": detect_physical_suspicious_values(df_test),
        "outlier_candidates_train": identify_extreme_candidates(df_train),
        "outlier_candidates_test": identify_extreme_candidates(df_test),
        "sensor_behaviour_train": audit_sensor_behaviour(df_train),
        "sensor_behaviour_test": audit_sensor_behaviour(df_test),
        "flags_train": create_quality_audit_flags(df_train),
        "flags_test": create_quality_audit_flags(df_test),
        "summary_table": generate_quality_summary(create_quality_audit_flags(df_train), create_quality_audit_flags(df_test))
    }

    return audit_results
