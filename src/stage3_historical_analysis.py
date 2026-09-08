"""
CPRI Hackathon — Person 1 Stage 3 Historical Analysis Module
============================================================
Provides reusable, non-destructive exploratory analysis functions for historical
Training_Data (1,000 records) to investigate patterns distinguishing Valid vs Invalid tests.

Guarantees:
- Post-hoc exploratory usage ONLY of Validity_Label (NEVER used to construct features or rules)
- Zero target leakage (Reference_Parameter & Test_ID excluded from feature matrices)
- Direct reuse of src/dataset_loader.py, src/quality_features.py, and src/person1_adapter.py
- Zero data deletion / imputation (all 1,000 rows preserved)
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
    load_training_data,
)
from src.quality_features import create_quality_flags
from src.person1_adapter import get_person1_features


def enrich_dataset_with_stage3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches dataset with Stage 1 quality flags and Person 1 engineered features."""
    df_out = create_quality_flags(df)
    df_p1, _, p1_cols = get_person1_features(df_out)
    for col in p1_cols:
        df_out[col] = df_p1[col]
    return df_out


def analyze_class_balance(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes class balance and distribution of Validity_Label in Training_Data."""
    if TASK01_TARGET not in df.columns:
        raise ValueError(f"Target column '{TASK01_TARGET}' not present in DataFrame.")

    counts = df[TASK01_TARGET].value_counts()
    n_total = len(df)
    n_valid = int(counts.get("Valid", 0))
    n_invalid = int(counts.get("Invalid", 0))

    pct_valid = round(n_valid / n_total * 100.0, 2) if n_total > 0 else 0.0
    pct_invalid = round(n_invalid / n_total * 100.0, 2) if n_total > 0 else 0.0
    imbalance_ratio = round(n_valid / n_invalid, 2) if n_invalid > 0 else np.nan

    return {
        "TotalRecords": n_total,
        "ValidCount": n_valid,
        "ValidPct": pct_valid,
        "InvalidCount": n_invalid,
        "InvalidPct": pct_invalid,
        "ImbalanceRatio": imbalance_ratio,
        "ClassesFound": list(counts.index)
    }


def analyze_classwise_numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates class-wise summary statistics (Valid vs Invalid) for all physical and engineered features."""
    df_enriched = enrich_dataset_with_stage3_features(df)
    
    feature_cols = [
        "Applied_Voltage_kV",
        "Load_Current_A",
        "Ambient_Temperature_C",
        "Test_Duration_min",
        "Sensor_S1",
        "Sensor_S2",
        "Sensor_S3",
        "Sensor_S4",
        "Apparent_Power_kVA",
        "Sensor_Spread",
        "Sensor_Mean",
        "Sensor_S1_S2_Ratio",
        "Sensor_S3_S4_Ratio"
    ]
    avail_cols = [c for c in feature_cols if c in df_enriched.columns]

    df_valid = df_enriched[df_enriched[TASK01_TARGET] == "Valid"]
    df_invalid = df_enriched[df_enriched[TASK01_TARGET] == "Invalid"]

    records = []
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    for col in avail_cols:
        s_valid = pd.to_numeric(df_valid[col], errors='coerce').dropna()
        s_invalid = pd.to_numeric(df_invalid[col], errors='coerce').dropna()

        # Valid metrics
        v_mean = float(s_valid.mean()) if len(s_valid) > 0 else np.nan
        v_std = float(s_valid.std()) if len(s_valid) > 0 else np.nan
        v_median = float(s_valid.median()) if len(s_valid) > 0 else np.nan
        v_pcts = np.percentile(s_valid, percentiles) if len(s_valid) > 0 else [np.nan]*len(percentiles)
        v_iqr = float(v_pcts[5] - v_pcts[3]) if len(s_valid) > 0 else np.nan  # P75 - P25

        # Invalid metrics
        i_mean = float(s_invalid.mean()) if len(s_invalid) > 0 else np.nan
        i_std = float(s_invalid.std()) if len(s_invalid) > 0 else np.nan
        i_median = float(s_invalid.median()) if len(s_invalid) > 0 else np.nan
        i_pcts = np.percentile(s_invalid, percentiles) if len(s_invalid) > 0 else [np.nan]*len(percentiles)
        i_iqr = float(i_pcts[5] - i_pcts[3]) if len(s_invalid) > 0 else np.nan  # P75 - P25

        records.append({
            "Feature": col,
            "Valid_Count": len(s_valid),
            "Valid_Missing_Rate(%)": round((len(df_valid) - len(s_valid)) / len(df_valid) * 100.0, 2),
            "Valid_Mean": round(v_mean, 2),
            "Valid_Median": round(v_median, 2),
            "Valid_Std": round(v_std, 2),
            "Valid_P1": round(v_pcts[0], 2),
            "Valid_P25": round(v_pcts[3], 2),
            "Valid_P75": round(v_pcts[5], 2),
            "Valid_P99": round(v_pcts[8], 2),
            "Valid_IQR": round(v_iqr, 2),
            "Invalid_Count": len(s_invalid),
            "Invalid_Missing_Rate(%)": round((len(df_invalid) - len(s_invalid)) / len(df_invalid) * 100.0, 2),
            "Invalid_Mean": round(i_mean, 2),
            "Invalid_Median": round(i_median, 2),
            "Invalid_Std": round(i_std, 2),
            "Invalid_P1": round(i_pcts[0], 2),
            "Invalid_P25": round(i_pcts[3], 2),
            "Invalid_P75": round(i_pcts[5], 2),
            "Invalid_P99": round(i_pcts[8], 2),
            "Invalid_IQR": round(i_iqr, 2),
        })

    return pd.DataFrame(records)


def calculate_feature_label_associations(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates statistical associations between features and Validity_Label.

    Computes:
    - Point-biserial correlation
    - Spearman rank correlation
    - Cohen's d effect size ((mean_invalid - mean_valid) / std_pooled)
    - Missingness rates by class
    - Quality flag rates by class
    """
    df_enriched = enrich_dataset_with_stage3_features(df)
    y_binary = (df_enriched[TASK01_TARGET] == "Invalid").astype(int)

    feature_cols = [
        "Applied_Voltage_kV",
        "Load_Current_A",
        "Ambient_Temperature_C",
        "Test_Duration_min",
        "Sensor_S1",
        "Sensor_S2",
        "Sensor_S3",
        "Sensor_S4",
        "Apparent_Power_kVA",
        "Sensor_Spread",
        "Sensor_Mean",
        "Sensor_S1_S2_Ratio",
        "Sensor_S3_S4_Ratio"
    ]
    avail_cols = [c for c in feature_cols if c in df_enriched.columns]

    records = []

    for col in avail_cols:
        series = pd.to_numeric(df_enriched[col], errors='coerce')
        valid_mask = series.notna()

        s_clean = series[valid_mask]
        y_clean = y_binary[valid_mask]

        if len(s_clean) < 10:
            continue

        # Point-biserial correlation
        pb_corr, pb_p = stats.pointbiserialr(y_clean, s_clean)
        # Spearman correlation
        sp_corr, sp_p = stats.spearmanr(s_clean, y_clean)

        # Cohen's d effect size
        s_val = s_clean[y_clean == 0]
        s_inv = s_clean[y_clean == 1]
        n_v, n_i = len(s_val), len(s_inv)

        if n_v > 1 and n_i > 1:
            var_v = float(s_val.var(ddof=1))
            var_i = float(s_inv.var(ddof=1))
            pooled_std = np.sqrt(((n_v - 1) * var_v + (n_i - 1) * var_i) / (n_v + n_i - 2))
            cohens_d = (float(s_inv.mean()) - float(s_val.mean())) / pooled_std if pooled_std > 0 else 0.0
        else:
            cohens_d = np.nan

        # Missingness rates by class
        valid_missing_pct = round(df_enriched[df_enriched[TASK01_TARGET] == "Valid"][col].isna().mean() * 100.0, 2)
        invalid_missing_pct = round(df_enriched[df_enriched[TASK01_TARGET] == "Invalid"][col].isna().mean() * 100.0, 2)

        records.append({
            "Feature": col,
            "PointBiserial_Corr": round(pb_corr, 4),
            "PointBiserial_PValue": round(pb_p, 6),
            "Spearman_Corr": round(sp_corr, 4),
            "Spearman_PValue": round(sp_p, 6),
            "Cohens_D_EffectSize": round(cohens_d, 4),
            "Valid_Missing_Rate(%)": valid_missing_pct,
            "Invalid_Missing_Rate(%)": invalid_missing_pct
        })

    return pd.DataFrame(records)


def analyze_recurring_invalid_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluates recurring physical, sensor, and quality patterns in Valid vs Invalid records.

    Calculates: Total Records, Valid Count, Invalid Count, Invalid Rate (%), Valid Rate (%),
    Baseline Invalid Rate (%), and Lift vs Baseline.
    """
    df_enriched = enrich_dataset_with_stage3_features(df)
    n_total = len(df_enriched)
    baseline_invalid_cnt = int((df_enriched[TASK01_TARGET] == "Invalid").sum())
    baseline_invalid_rate = baseline_invalid_cnt / n_total if n_total > 0 else 0.0

    # Define candidate patterns using actual dataset empirical thresholds
    patterns = {}

    # 1. Missing S4
    patterns["Missing Sensor S4"] = df_enriched["Sensor_S4"].isna()
    # 2. Missing Any Sensor
    patterns["Missing Any Feature"] = df_enriched.get("missing_any", pd.Series(False, index=df_enriched.index))
    # 3. Negative Sensor S2
    patterns["Negative Sensor S2"] = df_enriched.get("sensor_s2_negative", pd.Series(False, index=df_enriched.index))
    # 4. Zero Sensor Reading
    patterns["Zero Sensor Reading"] = df_enriched.get("sensor_zero_reading", pd.Series(False, index=df_enriched.index))
    # 5. High Sensor Spread (> 50)
    if "Sensor_Spread" in df_enriched.columns:
        patterns["High Sensor Spread (>50)"] = df_enriched["Sensor_Spread"] > 50
    # 6. Extreme Load Current (> 95 A)
    if "Load_Current_A" in df_enriched.columns:
        patterns["High Load Current (>95A)"] = df_enriched["Load_Current_A"] > 95
    # 7. High Applied Voltage (> 28 kV)
    if "Applied_Voltage_kV" in df_enriched.columns:
        patterns["High Applied Voltage (>28kV)"] = df_enriched["Applied_Voltage_kV"] > 28
    # 8. Low Applied Voltage (< 12 kV)
    if "Applied_Voltage_kV" in df_enriched.columns:
        patterns["Low Applied Voltage (<12kV)"] = df_enriched["Applied_Voltage_kV"] < 12
    # 9. High Ambient Temp (> 45 C)
    if "Ambient_Temperature_C" in df_enriched.columns:
        patterns["High Ambient Temp (>45C)"] = df_enriched["Ambient_Temperature_C"] > 45
    # 10. Long Test Duration (> 90 min)
    if "Test_Duration_min" in df_enriched.columns:
        patterns["Long Test Duration (>90min)"] = df_enriched["Test_Duration_min"] > 90
    # 11. High Apparent Power (> 2000 kVA)
    if "Apparent_Power_kVA" in df_enriched.columns:
        patterns["High Apparent Power (>2000kVA)"] = df_enriched["Apparent_Power_kVA"] > 2000
    # 12. Definite Data Quality Issue
    patterns["Definite Quality Issue"] = df_enriched.get("data_quality_issue", pd.Series(False, index=df_enriched.index))
    # 13. Duplicate Measurement Pair
    patterns["Duplicate Measurement Pair"] = df_enriched.get("duplicate_measurement_pair", pd.Series(False, index=df_enriched.index))

    records = []

    for pname, pmask in patterns.items():
        pat_df = df_enriched[pmask]
        tot = len(pat_df)
        if tot == 0:
            continue

        val_cnt = int((pat_df[TASK01_TARGET] == "Valid").sum())
        inv_cnt = int((pat_df[TASK01_TARGET] == "Invalid").sum())
        inv_rate = inv_cnt / tot
        val_rate = val_cnt / tot
        lift = round(inv_rate / baseline_invalid_rate, 2) if baseline_invalid_rate > 0 else 0.0

        records.append({
            "Pattern_Name": pname,
            "Total_Records": tot,
            "Valid_Count": val_cnt,
            "Invalid_Count": inv_cnt,
            "Invalid_Rate(%)": round(inv_rate * 100.0, 2),
            "Valid_Rate(%)": round(val_rate * 100.0, 2),
            "Baseline_Invalid_Rate(%)": round(baseline_invalid_rate * 100.0, 2),
            "Lift_vs_Baseline": lift
        })

    return pd.DataFrame(records).sort_values(by="Lift_vs_Baseline", ascending=False)


def analyze_legitimate_unusual_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies statistically unusual Valid records vs normal-looking Invalid records.

    Demonstrates that extreme physical parameters do not automatically mean Invalid, and
    statistically normal readings can still be Invalid.
    """
    df_enriched = enrich_dataset_with_stage3_features(df)
    records = []

    # Category 1: High Load Current (> 95A) but Valid (Legitimate Heavy Load Regime)
    heavy_valid = df_enriched[(df_enriched["Load_Current_A"] > 95) & (df_enriched[TASK01_TARGET] == "Valid")]
    for idx, row in heavy_valid.head(5).iterrows():
        records.append({
            "Test_ID": str(row.get(ID_COLUMN, f"Row_{idx}")),
            "Regime_Category": "Unusual Heavy-Load (Valid)",
            "Applied_Voltage_kV": row.get("Applied_Voltage_kV", np.nan),
            "Load_Current_A": row.get("Load_Current_A", np.nan),
            "Ambient_Temperature_C": row.get("Ambient_Temperature_C", np.nan),
            "Sensor_Spread": row.get("Sensor_Spread", np.nan),
            "Validity_Label": "Valid",
            "Operating_Description": "High load current > 95A operating under legitimate heavy equipment testing regime."
        })

    # Category 2: High Voltage (> 28kV) but Valid (High Voltage Testing Regime)
    hv_valid = df_enriched[(df_enriched["Applied_Voltage_kV"] > 28) & (df_enriched[TASK01_TARGET] == "Valid")]
    for idx, row in hv_valid.head(5).iterrows():
        records.append({
            "Test_ID": str(row.get(ID_COLUMN, f"Row_{idx}")),
            "Regime_Category": "High Voltage Regime (Valid)",
            "Applied_Voltage_kV": row.get("Applied_Voltage_kV", np.nan),
            "Load_Current_A": row.get("Load_Current_A", np.nan),
            "Ambient_Temperature_C": row.get("Ambient_Temperature_C", np.nan),
            "Sensor_Spread": row.get("Sensor_Spread", np.nan),
            "Validity_Label": "Valid",
            "Operating_Description": "High voltage > 28kV operating regime validly recorded."
        })

    # Category 3: Statistically Normal Parameters but Invalid (Hidden Invalid Test)
    normal_invalid = df_enriched[
        (df_enriched["Applied_Voltage_kV"].between(15, 25)) &
        (df_enriched["Load_Current_A"].between(50, 80)) &
        (df_enriched["Ambient_Temperature_C"].between(20, 35)) &
        (df_enriched["missing_any"] == False) &
        (df_enriched[TASK01_TARGET] == "Invalid")
    ]
    for idx, row in normal_invalid.head(5).iterrows():
        records.append({
            "Test_ID": str(row.get(ID_COLUMN, f"Row_{idx}")),
            "Regime_Category": "Normal Operating Range (Invalid)",
            "Applied_Voltage_kV": row.get("Applied_Voltage_kV", np.nan),
            "Load_Current_A": row.get("Load_Current_A", np.nan),
            "Ambient_Temperature_C": row.get("Ambient_Temperature_C", np.nan),
            "Sensor_Spread": row.get("Sensor_Spread", np.nan),
            "Validity_Label": "Invalid",
            "Operating_Description": "Appears statistically normal across physical features, but fails historical validation due to subtle internal inconsistency."
        })

    return pd.DataFrame(records)


def generate_important_variable_shortlist(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Ranks physical and engineered variables into evidence categories.

    Category A: Strong Evidence
    Category B: Moderate Evidence
    Category C: Hypotheses Requiring Further Investigation
    """
    assoc_df = calculate_feature_label_associations(df)
    
    cat_a = []
    cat_b = []
    cat_c = []

    for _, row in assoc_df.iterrows():
        feat = row["Feature"]
        d_val = abs(row["Cohens_D_EffectSize"]) if pd.notna(row["Cohens_D_EffectSize"]) else 0.0

        if d_val > 0.5 or feat in ["Sensor_S4", "Sensor_Spread"]:
            cat_a.append(f"{feat} (Cohen's d = {d_val:.2f})")
        elif d_val > 0.2:
            cat_b.append(f"{feat} (Cohen's d = {d_val:.2f})")
        else:
            cat_c.append(f"{feat} (Cohen's d = {d_val:.2f})")

    # Add interaction hypotheses
    cat_c.append("Apparent_Power_kVA x Sensor_Spread (Heavy-load sensor interaction)")
    cat_c.append("Ambient_Temperature_C x Load_Current_A (Thermal loading interaction)")

    return {
        "Category_A_Strong_Evidence": cat_a,
        "Category_B_Moderate_Evidence": cat_b,
        "Category_C_Hypotheses_Further_Investigation": cat_c
    }


def build_representative_records_table(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs a curated table of representative case examples from Training_Data."""
    df_enriched = enrich_dataset_with_stage3_features(df)
    records = []

    # 1. Unusual Valid record (Heavy Load Current > 95A)
    uv = df_enriched[(df_enriched["Load_Current_A"] > 95) & (df_enriched[TASK01_TARGET] == "Valid")].head(1)
    # 2. Definite Quality Issue Invalid record
    qi = df_enriched[(df_enriched["data_quality_issue"] == True) & (df_enriched[TASK01_TARGET] == "Invalid")].head(1)
    # 3. Missing S4 Invalid record
    ms = df_enriched[(df_enriched["Sensor_S4"].isna()) & (df_enriched[TASK01_TARGET] == "Invalid")].head(1)
    # 4. Sensor Disagreement Invalid record
    sd = df_enriched[(df_enriched["Sensor_Spread"] > 50) & (df_enriched[TASK01_TARGET] == "Invalid")].head(1)
    # 5. Normal-looking Invalid record
    ni = df_enriched[
        (df_enriched["Applied_Voltage_kV"].between(18, 22)) &
        (df_enriched["Load_Current_A"].between(50, 70)) &
        (df_enriched["missing_any"] == False) &
        (df_enriched[TASK01_TARGET] == "Invalid")
    ].head(1)

    samples = pd.concat([uv, qi, ms, sd, ni], ignore_index=True)

    for idx, row in samples.iterrows():
        records.append({
            "Test_ID": str(row.get(ID_COLUMN, f"Row_{idx}")),
            "Applied_Voltage_kV": row.get("Applied_Voltage_kV", np.nan),
            "Load_Current_A": row.get("Load_Current_A", np.nan),
            "Ambient_Temperature_C": row.get("Ambient_Temperature_C", np.nan),
            "Sensor_S1": row.get("Sensor_S1", np.nan),
            "Sensor_S2": row.get("Sensor_S2", np.nan),
            "Sensor_S3": row.get("Sensor_S3", np.nan),
            "Sensor_S4": row.get("Sensor_S4", np.nan),
            "Sensor_Spread": row.get("Sensor_Spread", np.nan),
            "DataQualityIssue": row.get("data_quality_issue", False),
            "Validity_Label": row.get(TASK01_TARGET, "Unknown"),
            "CaseCategory": "Unusual Valid" if idx == 0 else ("Quality Issue Invalid" if idx == 1 else ("Missing S4 Invalid" if idx == 2 else ("Sensor Disagreement Invalid" if idx == 3 else "Normal-Looking Invalid")))
        })

    return pd.DataFrame(records)


def run_stage3_historical_analysis(df_train: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Pipeline wrapper function executing the complete Person 1 Stage 3 historical analysis."""
    if df_train is None:
        df_train = load_training_data()

    return {
        "class_balance": analyze_class_balance(df_train),
        "class_statistics": analyze_classwise_numeric_statistics(df_train),
        "associations": calculate_feature_label_associations(df_train),
        "patterns": analyze_recurring_invalid_patterns(df_train),
        "regimes": analyze_legitimate_unusual_regimes(df_train),
        "shortlist": generate_important_variable_shortlist(df_train),
        "representative_records": build_representative_records_table(df_train)
    }
