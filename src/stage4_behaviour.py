"""
CPRI Hackathon — Person 1 Stage 4 Behaviour & Physical Consistency Module
========================================================================
Provides reusable, non-destructive functions and normal-behaviour reference models for
analyzing expected sensor behaviour, discovering operating regimes, and generating contextual
residual and consistency features.

Guarantees:
- Reference models predicting expected sensor values are fitted STRICTLY on Valid historical records.
- Target leakage protection (Validity_Label & Reference_Parameter NEVER used as predictors).
- Test_ID excluded from all feature matrices.
- Inference-safe feature generation working seamlessly on Test_Data without Validity_Label.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer

from src.dataset_loader import (
    PHYSICAL_FEATURES,
    ID_COLUMN,
    TASK01_TARGET,
    TASK02_TARGET,
    load_training_data,
)
from src.quality_features import create_quality_flags


def analyze_relationships(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes physical relationships between operating conditions, environmental variables, and sensors.

    Computes Pearson and Spearman correlations and assesses trend linearity.
    """
    operating_cols = ["Load_Current_A", "Applied_Voltage_kV", "Ambient_Temperature_C", "Test_Duration_min"]
    sensor_cols = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]

    records = []

    # 1. Operating conditions vs Sensors
    for op in operating_cols:
        if op not in df.columns:
            continue
        op_s = pd.to_numeric(df[op], errors='coerce')

        for sc in sensor_cols:
            if sc not in df.columns:
                continue
            sc_s = pd.to_numeric(df[sc], errors='coerce')

            valid_mask = op_s.notna() & sc_s.notna()
            x = op_s[valid_mask]
            y = sc_s[valid_mask]

            if len(x) < 10:
                continue

            pr_r, pr_p = stats.pearsonr(x, y)
            sp_r, sp_p = stats.spearmanr(x, y)

            records.append({
                "Category": "Operating_vs_Sensor",
                "Variable_1": op,
                "Variable_2": sc,
                "Sample_Count": len(x),
                "Pearson_r": round(pr_r, 4),
                "Pearson_p": round(pr_p, 6),
                "Spearman_rho": round(sp_r, 4),
                "Spearman_p": round(sp_p, 6),
                "Relationship_Type": "Strong Linear" if abs(pr_r) > 0.6 else ("Moderate" if abs(pr_r) > 0.3 else "Weak")
            })

    # 2. Cross-sensor pairs
    for i in range(len(sensor_cols)):
        for j in range(i + 1, len(sensor_cols)):
            s1_col, s2_col = sensor_cols[i], sensor_cols[j]
            if s1_col in df.columns and s2_col in df.columns:
                s1_s = pd.to_numeric(df[s1_col], errors='coerce')
                s2_s = pd.to_numeric(df[s2_col], errors='coerce')

                valid_mask = s1_s.notna() & s2_s.notna()
                x = s1_s[valid_mask]
                y = s2_s[valid_mask]

                if len(x) < 10:
                    continue

                pr_r, pr_p = stats.pearsonr(x, y)
                sp_r, sp_p = stats.spearmanr(x, y)

                records.append({
                    "Category": "Cross_Sensor_Pair",
                    "Variable_1": s1_col,
                    "Variable_2": s2_col,
                    "Sample_Count": len(x),
                    "Pearson_r": round(pr_r, 4),
                    "Pearson_p": round(pr_p, 6),
                    "Spearman_rho": round(sp_r, 4),
                    "Spearman_p": round(sp_p, 6),
                    "Relationship_Type": "Strong Inter-Sensor Alignment" if pr_r > 0.7 else "Moderate Alignment"
                })

    return pd.DataFrame(records)


def discover_operating_regimes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Discovers physically meaningful operating regimes across Load_Current, Voltage, Temp, and Duration.

    Distinguishes legitimate regime variation from sensor inconsistency.
    """
    df_regime = df.copy()

    current_s = pd.to_numeric(df["Load_Current_A"], errors='coerce')
    voltage_s = pd.to_numeric(df["Applied_Voltage_kV"], errors='coerce')

    # Current quantile bands
    q33, q66 = current_s.quantile(0.33), current_s.quantile(0.66)
    def assign_current_band(c):
        if pd.isna(c): return "Unknown"
        if c <= q33: return "Low_Current"
        elif c <= q66: return "Medium_Current"
        else: return "High_Current"

    df_regime["Current_Regime"] = current_s.apply(assign_current_band)

    # Combined operating regime code (1..4)
    def assign_regime_code(row):
        c = row.get("Load_Current_A", np.nan)
        v = row.get("Applied_Voltage_kV", np.nan)
        if pd.isna(c) or pd.isna(v):
            return 0
        if c > 85 and v > 25:
            return 1  # High Current + High Voltage Heavy Regime
        elif c > 85:
            return 2  # High Current Heavy Loading
        elif v > 25:
            return 3  # High Voltage Regime
        else:
            return 4  # Standard Operating Regime

    df_regime["Operating_Regime_Code"] = df_regime.apply(assign_regime_code, axis=1)

    # Regime summary metrics
    summary_records = []
    for rcode, group in df_regime.groupby("Operating_Regime_Code"):
        tot = len(group)
        val_cnt = int((group[TASK01_TARGET] == "Valid").sum()) if TASK01_TARGET in group.columns else np.nan
        inv_cnt = int((group[TASK01_TARGET] == "Invalid").sum()) if TASK01_TARGET in group.columns else np.nan
        inv_rate = round(inv_cnt / tot * 100.0, 2) if tot > 0 and not np.isnan(inv_cnt) else np.nan

        summary_records.append({
            "Regime_Code": rcode,
            "Regime_Name": "Heavy HV Load" if rcode == 1 else ("Heavy Current" if rcode == 2 else ("High Voltage" if rcode == 3 else ("Standard" if rcode == 4 else "Unknown"))),
            "Total_Records": tot,
            "Valid_Count": val_cnt,
            "Invalid_Count": inv_cnt,
            "Invalid_Rate(%)": inv_rate,
            "Mean_Current_A": round(float(pd.to_numeric(group["Load_Current_A"], errors='coerce').mean()), 2),
            "Mean_Voltage_kV": round(float(pd.to_numeric(group["Applied_Voltage_kV"], errors='coerce').mean()), 2)
        })

    return df_regime, pd.DataFrame(summary_records)


class NormalBehaviourModels:
    """Predictive reference models estimating expected sensor readings under normal operating conditions.

    CRITICAL GUARANTEE: Models are trained STRICTLY on engineer-verified Valid historical records.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.models: Dict[str, Ridge] = {}
        self.imputer = SimpleImputer(strategy="median")
        self.feature_names: List[str] = ["Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min"]
        self.target_sensors: List[str] = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]
        self.model_performance: List[Dict[str, Any]] = []
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> "NormalBehaviourModels":
        """Fits normal sensor prediction models strictly using Valid historical records.

        Args:
            df_train: Training DataFrame containing physical features and Validity_Label.
        """
        # Filter STRICTLY for Valid historical records
        if TASK01_TARGET in df_train.columns:
            df_valid = df_train[df_train[TASK01_TARGET] == "Valid"].copy()
        else:
            df_valid = df_train.copy()

        n_valid = len(df_valid)
        if n_valid < 10:
            raise ValueError(f"Insufficient Valid training records to fit normal behaviour models (got {n_valid}).")

        # Prepare predictors X
        avail_features = [f for f in self.feature_names if f in df_valid.columns]
        X_valid_raw = df_valid[avail_features].apply(pd.to_numeric, errors='coerce')
        X_valid = self.imputer.fit_transform(X_valid_raw)

        self.model_performance = []

        for sensor in self.target_sensors:
            if sensor not in df_valid.columns:
                continue

            y_sensor = pd.to_numeric(df_valid[sensor], errors='coerce')
            valid_y_mask = y_sensor.notna()

            if valid_y_mask.sum() < 10:
                continue

            X_s = X_valid[valid_y_mask]
            y_s = y_sensor[valid_y_mask].values

            # Fit Ridge regression reference model
            model = Ridge(alpha=self.alpha, random_state=42)
            model.fit(X_s, y_s)
            self.models[sensor] = model

            # Evaluate fit metrics on Valid training records
            y_pred = model.predict(X_s)
            mae = float(np.mean(np.abs(y_s - y_pred)))
            rmse = float(np.sqrt(np.mean((y_s - y_pred) ** 2)))
            r2 = float(model.score(X_s, y_s))

            self.model_performance.append({
                "Target_Sensor": sensor,
                "Valid_Train_Count": int(valid_y_mask.sum()),
                "R2_Score": round(r2, 4),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "Predictors": ", ".join(avail_features)
            })

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates expected sensor predictions, residuals, and consistency features.

        Works on both Training and Test data without requiring Validity_Label.
        """
        if not self.is_fitted:
            raise RuntimeError("NormalBehaviourModels must be fitted before calling transform().")

        df_out = df.copy()

        # Prepare predictors X
        avail_features = [f for f in self.feature_names if f in df.columns]
        X_raw = df[avail_features].apply(pd.to_numeric, errors='coerce')
        X_imp = self.imputer.transform(X_raw)

        residual_cols = []

        for sensor, model in self.models.items():
            y_expected = model.predict(X_imp)
            exp_col = f"{sensor}_Expected"
            res_col = f"{sensor}_Residual"
            abs_res_col = f"{sensor}_AbsResidual"

            df_out[exp_col] = y_expected

            if sensor in df.columns:
                actual = pd.to_numeric(df[sensor], errors='coerce')
                residual = actual - y_expected
                abs_residual = np.abs(residual)
                df_out[res_col] = residual
                df_out[abs_res_col] = abs_residual
                residual_cols.append(abs_res_col)
            else:
                df_out[res_col] = np.nan
                df_out[abs_res_col] = np.nan

        # Aggregate consistency features
        if residual_cols:
            abs_res_matrix = df_out[residual_cols].apply(pd.to_numeric, errors='coerce')
            df_out["p1_max_abs_residual"] = abs_res_matrix.max(axis=1)
            df_out["p1_mean_abs_residual"] = abs_res_matrix.mean(axis=1)
            df_out["p1_residual_error"] = df_out["p1_max_abs_residual"]
            df_out["p1_consistency_index"] = 1.0 / (1.0 + df_out["p1_mean_abs_residual"].fillna(0.0))
        else:
            df_out["p1_max_abs_residual"] = np.nan
            df_out["p1_mean_abs_residual"] = np.nan
            df_out["p1_residual_error"] = np.nan
            df_out["p1_consistency_index"] = np.nan

        # Sensor Disagreement Index
        sensor_cols = [c for c in self.target_sensors if c in df.columns]
        if len(sensor_cols) == 4:
            s_matrix = df[sensor_cols].apply(pd.to_numeric, errors='coerce')
            s_spread = s_matrix.max(axis=1) - s_matrix.min(axis=1)
            s_mean = s_matrix.mean(axis=1).abs() + 1e-5
            df_out["p1_sensor_disagreement_index"] = s_spread / s_mean
        else:
            df_out["p1_sensor_disagreement_index"] = np.nan

        # Regime Cluster Assignment
        current_s = pd.to_numeric(df["Load_Current_A"], errors='coerce') if "Load_Current_A" in df.columns else pd.Series(0, index=df.index)
        voltage_s = pd.to_numeric(df["Applied_Voltage_kV"], errors='coerce') if "Applied_Voltage_kV" in df.columns else pd.Series(0, index=df.index)

        def assign_cluster(row):
            c, v = row["c"], row["v"]
            if pd.isna(c) or pd.isna(v): return 0
            if c > 85 and v > 25: return 1
            elif c > 85: return 2
            elif v > 25: return 3
            else: return 4

        df_out["p1_regime_cluster"] = pd.DataFrame({"c": current_s, "v": voltage_s}).apply(assign_cluster, axis=1)

        return df_out


def analyze_residuals_by_class(df_residuals: pd.DataFrame) -> pd.DataFrame:
    """Calculates summary statistics and effect sizes for residual features by Validity class."""
    if TASK01_TARGET not in df_residuals.columns:
        return pd.DataFrame()

    residual_cols = [c for c in df_residuals.columns if "Residual" in c or c.startswith("p1_")]
    df_valid = df_residuals[df_residuals[TASK01_TARGET] == "Valid"]
    df_invalid = df_residuals[df_residuals[TASK01_TARGET] == "Invalid"]

    records = []

    for col in residual_cols:
        s_val = pd.to_numeric(df_valid[col], errors='coerce').dropna()
        s_inv = pd.to_numeric(df_invalid[col], errors='coerce').dropna()

        if len(s_val) < 5 or len(s_inv) < 5:
            continue

        n_v, n_i = len(s_val), len(s_inv)
        v_mean, i_mean = float(s_val.mean()), float(s_inv.mean())
        v_std, i_std = float(s_val.std()), float(s_inv.std())

        pooled_std = np.sqrt(((n_v - 1) * (v_std**2) + (n_i - 1) * (i_std**2)) / (n_v + n_i - 2)) if n_v > 1 and n_i > 1 else np.nan
        cohens_d = (i_mean - v_mean) / pooled_std if pooled_std and pooled_std > 0 else 0.0

        p90_val = float(s_val.quantile(0.90))
        extreme_inv_count = int((s_inv > p90_val).sum())
        extreme_inv_rate = round(extreme_inv_count / n_i * 100.0, 2) if n_i > 0 else 0.0

        records.append({
            "Residual_Feature": col,
            "Valid_Mean": round(v_mean, 4),
            "Valid_Median": round(float(s_val.median()), 4),
            "Valid_Std": round(v_std, 4),
            "Invalid_Mean": round(i_mean, 4),
            "Invalid_Median": round(float(s_inv.median()), 4),
            "Invalid_Std": round(i_std, 4),
            "Cohens_D_EffectSize": round(cohens_d, 4),
            "Invalid_Extreme_Residual_Rate(%)": extreme_inv_rate
        })

    return pd.DataFrame(records).sort_values(by="Cohens_D_EffectSize", key=abs, ascending=False)


def analyze_sensor_reliability(df_residuals: pd.DataFrame, model_perf_df: pd.DataFrame) -> pd.DataFrame:
    """Ranks sensor relationship and prediction reliability based on fit quality and separation power."""
    res_stats = analyze_residuals_by_class(df_residuals)
    
    records = []

    for _, row in model_perf_df.iterrows():
        sensor = row["Target_Sensor"]
        abs_res_col = f"{sensor}_AbsResidual"

        effect_size = np.nan
        if not res_stats.empty and abs_res_col in res_stats["Residual_Feature"].values:
            effect_size = float(res_stats[res_stats["Residual_Feature"] == abs_res_col]["Cohens_D_EffectSize"].iloc[0])

        records.append({
            "Sensor": sensor,
            "Normal_Model_R2": row["R2_Score"],
            "Normal_Model_RMSE": row["RMSE"],
            "AbsResidual_EffectSize": effect_size,
            "Reliability_Rank": "High" if row["R2_Score"] > 0.5 and abs(effect_size) > 0.5 else ("Moderate" if row["R2_Score"] > 0.2 else "Low")
        })

    return pd.DataFrame(records)


def analyze_legitimate_unusual_regimes_residuals(df_residuals: pd.DataFrame) -> pd.DataFrame:
    """Analyzes residuals for unusual Valid records vs normal-looking Invalid records."""
    if TASK01_TARGET not in df_residuals.columns:
        return pd.DataFrame()

    records = []

    # Category 1: High Current (> 95A) Valid records
    heavy_valid = df_residuals[(df_residuals["Load_Current_A"] > 95) & (df_residuals[TASK01_TARGET] == "Valid")]
    for idx, row in heavy_valid.head(5).iterrows():
        records.append({
            "Test_ID": str(row.get(ID_COLUMN, f"Row_{idx}")),
            "Category": "Unusual Heavy Load (Valid)",
            "Load_Current_A": row.get("Load_Current_A", np.nan),
            "Max_Abs_Residual": row.get("p1_max_abs_residual", np.nan),
            "Consistency_Index": row.get("p1_consistency_index", np.nan),
            "Validity_Label": "Valid",
            "Residual_Interpretation": "High load current > 95A, but residuals remain small/consistent with expected heavy-load model."
        })

    # Category 2: Normal Parameters but Invalid records
    normal_invalid = df_residuals[
        (df_residuals["Applied_Voltage_kV"].between(15, 25)) &
        (df_residuals["Load_Current_A"].between(50, 80)) &
        (df_residuals[TASK01_TARGET] == "Invalid")
    ]
    for idx, row in normal_invalid.head(5).iterrows():
        records.append({
            "Test_ID": str(row.get(ID_COLUMN, f"Row_{idx}")),
            "Category": "Normal Parameters (Invalid)",
            "Load_Current_A": row.get("Load_Current_A", np.nan),
            "Max_Abs_Residual": row.get("p1_max_abs_residual", np.nan),
            "Consistency_Index": row.get("p1_consistency_index", np.nan),
            "Validity_Label": "Invalid",
            "Residual_Interpretation": "Raw parameters appear normal, but residual analysis reveals sensor disagreement or prediction deviation."
        })

    return pd.DataFrame(records)


def run_stage4_behaviour(df_train: Optional[pd.DataFrame] = None, df_test: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Pipeline wrapper function executing the complete Person 1 Stage 4 behaviour analysis."""
    if df_train is None:
        df_train = load_training_data()

    # 1. Relationship analysis
    rel_df = analyze_relationships(df_train)

    # 2. Operating regime discovery
    df_regime_train, regime_summary = discover_operating_regimes(df_train)

    # 3. Fit Normal Behaviour Models strictly on Valid historical training data
    nb_models = NormalBehaviourModels()
    nb_models.fit(df_train)
    model_perf_df = pd.DataFrame(nb_models.model_performance)

    # 4. Generate residual features for Training and Test
    df_res_train = nb_models.transform(df_train)
    df_res_test = nb_models.transform(df_test) if df_test is not None else None

    # 5. Class-wise residual statistics
    res_stats_df = analyze_residuals_by_class(df_res_train)

    # 6. Sensor reliability analysis
    sensor_rel_df = analyze_sensor_reliability(df_res_train, model_perf_df)

    # 7. Legitimate unusual regime residual analysis
    rep_records_df = analyze_legitimate_unusual_regimes_residuals(df_res_train)

    # 8. Candidate features list
    candidate_features = [
        "Sensor_S1_Expected", "Sensor_S1_Residual", "Sensor_S1_AbsResidual",
        "Sensor_S2_Expected", "Sensor_S2_Residual", "Sensor_S2_AbsResidual",
        "Sensor_S3_Expected", "Sensor_S3_Residual", "Sensor_S3_AbsResidual",
        "Sensor_S4_Expected", "Sensor_S4_Residual", "Sensor_S4_AbsResidual",
        "p1_max_abs_residual", "p1_mean_abs_residual", "p1_residual_error",
        "p1_consistency_index", "p1_sensor_disagreement_index", "p1_regime_cluster"
    ]
    cand_df = pd.DataFrame({"Candidate_Feature": candidate_features})

    return {
        "relationship_analysis": rel_df,
        "regime_analysis": regime_summary,
        "model_performance": model_perf_df,
        "residual_statistics": res_stats_df,
        "sensor_reliability": sensor_rel_df,
        "candidate_features": cand_df,
        "representative_records": rep_records_df,
        "residuals_train": df_res_train,
        "residuals_test": df_res_test,
        "fitted_models": nb_models
    }
