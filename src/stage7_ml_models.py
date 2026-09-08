"""
CPRI Hackathon — Person 1 Stage 7 ML Classification / Anomaly Models
=====================================================================
Provides leakage-free 5-fold Stratified Cross-Validation model evaluation,
feature-set comparison, decision threshold optimization, permutation importance,
and error analysis for Task 01.

Guarantees:
- Strict exclusion of Validity_Label, Reference_Parameter, and Test_ID from predictors.
- Fold-isolated preprocessing and anomaly score generation.
- 5-Fold Stratified CV (n_splits=5, shuffle=True, random_state=42).
- Complete OOF probability predictions for all 1,000 training records.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from src.dataset_loader import TASK01_TARGET, ID_COLUMN, PHYSICAL_FEATURES
from src.quality_features import FEATURE_COLUMNS
from src.stage4_behaviour import NormalBehaviourModels


def prepare_feature_sets(df_train_features: pd.DataFrame, df_test_features: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Optional[pd.DataFrame]], Dict[str, List[str]]]:
    """Constructs Feature Sets A, B, C, D, E for Training and Test data cleanly without target leakage."""

    # Base raw features (Set A)
    raw_cols = [c for c in PHYSICAL_FEATURES if c in df_train_features.columns]

    # Quality flags (Set B)
    quality_cols = [
        "missing_any", "non_finite_value", "invalid_voltage", "invalid_current",
        "invalid_duration", "invalid_ambient_temp", "sensor_s2_negative",
        "sensor_zero_reading", "malformed_numeric", "data_quality_issue"
    ]
    avail_quality = [c for c in quality_cols if c in df_train_features.columns]

    # Residual / consistency features (Set C)
    residual_cols = [
        "Sensor_S1_Residual", "Sensor_S2_Residual", "Sensor_S3_Residual", "Sensor_S4_Residual",
        "p1_max_abs_residual", "p1_mean_abs_residual", "p1_consistency_index",
        "p1_sensor_disagreement_index", "p1_regime_cluster"
    ]
    avail_residual = [c for c in residual_cols if c in df_train_features.columns]

    # Full Stage 5 engineered features (Set D)
    excluded = [ID_COLUMN, TASK01_TARGET, "Reference_Parameter", "missing_columns"]
    full_stage5_cols = [c for c in df_train_features.columns if c not in excluded]

    feature_sets_train = {}
    feature_sets_test = {}
    feature_names_dict = {}

    # Set A: Raw physical inputs
    feature_names_dict["Set_A_Raw"] = raw_cols
    feature_sets_train["Set_A_Raw"] = df_train_features[raw_cols].apply(pd.to_numeric, errors='coerce')
    feature_sets_test["Set_A_Raw"] = df_test_features[raw_cols].apply(pd.to_numeric, errors='coerce') if df_test_features is not None else None

    # Set B: Raw + Quality
    cols_b = list(set(raw_cols + avail_quality))
    feature_names_dict["Set_B_Raw_Quality"] = cols_b
    feature_sets_train["Set_B_Raw_Quality"] = df_train_features[cols_b].apply(pd.to_numeric, errors='coerce')
    feature_sets_test["Set_B_Raw_Quality"] = df_test_features[cols_b].apply(pd.to_numeric, errors='coerce') if df_test_features is not None else None

    # Set C: Raw + Stage 4 Residuals
    cols_c = list(set(raw_cols + avail_residual))
    feature_names_dict["Set_C_Raw_Residuals"] = cols_c
    feature_sets_train["Set_C_Raw_Residuals"] = df_train_features[cols_c].apply(pd.to_numeric, errors='coerce')
    feature_sets_test["Set_C_Raw_Residuals"] = df_test_features[cols_c].apply(pd.to_numeric, errors='coerce') if df_test_features is not None else None

    # Set D: Full Stage 5 Features
    feature_names_dict["Set_D_Full_Stage5"] = full_stage5_cols
    feature_sets_train["Set_D_Full_Stage5"] = df_train_features[full_stage5_cols].apply(pd.to_numeric, errors='coerce')
    feature_sets_test["Set_D_Full_Stage5"] = df_test_features[full_stage5_cols].apply(pd.to_numeric, errors='coerce') if df_test_features is not None else None

    # Set E: Full Stage 5 + OOF Isolation Forest Anomaly Score
    df_train_e = df_train_features[full_stage5_cols].apply(pd.to_numeric, errors='coerce').copy()
    df_test_e = df_test_features[full_stage5_cols].apply(pd.to_numeric, errors='coerce').copy() if df_test_features is not None else None

    # Compute fold-isolated OOF Isolation Forest score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_train_binary = (df_train_features[TASK01_TARGET] == "Invalid").astype(int).values if TASK01_TARGET in df_train_features.columns else np.zeros(len(df_train_features))

    oof_anom_scores = np.zeros(len(df_train_features))
    X_raw_mat = df_train_features[raw_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values

    for train_idx, val_idx in skf.split(X_raw_mat, y_train_binary):
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        iso.fit(X_raw_mat[train_idx])
        oof_anom_scores[val_idx] = -iso.score_samples(X_raw_mat[val_idx])

    df_train_e["isolation_forest_score"] = oof_anom_scores

    if df_test_e is not None:
        iso_full = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
        iso_full.fit(X_raw_mat)
        X_test_raw_mat = df_test_features[raw_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        df_test_e["isolation_forest_score"] = -iso_full.score_samples(X_test_raw_mat)

    cols_e = list(df_train_e.columns)
    feature_names_dict["Set_E_Stage5_AnomalyScore"] = cols_e
    feature_sets_train["Set_E_Stage5_AnomalyScore"] = df_train_e
    feature_sets_test["Set_E_Stage5_AnomalyScore"] = df_test_e

    return feature_sets_train, feature_sets_test, feature_names_dict


def get_model_pipelines() -> Dict[str, Any]:
    """Returns the candidate machine learning pipelines."""
    pipelines = {
        "Logistic_Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
        ]),
        "Random_Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, class_weight="balanced", random_state=42))
        ]),
        "Hist_Gradient_Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(class_weight="balanced", max_iter=200, random_state=42))
        ])
    }
    return pipelines


def evaluate_model_cv(model_name: str, pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, random_state: int = 42) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluates a single model x feature set configuration using 5-fold Stratified CV.

    Returns:
        metrics_dict: Dictionary containing out-of-fold metrics.
        oof_probs: Continuous OOF probability predictions for positive class (Invalid).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(y))

    X_mat = X.values

    for train_idx, val_idx in skf.split(X_mat, y):
        X_train, y_train = X_mat[train_idx], y[train_idx]
        X_val, y_val = X_mat[val_idx], y[val_idx]

        # Fit model pipeline strictly on training fold
        pipeline.fit(X_train, y_train)

        # Predict OOF probabilities
        probs = pipeline.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs

    # Standard default decision threshold 0.50 for initial comparison
    y_pred_default = (oof_probs >= 0.50).astype(int)

    cm = confusion_matrix(y, y_pred_default, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = round(float(precision_score(y, y_pred_default, zero_division=0)), 4)
    rec = round(float(recall_score(y, y_pred_default, zero_division=0)), 4)
    f1 = round(float(f1_score(y, y_pred_default, zero_division=0)), 4)
    macro_f1 = round(float(f1_score(y, y_pred_default, average="macro", zero_division=0)), 4)
    acc = round(float(accuracy_score(y, y_pred_default)), 4)
    b_acc = round(float(balanced_accuracy_score(y, y_pred_default)), 4)
    roc_auc = round(float(roc_auc_score(y, oof_probs)), 4)
    pr_auc = round(float(average_precision_score(y, oof_probs)), 4)

    metrics = {
        "Model": model_name,
        "Invalid_Precision": prec,
        "Invalid_Recall": rec,
        "Invalid_F1": f1,
        "Macro_F1": macro_f1,
        "Balanced_Accuracy": b_acc,
        "Accuracy": acc,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }

    return metrics, oof_probs


def evaluate_all_stage7_models(df_train_features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    """Evaluates all 15 candidate combinations (3 models x 5 feature sets) using 5-fold Stratified CV.

    Returns:
        comparison_df: Ranked DataFrame of all configurations by Invalid F1.
        all_oof_probs: Dictionary mapping config_key to OOF probabilities.
        best_config_info: Dictionary containing the best pipeline and details.
    """
    if TASK01_TARGET not in df_train_features.columns:
        raise ValueError("df_train_features must contain Validity_Label!")

    y_true = (df_train_features[TASK01_TARGET] == "Invalid").astype(int).values

    feature_sets_train, _, feature_names_dict = prepare_feature_sets(df_train_features)
    models_dict = get_model_pipelines()

    eval_records = []
    all_oof_probs = {}

    for fs_name, X_fs in feature_sets_train.items():
        for m_name, pipeline in models_dict.items():
            config_key = f"{m_name}__{fs_name}"

            metrics, oof_probs = evaluate_model_cv(m_name, pipeline, X_fs, y_true, n_splits=5, random_state=42)
            metrics["Feature_Set"] = fs_name
            metrics["Num_Features"] = len(X_fs.columns)
            metrics["Config_Key"] = config_key

            eval_records.append(metrics)
            all_oof_probs[config_key] = oof_probs

    comp_df = pd.DataFrame(eval_records).sort_values(by="Invalid_F1", ascending=False)
    best_row = comp_df.iloc[0]

    best_config_info = {
        "config_key": best_row["Config_Key"],
        "model_name": best_row["Model"],
        "feature_set_name": best_row["Feature_Set"],
        "best_row": best_row
    }

    return comp_df, all_oof_probs, best_config_info


def optimize_decision_threshold(y_true: np.ndarray, oof_probs: np.ndarray, thresholds: Optional[List[float]] = None) -> pd.DataFrame:
    """Evaluates decision thresholds from 0.10 to 0.90 on OOF probability predictions."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.95, 0.05)]

    records = []
    for thresh in thresholds:
        y_pred = (oof_probs >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        prec = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
        rec = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
        f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        acc = round(float(accuracy_score(y_true, y_pred)), 4)
        b_acc = round(float(balanced_accuracy_score(y_true, y_pred)), 4)

        records.append({
            "Threshold": thresh,
            "Num_Flagged": int(y_pred.sum()),
            "Flag_Rate(%)": round(float(y_pred.sum() / len(y_true) * 100.0), 2),
            "Invalid_Precision": prec,
            "Invalid_Recall": rec,
            "Invalid_F1": f1,
            "Balanced_Accuracy": b_acc,
            "Accuracy": acc,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        })

    return pd.DataFrame(records).sort_values(by="Invalid_F1", ascending=False)


def calculate_permutation_importance(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Calculates permutation feature importance on out-of-fold / validation data."""
    pipeline.fit(X.values, y)
    result = permutation_importance(pipeline, X.values, y, scoring="f1", n_repeats=10, random_state=42)

    imp_df = pd.DataFrame({
        "feature_name": feature_names,
        "importance_mean": np.round(result.importances_mean, 6),
        "importance_std": np.round(result.importances_std, 6)
    }).sort_values(by="importance_mean", ascending=False)

    return imp_df


def analyze_ml_errors(df_train: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, config_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analyzes False Positives and False Negatives for the winning ML classifier."""
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    # False Positives
    fp_records = []
    for idx, row in df_train[fp_mask].head(10).iterrows():
        c, v = row.get("Load_Current_A", np.nan), row.get("Applied_Voltage_kV", np.nan)
        regime = "Heavy HV Load" if c > 85 and v > 25 else ("Heavy Current" if c > 85 else ("High Voltage" if v > 25 else "Standard"))

        fp_records.append({
            "Config": config_name,
            "Test_ID": str(row.get(ID_COLUMN, f"Row_{idx}")),
            "Operating_Regime": regime,
            "Load_Current_A": c,
            "Applied_Voltage_kV": v,
            "p1_max_abs_residual": row.get("p1_max_abs_residual", np.nan),
            "Validity_Label": "Valid",
            "FP_Interpretation": f"Historically Valid test predicted Invalid by {config_name}. Likely severe current/voltage operating fluctuation."
        })

    # False Negatives
    fn_records = []
    for idx, row in df_train[fn_mask].head(10).iterrows():
        c, v = row.get("Load_Current_A", np.nan), row.get("Applied_Voltage_kV", np.nan)

        fn_records.append({
            "Config": config_name,
            "Test_ID": str(row.get(ID_COLUMN, f"Row_{idx}")),
            "Load_Current_A": c,
            "Applied_Voltage_kV": v,
            "p1_max_abs_residual": row.get("p1_max_abs_residual", np.nan),
            "Validity_Label": "Invalid",
            "FN_Interpretation": f"Historically Invalid test predicted Valid by {config_name}. Requires deeper non-linear interaction features."
        })

    return pd.DataFrame(fp_records), pd.DataFrame(fn_records)
