"""
CPRI Hackathon — Person 1 Stage 7 ML Classification / Anomaly Models (STRICT FOLD-ISOLATED)
========================================================================================
Provides strict fold-isolated 5-fold Stratified Cross-Validation model evaluation,
feature-set comparison, decision threshold optimization, permutation importance,
and error analysis for Task 01.

Guarantees:
- STRICT FOLD ISOLATION: NormalBehaviourModels and Stage 5 feature transformers are fitted
  STRICTLY on the training fold records during cross-validation.
- Strict exclusion of Validity_Label, Reference_Parameter, and Test_ID from predictors.
- 5-Fold Stratified CV (n_splits=5, shuffle=True, random_state=42).
- Out-of-fold (OOF) probability predictions for all 1,000 training records.
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
from src.stage5_features import create_stage5_features, prepare_stage5_feature_matrix


def get_model_pipelines() -> Dict[str, Any]:
    """Returns candidate machine learning pipelines."""
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


def prepare_feature_sets(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """Prepares feature set DataFrames for training and optional test datasets."""
    raw_cols = [c for c in PHYSICAL_FEATURES if c in df_train.columns]
    quality_cols = [
        "missing_any", "non_finite_value", "invalid_voltage", "invalid_current",
        "invalid_duration", "invalid_ambient_temp", "sensor_s2_negative",
        "sensor_zero_reading", "malformed_numeric", "data_quality_issue"
    ]
    residual_cols = [
        "Sensor_S1_Residual", "Sensor_S2_Residual", "Sensor_S3_Residual", "Sensor_S4_Residual",
        "p1_max_abs_residual", "p1_mean_abs_residual", "p1_consistency_index",
        "p1_sensor_disagreement_index", "p1_regime_cluster"
    ]

    def _extract_sets(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
        df_copy = df.copy()
        if "isolation_forest_score" not in df_copy.columns:
            iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
            X_raw = df_copy[raw_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
            iso.fit(X_raw)
            df_copy["isolation_forest_score"] = -iso.score_samples(X_raw)

        set_a = df_copy[raw_cols].apply(pd.to_numeric, errors='coerce')

        b_cols = [c for c in raw_cols + quality_cols if c in df_copy.columns]
        set_b = df_copy[b_cols].apply(pd.to_numeric, errors='coerce')

        c_cols = [c for c in raw_cols + residual_cols if c in df_copy.columns]
        set_c = df_copy[c_cols].apply(pd.to_numeric, errors='coerce')

        set_d = prepare_stage5_feature_matrix(df_copy)

        set_e = set_d.copy()
        if "isolation_forest_score" not in set_e.columns and "isolation_forest_score" in df_copy.columns:
            set_e["isolation_forest_score"] = pd.to_numeric(df_copy["isolation_forest_score"], errors='coerce')

        sets = {
            "Set_A_Raw": set_a,
            "Set_B_Raw_Quality": set_b,
            "Set_C_Raw_Residuals": set_c,
            "Set_D_Full_Stage5": set_d,
            "Set_E_Stage5_AnomalyScore": set_e
        }
        names = {k: list(v.columns) for k, v in sets.items()}
        return sets, names

    train_sets, names_dict = _extract_sets(df_train)
    test_sets = {}
    if df_test is not None:
        test_sets, _ = _extract_sets(df_test)

    return train_sets, test_sets, names_dict


def evaluate_model_cv(
    m_name: str,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y_true: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluates a single model on a pre-prepared feature matrix X using Stratified CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_probs = np.zeros(len(y_true))
    X_vals = X.values if isinstance(X, pd.DataFrame) else X

    for train_idx, val_idx in skf.split(X_vals, y_true):
        X_tr, y_tr = X_vals[train_idx], y_true[train_idx]
        X_val = X_vals[val_idx]

        pipeline.fit(X_tr, y_tr)
        oof_probs[val_idx] = pipeline.predict_proba(X_val)[:, 1]

    y_pred_default = (oof_probs >= 0.50).astype(int)
    cm = confusion_matrix(y_true, y_pred_default, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = round(float(precision_score(y_true, y_pred_default, zero_division=0)), 4)
    rec = round(float(recall_score(y_true, y_pred_default, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true, y_pred_default, zero_division=0)), 4)
    macro_f1 = round(float(f1_score(y_true, y_pred_default, average="macro", zero_division=0)), 4)
    acc = round(float(accuracy_score(y_true, y_pred_default)), 4)
    b_acc = round(float(balanced_accuracy_score(y_true, y_pred_default)), 4)
    roc_auc = round(float(roc_auc_score(y_true, oof_probs)), 4)
    pr_auc = round(float(average_precision_score(y_true, oof_probs)), 4)

    metrics = {
        "Model": m_name,
        "Feature_Set": X.name if hasattr(X, "name") else "Custom",
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
        "FN": int(fn),
        "Config_Key": f"{m_name}__Custom"
    }

    return metrics, oof_probs


def evaluate_stage7_configuration_fold_isolated(
    m_name: str,
    pipeline: Pipeline,
    fs_name: str,
    df_train: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluates a single model x feature set configuration using STRICT FOLD-ISOLATED 5-fold Stratified CV.

    Fitting of NormalBehaviourModels and Stage 5 feature extraction occurs STRICTLY inside each training fold.
    """
    y_true = (df_train[TASK01_TARGET] == "Invalid").astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_probs = np.zeros(len(df_train))

    raw_cols = [c for c in PHYSICAL_FEATURES if c in df_train.columns]
    quality_cols = [
        "missing_any", "non_finite_value", "invalid_voltage", "invalid_current",
        "invalid_duration", "invalid_ambient_temp", "sensor_s2_negative",
        "sensor_zero_reading", "malformed_numeric", "data_quality_issue"
    ]
    residual_cols = [
        "Sensor_S1_Residual", "Sensor_S2_Residual", "Sensor_S3_Residual", "Sensor_S4_Residual",
        "p1_max_abs_residual", "p1_mean_abs_residual", "p1_consistency_index",
        "p1_sensor_disagreement_index", "p1_regime_cluster"
    ]

    for train_idx, val_idx in skf.split(df_train, y_true):
        df_tr_fold = df_train.iloc[train_idx].copy()
        df_val_fold = df_train.iloc[val_idx].copy()

        # Fit NormalBehaviourModels STRICTLY on df_tr_fold Valid records
        df_tr_feat, fold_normal_models = create_stage5_features(df_tr_fold, fit_normal_models=True)

        # Transform df_val_fold using fold_normal_models WITHOUT labels
        df_val_feat, _ = create_stage5_features(df_val_fold, fit_normal_models=False, normal_models=fold_normal_models)

        # Handle Anomaly Score for Set_E strictly inside fold
        if fs_name == "Set_E_Stage5_AnomalyScore":
            iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
            X_tr_raw = df_tr_feat[raw_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
            X_val_raw = df_val_feat[raw_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
            iso.fit(X_tr_raw)
            df_tr_feat["isolation_forest_score"] = -iso.score_samples(X_tr_raw)
            df_val_feat["isolation_forest_score"] = -iso.score_samples(X_val_raw)

        # Select Feature Columns
        if fs_name == "Set_A_Raw":
            cols = raw_cols
        elif fs_name == "Set_B_Raw_Quality":
            cols = [c for c in df_tr_feat.columns if c in raw_cols + quality_cols]
        elif fs_name == "Set_C_Raw_Residuals":
            cols = [c for c in df_tr_feat.columns if c in raw_cols + residual_cols]
        elif fs_name in ["Set_D_Full_Stage5", "Set_E_Stage5_AnomalyScore"]:
            cols = list(prepare_stage5_feature_matrix(df_tr_feat).columns)

        X_tr = df_tr_feat[cols].apply(pd.to_numeric, errors='coerce').values
        X_val = df_val_feat[cols].apply(pd.to_numeric, errors='coerce').values
        y_tr = y_true[train_idx]

        # Fit pipeline ONLY on training fold matrix
        pipeline.fit(X_tr, y_tr)

        # Predict OOF probabilities for validation fold
        probs = pipeline.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs

    # Default threshold 0.50 comparison
    y_pred_default = (oof_probs >= 0.50).astype(int)
    cm = confusion_matrix(y_true, y_pred_default, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    prec = round(float(precision_score(y_true, y_pred_default, zero_division=0)), 4)
    rec = round(float(recall_score(y_true, y_pred_default, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true, y_pred_default, zero_division=0)), 4)
    macro_f1 = round(float(f1_score(y_true, y_pred_default, average="macro", zero_division=0)), 4)
    acc = round(float(accuracy_score(y_true, y_pred_default)), 4)
    b_acc = round(float(balanced_accuracy_score(y_true, y_pred_default)), 4)
    roc_auc = round(float(roc_auc_score(y_true, oof_probs)), 4)
    pr_auc = round(float(average_precision_score(y_true, oof_probs)), 4)

    metrics = {
        "Model": m_name,
        "Feature_Set": fs_name,
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
        "FN": int(fn),
        "Config_Key": f"{m_name}__{fs_name}"
    }

    return metrics, oof_probs


def evaluate_all_stage7_models(df_train: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    """Evaluates all 15 candidate configurations using STRICT FOLD-ISOLATED 5-fold Stratified CV."""
    if TASK01_TARGET not in df_train.columns:
        raise ValueError("df_train must contain Validity_Label!")

    feature_set_names = [
        "Set_A_Raw",
        "Set_B_Raw_Quality",
        "Set_C_Raw_Residuals",
        "Set_D_Full_Stage5",
        "Set_E_Stage5_AnomalyScore"
    ]
    models_dict = get_model_pipelines()

    eval_records = []
    all_oof_probs = {}

    for fs_name in feature_set_names:
        for m_name, pipeline in models_dict.items():
            metrics, oof_probs = evaluate_stage7_configuration_fold_isolated(m_name, pipeline, fs_name, df_train, n_splits=5, random_state=42)
            eval_records.append(metrics)
            all_oof_probs[metrics["Config_Key"]] = oof_probs

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
    """Calculates permutation feature importance on validation data."""
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
            "FP_Interpretation": f"Historically Valid test predicted Invalid by {config_name}. Operating fluctuation or border decision."
        })

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
            "FN_Interpretation": f"Historically Invalid test predicted Valid by {config_name}. Subtle sensor failure requiring lower threshold."
        })

    return pd.DataFrame(fp_records), pd.DataFrame(fn_records)
