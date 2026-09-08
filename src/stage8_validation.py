"""
CPRI Hackathon -- Person 1 Stage 8 Validation and Threshold Selection Module
=============================================================================

Reusable, leakage-free validation utilities for Task 01.

This module reuses the validated Stage 7 Random Forest + Stage 5 feature
engineering stack, computes strictly out-of-fold probabilities with
StratifiedKFold(5, shuffle=True, random_state=42), sweeps decision thresholds
on OOF predictions only, evaluates fold stability, and performs systematic
false-positive / false-negative analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from src.dataset_loader import ID_COLUMN, PHYSICAL_FEATURES, TASK01_TARGET, load_training_data
from src.stage5_features import create_stage5_features, prepare_stage5_feature_matrix
from src.stage7_ml_models import get_model_pipelines

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_FIGURES_DIR = DEFAULT_OUTPUT_DIR / "figures"

DEFAULT_MODEL_NAME = "Random_Forest"
DEFAULT_FEATURE_SET_NAME = "Set_D_Full_Stage5"
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_SPLITS = 5

QUALITY_COLUMNS = [
    "missing_any",
    "non_finite_value",
    "invalid_voltage",
    "invalid_current",
    "invalid_duration",
    "invalid_ambient_temp",
    "sensor_s2_negative",
    "sensor_zero_reading",
    "malformed_numeric",
    "data_quality_issue",
]

RESIDUAL_COLUMNS = [
    "Sensor_S1_Residual",
    "Sensor_S2_Residual",
    "Sensor_S3_Residual",
    "Sensor_S4_Residual",
    "p1_max_abs_residual",
    "p1_mean_abs_residual",
    "p1_consistency_index",
    "p1_sensor_disagreement_index",
    "p1_regime_cluster",
]

THRESHOLD_CANDIDATES = tuple(np.round(np.arange(0.10, 0.91, 0.05), 2).tolist())


@dataclass(frozen=True)
class Stage8Config:
    """Resolved Stage 7 configuration used as the Stage 8 validation baseline."""

    model_name: str
    feature_set_name: str
    config_key: str


def get_threshold_candidates() -> List[float]:
    """Returns the exact Stage 8 threshold candidate grid."""

    return [float(x) for x in THRESHOLD_CANDIDATES]


def _labels_to_binary(labels: Sequence[Any]) -> np.ndarray:
    """Converts Validity_Label values to binary Invalid=1 / Valid=0."""

    s = pd.Series(labels)
    return (s.astype(str).str.strip() == "Invalid").astype(int).to_numpy()


def _binary_to_labels(binary: Sequence[int]) -> np.ndarray:
    """Converts binary Invalid flags to string labels."""

    arr = np.asarray(binary, dtype=int)
    return np.where(arr == 1, "Invalid", "Valid")


def _regime_name(row: pd.Series) -> str:
    """Maps a record to the Stage 4 operating regime language."""

    current = pd.to_numeric(row.get("Load_Current_A"), errors="coerce")
    voltage = pd.to_numeric(row.get("Applied_Voltage_kV"), errors="coerce")
    if pd.isna(current) or pd.isna(voltage):
        return "Unknown"
    if current > 85 and voltage > 25:
        return "Heavy HV Load"
    if current > 85:
        return "Heavy Current"
    if voltage > 25:
        return "High Voltage"
    return "Standard"


def _ensure_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerces a feature frame to numeric values and replaces infinities with NaN."""

    return df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _feature_columns_for_set(df_features: pd.DataFrame, feature_set_name: str) -> List[str]:
    """Returns the Stage 7 feature columns for the requested feature set."""

    raw_cols = [c for c in PHYSICAL_FEATURES if c in df_features.columns]
    if feature_set_name == "Set_A_Raw":
        cols = raw_cols
    elif feature_set_name == "Set_B_Raw_Quality":
        cols = [c for c in raw_cols + QUALITY_COLUMNS if c in df_features.columns]
    elif feature_set_name == "Set_C_Raw_Residuals":
        cols = [c for c in raw_cols + RESIDUAL_COLUMNS if c in df_features.columns]
    elif feature_set_name in {"Set_D_Full_Stage5", "Set_E_Stage5_AnomalyScore"}:
        cols = list(prepare_stage5_feature_matrix(df_features).columns)
    else:
        raise ValueError(f"Unknown feature set: {feature_set_name}")

    return cols


def _maybe_add_isolation_score(
    df_train_feat: pd.DataFrame,
    df_val_feat: pd.DataFrame,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Adds a fold-isolated Isolation Forest score for Set_E."""

    raw_cols = [c for c in PHYSICAL_FEATURES if c in df_train_feat.columns]
    if not raw_cols:
        return df_train_feat, df_val_feat

    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state)
    x_train = df_train_feat[raw_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    x_val = df_val_feat[raw_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    iso.fit(x_train)
    df_train_feat = df_train_feat.copy()
    df_val_feat = df_val_feat.copy()
    df_train_feat["isolation_forest_score"] = -iso.score_samples(x_train)
    df_val_feat["isolation_forest_score"] = -iso.score_samples(x_val)
    return df_train_feat, df_val_feat


def load_stage7_best_configuration(
    df_train: Optional[pd.DataFrame] = None,
    *,
    prefer_cached: bool = True,
) -> Stage8Config:
    """
    Resolves the best Stage 7 configuration.

    The cached Stage 7 comparison CSV is used when present. If it is missing,
    the Stage 7 evaluation is recomputed from the current training data.
    """

    cached_path = DEFAULT_OUTPUT_DIR / "p1_stage7_ml_model_comparison.csv"
    if prefer_cached and cached_path.exists():
        comp_df = pd.read_csv(cached_path)
    else:
        if df_train is None:
            df_train = load_training_data()
        from src.stage7_ml_models import evaluate_all_stage7_models

        comp_df, _, _ = evaluate_all_stage7_models(df_train)

    required = {"Model", "Feature_Set", "Config_Key", "Invalid_F1", "Invalid_Recall", "Invalid_Precision"}
    missing = required - set(comp_df.columns)
    if missing:
        raise ValueError(f"Stage 7 comparison data missing columns: {sorted(missing)}")

    ordered = comp_df.sort_values(
        by=["Invalid_F1", "Invalid_Recall", "Invalid_Precision", "Accuracy"],
        ascending=[False, False, False, False],
    )
    best = ordered.iloc[0]
    return Stage8Config(
        model_name=str(best["Model"]),
        feature_set_name=str(best["Feature_Set"]),
        config_key=str(best["Config_Key"]),
    )


def prepare_fold_feature_matrices(
    df_train_fold: pd.DataFrame,
    df_val_fold: pd.DataFrame,
    *,
    feature_set_name: str = DEFAULT_FEATURE_SET_NAME,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Generates fold-isolated Stage 5 features and feature matrices for one CV fold.

    Returns a dictionary containing:
    - X_train: training matrix
    - X_val: validation matrix
    - train_features / val_features: enriched fold dataframes
    - feature_names: ordered list of model feature columns
    """

    df_train_feat, fold_models = create_stage5_features(df_train_fold, fit_normal_models=True)
    df_val_feat, _ = create_stage5_features(df_val_fold, fit_normal_models=False, normal_models=fold_models)

    if feature_set_name == "Set_E_Stage5_AnomalyScore":
        df_train_feat, df_val_feat = _maybe_add_isolation_score(df_train_feat, df_val_feat, random_state)

    feature_cols = _feature_columns_for_set(df_train_feat, feature_set_name)
    feature_cols = [c for c in feature_cols if c in df_val_feat.columns]
    x_train = _ensure_numeric_frame(df_train_feat[feature_cols])
    x_val = _ensure_numeric_frame(df_val_feat[feature_cols])

    return {
        "X_train": x_train,
        "X_val": x_val,
        "train_features": df_train_feat,
        "val_features": df_val_feat,
        "feature_names": list(feature_cols),
    }


def _compute_metrics_from_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes the standard Invalid-positive metrics used throughout Stage 8."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "Invalid_Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Invalid_Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "Invalid_F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Balanced_Accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["ROC_AUC"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        metrics["PR_AUC"] = round(float(average_precision_score(y_true, y_prob)), 4)
    return metrics


def _validate_oof_predictions(oof_df: pd.DataFrame) -> None:
    """Raises if the OOF prediction frame violates Stage 8 integrity constraints."""

    required = {ID_COLUMN, TASK01_TARGET, "oof_probability_invalid", "fold", "source_index"}
    missing = required - set(oof_df.columns)
    if missing:
        raise ValueError(f"OOF predictions missing required columns: {sorted(missing)}")

    if oof_df[ID_COLUMN].nunique() != len(oof_df):
        raise ValueError("Each Test_ID must appear exactly once in the OOF prediction table.")

    probs = pd.to_numeric(oof_df["oof_probability_invalid"], errors="coerce")
    if probs.isna().any():
        raise ValueError("OOF probabilities contain missing values.")
    if not ((probs >= 0.0) & (probs <= 1.0)).all():
        raise ValueError("OOF probabilities must lie in [0, 1].")


def generate_oof_predictions(
    df_train: Optional[pd.DataFrame] = None,
    *,
    model_name: Optional[str] = None,
    feature_set_name: Optional[str] = None,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generates strictly fold-isolated out-of-fold predictions for the training data.

    The returned frame contains the raw record fields, engineered Stage 5 features,
    the fold number, the source row index, and the OOF Invalid probability.
    """

    if df_train is None:
        df_train = load_training_data()
    else:
        df_train = df_train.copy()

    if model_name is None or feature_set_name is None:
        best = load_stage7_best_configuration(df_train=df_train, prefer_cached=True)
        model_name = model_name or best.model_name
        feature_set_name = feature_set_name or best.feature_set_name

    if model_name not in get_model_pipelines():
        raise ValueError(f"Unknown model_name: {model_name}")

    y_true = _labels_to_binary(df_train[TASK01_TARGET])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipeline_template = get_model_pipelines()[model_name]

    fold_frames: List[pd.DataFrame] = []
    for fold_number, (train_idx, val_idx) in enumerate(skf.split(df_train, y_true), start=1):
        df_train_fold = df_train.iloc[train_idx].copy()
        df_val_fold = df_train.iloc[val_idx].copy()

        matrices = prepare_fold_feature_matrices(
            df_train_fold,
            df_val_fold,
            feature_set_name=feature_set_name,
            random_state=random_state,
        )
        pipeline = clone(pipeline_template)
        pipeline.fit(matrices["X_train"], y_true[train_idx])
        probs = pipeline.predict_proba(matrices["X_val"])[:, 1]

        fold_df = matrices["val_features"].copy()
        fold_df["source_index"] = val_idx
        fold_df["fold"] = fold_number
        fold_df["true_invalid"] = y_true[val_idx]
        fold_df["oof_probability_invalid"] = probs
        fold_df["model_name"] = model_name
        fold_df["feature_set_name"] = feature_set_name
        fold_frames.append(fold_df)

    oof_df = pd.concat(fold_frames, ignore_index=True).sort_values("source_index").reset_index(drop=True)
    oof_df["OOF_Rank"] = np.arange(1, len(oof_df) + 1)
    _validate_oof_predictions(oof_df)
    return oof_df


def evaluate_thresholds(
    oof_df: pd.DataFrame,
    thresholds: Optional[Sequence[float]] = None,
    *,
    probability_col: str = "oof_probability_invalid",
    label_col: str = TASK01_TARGET,
) -> pd.DataFrame:
    """
    Evaluates the full Stage 8 threshold grid on OOF probabilities only.

    Invalid is treated as the positive class.
    """

    if thresholds is None:
        thresholds = get_threshold_candidates()

    y_true = _labels_to_binary(oof_df[label_col])
    probs = pd.to_numeric(oof_df[probability_col], errors="coerce").to_numpy()
    if np.isnan(probs).any():
        raise ValueError("Threshold analysis cannot proceed with missing OOF probabilities.")

    records: List[Dict[str, Any]] = []
    for threshold in thresholds:
        threshold = float(round(threshold, 2))
        y_pred = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics = _compute_metrics_from_labels(y_true, y_pred)
        records.append(
            {
                "Threshold": threshold,
                "Num_Flagged": int(y_pred.sum()),
                "Flag_Rate(%)": round(float(y_pred.mean() * 100.0), 2),
                "Invalid_Precision": metrics["Invalid_Precision"],
                "Invalid_Recall": metrics["Invalid_Recall"],
                "Invalid_F1": metrics["Invalid_F1"],
                "Balanced_Accuracy": metrics["Balanced_Accuracy"],
                "Accuracy": metrics["Accuracy"],
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "Confusion_Matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]",
            }
        )

    return pd.DataFrame(records).sort_values("Threshold").reset_index(drop=True)


def select_threshold(
    threshold_df: pd.DataFrame,
    *,
    f1_tolerance: float = 0.001,
) -> Dict[str, Any]:
    """
    Selects a defensible final threshold from the OOF threshold sweep.

    Selection rule:
    1. Maximize Invalid F1.
    2. If thresholds are nearly tied within `f1_tolerance`, prefer higher Invalid Recall.
    3. Then prefer higher Invalid Precision.
    4. Finally, use the lowest threshold among remaining ties for determinism.
    """

    if threshold_df.empty:
        raise ValueError("Threshold analysis table is empty.")

    work = threshold_df.copy().sort_values("Threshold").reset_index(drop=True)
    best_f1 = float(work["Invalid_F1"].max())
    candidate_mask = work["Invalid_F1"] >= (best_f1 - f1_tolerance)
    candidates = work.loc[candidate_mask].copy()
    candidates = candidates.sort_values(
        by=["Invalid_Recall", "Invalid_Precision", "FP", "Threshold"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    selected = candidates.iloc[0]

    rationale = (
        f"Selected threshold {float(selected['Threshold']):.2f} from {len(candidates)} threshold(s) "
        f"within {f1_tolerance:.3f} of the best Invalid F1 ({best_f1:.4f}). "
        "Primary criterion: Invalid F1. Tie-breaks favored higher Invalid recall, then higher Invalid precision, "
        "then fewer false positives and a deterministic lower-threshold choice."
    )

    return {
        "selected_threshold": float(selected["Threshold"]),
        "best_f1": best_f1,
        "candidate_thresholds": [float(x) for x in candidates["Threshold"].tolist()],
        "candidate_count": int(len(candidates)),
        "selection_rule": rationale,
        "selected_row": selected.to_dict(),
    }


def evaluate_fold_stability(
    oof_df: pd.DataFrame,
    selected_threshold: float,
    *,
    probability_col: str = "oof_probability_invalid",
    label_col: str = TASK01_TARGET,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates the selected threshold separately on each CV fold and summarizes stability.
    """

    work = oof_df.copy()
    work["selected_prediction"] = (pd.to_numeric(work[probability_col], errors="coerce") >= selected_threshold).astype(int)
    y_true = _labels_to_binary(work[label_col])
    work["true_invalid_binary"] = y_true

    fold_rows: List[Dict[str, Any]] = []
    for fold_number in sorted(work["fold"].dropna().astype(int).unique()):
        fold_df = work[work["fold"].astype(int) == fold_number].copy()
        fold_y_true = _labels_to_binary(fold_df[label_col])
        fold_y_pred = fold_df["selected_prediction"].astype(int).to_numpy()
        fold_probs = pd.to_numeric(fold_df[probability_col], errors="coerce").to_numpy()
        metrics = _compute_metrics_from_labels(fold_y_true, fold_y_pred, fold_probs)
        fold_rows.append(
            {
                "Fold": int(fold_number),
                "Fold_Size": int(len(fold_df)),
                "Invalid_Precision": metrics["Invalid_Precision"],
                "Invalid_Recall": metrics["Invalid_Recall"],
                "Invalid_F1": metrics["Invalid_F1"],
                "Balanced_Accuracy": metrics["Balanced_Accuracy"],
                "Accuracy": metrics["Accuracy"],
                "TP": metrics["TP"],
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
            }
        )

    fold_metrics_df = pd.DataFrame(fold_rows).sort_values("Fold").reset_index(drop=True)
    summary_rows: List[Dict[str, Any]] = []
    for metric in ["Invalid_Precision", "Invalid_Recall", "Invalid_F1", "Balanced_Accuracy", "Accuracy"]:
        values = fold_metrics_df[metric].astype(float).to_numpy()
        summary_rows.append(
            {
                "Metric": metric,
                "Mean": round(float(np.mean(values)), 4),
                "Std": round(float(np.std(values, ddof=0)), 4),
                "Min": round(float(np.min(values)), 4),
                "Max": round(float(np.max(values)), 4),
            }
        )

    return fold_metrics_df, pd.DataFrame(summary_rows)


def _summarize_error_patterns(error_df: pd.DataFrame, error_type: str) -> pd.DataFrame:
    """Builds a concise systematics table for the misclassification report."""

    if error_df.empty:
        return pd.DataFrame(
            [
                {
                    "Pattern_Category": f"{error_type} Count",
                    "Pattern": "None",
                    "Count": 0,
                    "Rate(%)": 0.0,
                    "Notes": f"No {error_type} records at the selected threshold.",
                }
            ]
        )

    total = len(error_df)
    rows: List[Dict[str, Any]] = []

    regime_counts = error_df["operating_regime"].value_counts().to_dict()
    for regime, count in regime_counts.items():
        rows.append(
            {
                "Pattern_Category": "Operating Regime",
                "Pattern": regime,
                "Count": int(count),
                "Rate(%)": round(float(count / total * 100.0), 2),
                "Notes": f"{count} / {total} {error_type} cases occur in this regime.",
            }
        )

    quality_count = int(error_df.get("data_quality_issue", pd.Series(False, index=error_df.index)).sum())
    if quality_count:
        rows.append(
            {
                "Pattern_Category": "Data Quality",
                "Pattern": "data_quality_issue",
                "Count": quality_count,
                "Rate(%)": round(float(quality_count / total * 100.0), 2),
                "Notes": "Definite data-quality failure signal appears in the error subset.",
            }
        )

    high_residual = int((pd.to_numeric(error_df.get("p1_max_abs_residual"), errors="coerce") >= 10).sum())
    rows.append(
        {
            "Pattern_Category": "Residual Severity",
            "Pattern": "p1_max_abs_residual >= 10",
            "Count": high_residual,
            "Rate(%)": round(float(high_residual / total * 100.0), 2),
            "Notes": "Counts cases with large fold-isolated residuals.",
        }
    )

    if "Load_Current_A" in error_df.columns:
        heavy_current = int((pd.to_numeric(error_df["Load_Current_A"], errors="coerce") > 85).sum())
        rows.append(
            {
                "Pattern_Category": "Load",
                "Pattern": "Load_Current_A > 85",
                "Count": heavy_current,
                "Rate(%)": round(float(heavy_current / total * 100.0), 2),
                "Notes": "High-current operating regime.",
            }
        )

    if "Applied_Voltage_kV" in error_df.columns:
        high_voltage = int((pd.to_numeric(error_df["Applied_Voltage_kV"], errors="coerce") > 25).sum())
        rows.append(
            {
                "Pattern_Category": "Voltage",
                "Pattern": "Applied_Voltage_kV > 25",
                "Count": high_voltage,
                "Rate(%)": round(float(high_voltage / total * 100.0), 2),
                "Notes": "High-voltage operating regime.",
            }
        )

    return pd.DataFrame(rows)


def analyze_misclassifications(
    oof_df: pd.DataFrame,
    selected_threshold: float,
    *,
    probability_col: str = "oof_probability_invalid",
    label_col: str = TASK01_TARGET,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extracts every false positive and false negative and adds systematic pattern flags.
    """

    work = oof_df.copy()
    work["selected_prediction_binary"] = (pd.to_numeric(work[probability_col], errors="coerce") >= selected_threshold).astype(int)
    work["selected_prediction"] = _binary_to_labels(work["selected_prediction_binary"])
    work["operating_regime"] = work.apply(_regime_name, axis=1)
    work["high_current_flag"] = pd.to_numeric(work.get("Load_Current_A"), errors="coerce") > 85
    work["high_voltage_flag"] = pd.to_numeric(work.get("Applied_Voltage_kV"), errors="coerce") > 25
    work["high_temperature_flag"] = pd.to_numeric(work.get("Ambient_Temperature_C"), errors="coerce") > 30
    work["long_duration_flag"] = pd.to_numeric(work.get("Test_Duration_min"), errors="coerce") >= 120
    work["high_residual_flag"] = pd.to_numeric(work.get("p1_max_abs_residual"), errors="coerce") >= 10
    work["sensor_disagreement_flag"] = pd.to_numeric(work.get("p1_sensor_disagreement_index"), errors="coerce") >= 1.5

    actual_invalid = _labels_to_binary(work[label_col])
    fp_mask = (actual_invalid == 0) & (work["selected_prediction_binary"].to_numpy() == 1)
    fn_mask = (actual_invalid == 1) & (work["selected_prediction_binary"].to_numpy() == 0)

    columns_to_keep = [
        ID_COLUMN,
        label_col,
        "fold",
        "source_index",
        "oof_probability_invalid",
        "selected_prediction",
        "selected_prediction_binary",
        "operating_regime",
        "Load_Current_A",
        "Applied_Voltage_kV",
        "Ambient_Temperature_C",
        "Test_Duration_min",
        "Sensor_S1",
        "Sensor_S2",
        "Sensor_S3",
        "Sensor_S4",
        "missing_any",
        "non_finite_value",
        "invalid_voltage",
        "invalid_current",
        "invalid_duration",
        "invalid_ambient_temp",
        "sensor_s2_negative",
        "sensor_zero_reading",
        "malformed_numeric",
        "data_quality_issue",
        "p1_max_abs_residual",
        "p1_mean_abs_residual",
        "p1_consistency_index",
        "p1_sensor_disagreement_index",
        "p1_regime_cluster",
        "high_current_flag",
        "high_voltage_flag",
        "high_temperature_flag",
        "long_duration_flag",
        "high_residual_flag",
        "sensor_disagreement_flag",
    ]
    available_cols = [c for c in columns_to_keep if c in work.columns]

    fp_df = work.loc[fp_mask, available_cols].copy().reset_index(drop=True)
    fn_df = work.loc[fn_mask, available_cols].copy().reset_index(drop=True)
    fp_df["error_type"] = "FP"
    fn_df["error_type"] = "FN"

    pattern_rows = []
    pattern_rows.extend(_summarize_error_patterns(fp_df, "FP").assign(Error_Type="FP").to_dict("records"))
    pattern_rows.extend(_summarize_error_patterns(fn_df, "FN").assign(Error_Type="FN").to_dict("records"))
    summary_df = pd.DataFrame(pattern_rows)

    return fp_df, fn_df, summary_df


def build_confusion_matrix(
    oof_df: pd.DataFrame,
    selected_threshold: float,
    *,
    probability_col: str = "oof_probability_invalid",
    label_col: str = TASK01_TARGET,
) -> pd.DataFrame:
    """Builds the final confusion matrix using all OOF predictions."""

    y_true = _labels_to_binary(oof_df[label_col])
    y_pred = (pd.to_numeric(oof_df[probability_col], errors="coerce") >= selected_threshold).astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return pd.DataFrame(
        [
            {
                "Actual": "True Valid",
                "Predicted_Valid": int(tn),
                "Predicted_Invalid": int(fp),
                "Row_Total": int(tn + fp),
            },
            {
                "Actual": "True Invalid",
                "Predicted_Valid": int(fn),
                "Predicted_Invalid": int(tp),
                "Row_Total": int(fn + tp),
            },
        ]
    )


def _markdown_table(df: pd.DataFrame, *, index: bool = False) -> str:
    """Safely renders a dataframe as markdown."""

    if df.empty:
        return "_No records_"
    return df.to_markdown(index=index)


def generate_threshold_selection_report(
    threshold_df: pd.DataFrame,
    selection: Dict[str, Any],
    *,
    output_path: Optional[Path] = None,
) -> str:
    """Builds and optionally writes the threshold selection report."""

    selected_row = selection["selected_row"]
    top_region = threshold_df[
        threshold_df["Invalid_F1"] >= (threshold_df["Invalid_F1"].max() - 0.001)
    ].sort_values(["Invalid_Recall", "Invalid_Precision", "Threshold"], ascending=[False, False, True])

    report = f"""# Person 1 - Stage 8: Threshold Selection Report

## Threshold Candidates

The following thresholds were evaluated on leakage-free out-of-fold probabilities:

{_markdown_table(threshold_df)}

## Best F1 Region

Thresholds within 0.001 of the best Invalid F1:

{_markdown_table(top_region[['Threshold', 'Invalid_Precision', 'Invalid_Recall', 'Invalid_F1', 'TP', 'TN', 'FP', 'FN']])}

## Selection Rule

{selection['selection_rule']}

## Selected Threshold

- Selected threshold: `{float(selection['selected_threshold']):.2f}`
- Best Invalid F1: `{float(selection['best_f1']):.4f}`
- Selected Invalid Precision: `{float(selected_row['Invalid_Precision']):.4f}`
- Selected Invalid Recall: `{float(selected_row['Invalid_Recall']):.4f}`
- Selected Invalid F1: `{float(selected_row['Invalid_F1']):.4f}`
- Selected Accuracy: `{float(selected_row['Accuracy']):.4f}`
- Selected Balanced Accuracy: `{float(selected_row['Balanced_Accuracy']):.4f}`

## Operational Interpretation

The selected threshold is favored because it preserves the strongest Invalid recall in the near-optimal F1 region while keeping false positives low enough for practical screening use.
"""

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    return report


def generate_misclassification_report(
    fp_df: pd.DataFrame,
    fn_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    selected_threshold: float,
    output_path: Optional[Path] = None,
) -> str:
    """Builds and optionally writes the false-positive / false-negative report."""

    fp_regime = fp_df["operating_regime"].value_counts().to_dict() if not fp_df.empty else {}
    fn_regime = fn_df["operating_regime"].value_counts().to_dict() if not fn_df.empty else {}

    def _mean_or_na(frame: pd.DataFrame, column: str) -> str:
        if frame.empty or column not in frame.columns:
            return "N/A"
        value = pd.to_numeric(frame[column], errors="coerce").mean()
        return "N/A" if pd.isna(value) else f"{value:.4f}"

    report = f"""# Person 1 - Stage 8: Misclassification Analysis

## Selected Threshold

`{selected_threshold:.2f}`

## False Positives

- Count: `{len(fp_df)}`
- Most common regimes: `{fp_regime if fp_regime else 'None'}`
- Mean current: `{_mean_or_na(fp_df, 'Load_Current_A')}`
- Mean voltage: `{_mean_or_na(fp_df, 'Applied_Voltage_kV')}`
- Mean p1_max_abs_residual: `{_mean_or_na(fp_df, 'p1_max_abs_residual')}`

## False Negatives

- Count: `{len(fn_df)}`
- Most common regimes: `{fn_regime if fn_regime else 'None'}`
- Mean current: `{_mean_or_na(fn_df, 'Load_Current_A')}`
- Mean voltage: `{_mean_or_na(fn_df, 'Applied_Voltage_kV')}`
- Mean p1_max_abs_residual: `{_mean_or_na(fn_df, 'p1_max_abs_residual')}`

## Systematic Pattern Table

{_markdown_table(summary_df)}

## Interpretation

False positives are reviewed for legitimate unusual operating regimes, high load, high voltage, temperature, duration, and elevated residual or sensor disagreement patterns.
False negatives, if present, are reviewed for subtle residual abnormalities, missing values, or cases that look physically normal but remain inconsistent in the engineered features.
"""

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    return report


def generate_validation_report(
    df_train: pd.DataFrame,
    oof_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    selection: Dict[str, Any],
    fold_metrics_df: pd.DataFrame,
    fold_summary_df: pd.DataFrame,
    fp_df: pd.DataFrame,
    fn_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    *,
    output_path: Optional[Path] = None,
) -> str:
    """Builds and optionally writes the full Stage 8 validation report."""

    y_true = _labels_to_binary(oof_df[TASK01_TARGET])
    y_prob = pd.to_numeric(oof_df["oof_probability_invalid"], errors="coerce").to_numpy()
    selected_threshold = float(selection["selected_threshold"])
    y_pred = (y_prob >= selected_threshold).astype(int)
    overall = _compute_metrics_from_labels(y_true, y_pred, y_prob)
    valid_count = int((df_train[TASK01_TARGET].astype(str).str.strip() == "Valid").sum())
    invalid_count = int((df_train[TASK01_TARGET].astype(str).str.strip() == "Invalid").sum())
    invalid_rate = invalid_count / len(df_train) * 100.0 if len(df_train) else 0.0
    valid_rate = valid_count / len(df_train) * 100.0 if len(df_train) else 0.0

    report = f"""# Person 1 - Stage 8: Validation & Threshold Selection Report

## A. Dataset Summary

- Training records: `{len(df_train)}`
- Valid count: `{valid_count}` (`{valid_rate:.2f}%`)
- Invalid count: `{invalid_count}` (`{invalid_rate:.2f}%`)
- Class imbalance: `Valid:Invalid = {valid_count}:{invalid_count}`

## B. CV Methodology

- 5-fold StratifiedKFold
- shuffle=True
- random_state=42
- Out-of-fold prediction generation only
- Threshold tuning performed exclusively on OOF probabilities
- Preprocessing and feature engineering kept inside fold-isolated pipelines
- `Test_ID`, `Validity_Label`, and `Reference_Parameter` are excluded from model features

## C. Threshold Analysis

Selected threshold: `{selected_threshold:.2f}`

Selection rule:

{selection['selection_rule']}

Threshold sweep:

{_markdown_table(threshold_df)}

## D. Overall OOF Performance

- Invalid Precision: `{overall['Invalid_Precision']:.4f}`
- Invalid Recall: `{overall['Invalid_Recall']:.4f}`
- Invalid F1: `{overall['Invalid_F1']:.4f}`
- Accuracy: `{overall['Accuracy']:.4f}`
- Balanced Accuracy: `{overall['Balanced_Accuracy']:.4f}`
- ROC-AUC: `{overall.get('ROC_AUC', float('nan')):.4f}`
- PR-AUC: `{overall.get('PR_AUC', float('nan')):.4f}`

### Confusion Matrix

{_markdown_table(confusion_df)}

### Named Confusion Counts

- True Valid: `{int(overall['TN'])}`
- False Invalid: `{int(overall['FP'])}`
- True Invalid: `{int(overall['TP'])}`
- Missed Invalid: `{int(overall['FN'])}`

## E. Fold-Level Performance

### Per Fold

{_markdown_table(fold_metrics_df)}

### Stability Summary

{_markdown_table(fold_summary_df)}

## F. Misclassification Analysis

- False Positive count: `{len(fp_df)}`
- False Negative count: `{len(fn_df)}`

### False Positive Summary

{_markdown_table(fp_df.head(10))}

### False Negative Summary

{_markdown_table(fn_df.head(10))}

### Systematic Pattern Table

{_markdown_table(summary_df)}

## G. Final Recommendation

- Final model: `{DEFAULT_MODEL_NAME}`
- Final feature set: `{DEFAULT_FEATURE_SET_NAME}`
- Final threshold: `{selected_threshold:.2f}`
- Expected Invalid detection behavior: prioritize catching Invalid records first, with a very small number of false positives concentrated in unusual but physically plausible operating regimes.
- Integration readiness: {'Stable enough for integration' if fold_summary_df["Std"].max() <= 0.05 else 'Needs additional review before integration'}
"""

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    return report


def generate_gap_analysis(output_path: Optional[Path] = None) -> str:
    """Writes the Stage 8 gap analysis report."""

    rows = [
        {
            "Requirement": "Leakage-free 5-fold CV with OOF predictions",
            "Already Implemented": "Stage 7 provided fold-isolated CV and OOF predictions.",
            "Missing": "Stage 8-specific fold numbering and validation artifacts.",
            "Action Taken": "Reused the Stage 7 pipeline and regenerated fold-isolated OOF predictions with fold metadata.",
        },
        {
            "Requirement": "Threshold sweep on OOF probabilities",
            "Already Implemented": "Stage 7 threshold sweep existed on OOF predictions.",
            "Missing": "Stage 8 selection logic and explanatory report.",
            "Action Taken": "Retained the OOF sweep and added explicit selection rules with tie-breaking on recall and precision.",
        },
        {
            "Requirement": "Cross-fold threshold stability",
            "Already Implemented": "No dedicated Stage 7 fold-level threshold report.",
            "Missing": "Per-fold metrics and stability summary at the final threshold.",
            "Action Taken": "Added fold-level metrics, summary statistics, and a fold-stability figure.",
        },
        {
            "Requirement": "Final confusion matrix",
            "Already Implemented": "Stage 7 had a confusion matrix figure for the winning model.",
            "Missing": "Stage 8-specific final confusion matrix and named cell interpretation.",
            "Action Taken": "Generated a Stage 8 confusion matrix using all OOF predictions at the selected threshold.",
        },
        {
            "Requirement": "Full FP/FN analysis",
            "Already Implemented": "Stage 7 contained top FP/FN examples.",
            "Missing": "Row-level analysis for every FP and FN with engineered features.",
            "Action Taken": "Exported complete FP and FN tables and summarized systematic operating-regime patterns.",
        },
        {
            "Requirement": "Validation report",
            "Already Implemented": "Stage 7 had a summary report.",
            "Missing": "Stage 8 validation report with methodology, stability, and final recommendation.",
            "Action Taken": "Wrote a dedicated Stage 8 validation report with all required sections.",
        },
        {
            "Requirement": "Notebook reproduction",
            "Already Implemented": "Stage 7 notebook exists.",
            "Missing": "Stage 8 notebook that reproduces the validation workflow.",
            "Action Taken": "Created a Stage 8 notebook to load artifacts, sweep thresholds, and inspect errors.",
        },
        {
            "Requirement": "Reusable module and tests",
            "Already Implemented": "Stage 7 reused validation helpers but had no Stage 8 module.",
            "Missing": "Public Stage 8 module and Stage 8 validation tests.",
            "Action Taken": "Implemented `src/stage8_validation.py` and `tests/test_person1_stage8_validation.py`.",
        },
    ]

    report = "# Person 1 - Stage 8: Gap Analysis\n\n| Requirement | Already Implemented | Missing | Action Taken |\n|---|---|---|---|\n"
    for row in rows:
        report += f"| {row['Requirement']} | {row['Already Implemented']} | {row['Missing']} | {row['Action Taken']} |\n"

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    return report


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_stage8_figures(
    oof_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    selected_threshold: float,
    fold_metrics_df: pd.DataFrame,
    *,
    output_dir: Path = DEFAULT_FIGURES_DIR,
) -> None:
    """Generates the required Stage 8 figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    y_true = _labels_to_binary(oof_df[TASK01_TARGET])
    y_pred = (pd.to_numeric(oof_df["oof_probability_invalid"], errors="coerce") >= selected_threshold).astype(int)

    # Confusion matrix
    cm = pd.DataFrame(
        [[int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
         [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())]],
        index=["True Valid", "True Invalid"],
        columns=["Predicted Valid", "Predicted Invalid"],
    )
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Stage 8 Confusion Matrix (threshold={selected_threshold:.2f})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    _save_figure(output_dir / "p1_stage8_confusion_matrix.png")

    # Threshold metrics
    plt.figure(figsize=(10, 5.5))
    thresh_sorted = threshold_df.sort_values("Threshold")
    sns.lineplot(data=thresh_sorted, x="Threshold", y="Invalid_Precision", marker="o", label="Invalid Precision")
    sns.lineplot(data=thresh_sorted, x="Threshold", y="Invalid_Recall", marker="o", label="Invalid Recall")
    sns.lineplot(data=thresh_sorted, x="Threshold", y="Invalid_F1", marker="o", label="Invalid F1")
    plt.axvline(selected_threshold, color="black", linestyle="--", label=f"Selected threshold {selected_threshold:.2f}")
    plt.title("Stage 8 Threshold vs Performance")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    _save_figure(output_dir / "p1_stage8_threshold_metrics.png")

    # Fold stability
    plt.figure(figsize=(8, 5))
    sns.barplot(data=fold_metrics_df, x="Fold", y="Invalid_F1", color="#4C72B0")
    mean_f1 = float(fold_metrics_df["Invalid_F1"].mean())
    plt.axhline(mean_f1, color="crimson", linestyle="--", label=f"Mean Invalid F1 = {mean_f1:.4f}")
    plt.title("Stage 8 Fold Stability: Invalid F1 by Fold")
    plt.ylim(0, 1.05)
    plt.legend()
    _save_figure(output_dir / "p1_stage8_fold_stability.png")


def run_stage8_pipeline(output_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    """
    Executes the full Stage 8 validation workflow and writes all required artifacts.
    """

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_train = load_training_data()
    best_config = load_stage7_best_configuration(df_train=df_train, prefer_cached=True)
    oof_df = generate_oof_predictions(
        df_train,
        model_name=best_config.model_name,
        feature_set_name=best_config.feature_set_name,
        n_splits=DEFAULT_N_SPLITS,
        random_state=DEFAULT_RANDOM_STATE,
    )

    threshold_df = evaluate_thresholds(oof_df)
    selection = select_threshold(threshold_df)
    selected_threshold = float(selection["selected_threshold"])

    oof_df = oof_df.copy()
    oof_df["selected_threshold"] = selected_threshold
    oof_df["prediction_at_selected_threshold"] = _binary_to_labels(
        (pd.to_numeric(oof_df["oof_probability_invalid"], errors="coerce") >= selected_threshold).astype(int)
    )

    fold_metrics_df, fold_summary_df = evaluate_fold_stability(oof_df, selected_threshold)
    fp_df, fn_df, summary_df = analyze_misclassifications(oof_df, selected_threshold)
    confusion_df = build_confusion_matrix(oof_df, selected_threshold)

    # Required CSV outputs
    oof_export_cols = [
        ID_COLUMN,
        TASK01_TARGET,
        "oof_probability_invalid",
        "prediction_at_selected_threshold",
        "fold",
        "selected_threshold",
    ]
    extra_oof_cols = [c for c in oof_df.columns if c not in oof_export_cols]
    oof_out = oof_df[oof_export_cols + extra_oof_cols].copy()
    oof_out.to_csv(output_dir / "p1_stage8_oof_predictions.csv", index=False)
    threshold_df.to_csv(output_dir / "p1_stage8_threshold_analysis.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "p1_stage8_fold_metrics.csv", index=False)
    fp_df.to_csv(output_dir / "p1_stage8_false_positive_analysis.csv", index=False)
    fn_df.to_csv(output_dir / "p1_stage8_false_negative_analysis.csv", index=False)

    # Required markdown reports
    generate_gap_analysis(output_dir / "p1_stage8_gap_analysis.md")
    generate_threshold_selection_report(
        threshold_df,
        selection,
        output_path=output_dir / "p1_stage8_threshold_selection.md",
    )
    generate_misclassification_report(
        fp_df,
        fn_df,
        summary_df,
        selected_threshold=selected_threshold,
        output_path=output_dir / "p1_stage8_misclassification_report.md",
    )
    generate_validation_report(
        df_train,
        oof_df,
        threshold_df,
        selection,
        fold_metrics_df,
        fold_summary_df,
        fp_df,
        fn_df,
        summary_df,
        confusion_df,
        output_path=output_dir / "p1_stage8_validation_report.md",
    )

    # Required figures
    generate_stage8_figures(oof_df, threshold_df, selected_threshold, fold_metrics_df, output_dir=figures_dir)

    return {
        "best_config": best_config,
        "oof_predictions": oof_out,
        "threshold_analysis": threshold_df,
        "selection": selection,
        "fold_metrics": fold_metrics_df,
        "fold_summary": fold_summary_df,
        "false_positives": fp_df,
        "false_negatives": fn_df,
        "misclassification_summary": summary_df,
        "confusion_matrix": confusion_df,
    }
