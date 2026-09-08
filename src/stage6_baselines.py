"""
CPRI Hackathon — Person 1 Stage 6 Baseline Detectors Module
============================================================
Provides unified baseline anomaly detectors and evaluation frameworks for Task 01:
1. Deterministic Quality Baseline (rule-based)
2. Robust Statistical Baseline (IQR / Boxplot outliers)
3. Z-Score Statistical Baseline (|Z| > 3.0)
4. Unsupervised Baseline (Isolation Forest & LOF reused from Stage 3)
5. Residual / Consistency Baseline (Stage 4 residual thresholds)

Guarantees:
- Zero target leakage in detector fitting / rule definitions.
- Evaluation metrics computed consistently for Invalid = positive class.
- False Positive (FP) and False Negative (FN) case analysis.
- Operating-regime performance breakdown across 4 regimes.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
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
from src.stage3_anomaly import IsolationForestAnomalyDetector, LOFAnomalyDetector
from src.stage4_behaviour import NormalBehaviourModels


def run_deterministic_quality_baseline(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """1. Deterministic Quality Baseline.

    Flags records with missing values, infinities, physical law violations, or negative/zero sensor anomalies.
    Returns:
        preds: Binary predictions (1 = Invalid/Flagged, 0 = Normal)
        scores: Continuous risk score (count of quality flags triggered)
    """
    quality_cols = [
        "missing_any", "non_finite_value", "invalid_voltage", "invalid_current",
        "invalid_duration", "invalid_ambient_temp", "sensor_s2_negative",
        "sensor_zero_reading", "malformed_numeric", "data_quality_issue"
    ]
    avail = [c for c in quality_cols if c in df.columns]

    if not avail:
        return np.zeros(len(df), dtype=int), np.zeros(len(df), dtype=float)

    flag_matrix = df[avail].astype(bool)
    scores = flag_matrix.sum(axis=1).values.astype(float)
    preds = (scores > 0).astype(int)
    return preds, scores


def run_iqr_statistical_baseline(df: pd.DataFrame, physical_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """2. Robust Statistical IQR Outlier Baseline.

    Flags records exceeding [Q1 - 1.5*IQR, Q3 + 1.5*IQR] on physical measurements.
    """
    if physical_cols is None:
        physical_cols = [c for c in PHYSICAL_FEATURES if c in df.columns]

    scores = np.zeros(len(df), dtype=float)
    for col in physical_cols:
        s = pd.to_numeric(df[col], errors='coerce')
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_mask = (s < lower) | (s > upper)
            scores += outlier_mask.astype(float)

    preds = (scores > 0).astype(int)
    return preds, scores


def run_zscore_statistical_baseline(df: pd.DataFrame, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """3. Z-Score Statistical Baseline.

    Flags records with max absolute Z-score > threshold across physical features.
    """
    cols = [c for c in PHYSICAL_FEATURES if c in df.columns]
    X_phys = df[cols].apply(pd.to_numeric, errors='coerce')

    means = X_phys.mean()
    stds = X_phys.std().replace(0, 1.0)

    z_scores = ((X_phys - means) / stds).abs()
    max_z = z_scores.max(axis=1).fillna(0.0).values

    preds = (max_z > threshold).astype(int)
    return preds, max_z


def run_unsupervised_baseline(df_train: pd.DataFrame, df_eval: pd.DataFrame, method: str = "isolation_forest") -> Tuple[np.ndarray, np.ndarray]:
    """4. Unsupervised Baseline (Isolation Forest or LOF) reusing Stage 3 implementation."""
    cols = [c for c in PHYSICAL_FEATURES if c in df_eval.columns]

    if method == "isolation_forest":
        detector = IsolationForestAnomalyDetector(n_estimators=200, contamination="auto", random_state=42)
        detector.fit(df_train[cols])
        scores = detector.score_samples(df_eval[cols])
        preds = detector.predict(df_eval[cols])
    elif method == "lof":
        detector = LOFAnomalyDetector(n_neighbors=20, contamination="auto")
        detector.fit(df_train[cols])
        scores = detector.score_samples(df_eval[cols])
        preds = detector.predict(df_eval[cols])
    else:
        raise ValueError(f"Unknown unsupervised method: {method}")

    return preds, scores


def run_residual_consistency_baseline(df: pd.DataFrame, max_res_threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """5. Residual / Physical Consistency Baseline.

    Flags records where p1_max_abs_residual > threshold or p1_consistency_index < 0.1.
    """
    max_res = pd.to_numeric(df.get("p1_max_abs_residual"), errors='coerce').fillna(0.0).values
    cons_idx = pd.to_numeric(df.get("p1_consistency_index"), errors='coerce').fillna(1.0).values

    scores = max_res.copy()
    preds = ((max_res > max_res_threshold) | (cons_idx < 0.1)).astype(int)
    return preds, scores


def evaluate_detector(y_true_binary: np.ndarray, y_pred_binary: np.ndarray, y_scores: Optional[np.ndarray] = None, method_name: str = "", feature_basis: str = "", threshold_desc: str = "") -> Dict[str, Any]:
    """Calculates standardized evaluation metrics for Invalid = positive class."""
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tot = len(y_true_binary)
    num_flagged = int(y_pred_binary.sum())
    flag_pct = round(num_flagged / tot * 100.0, 2) if tot > 0 else 0.0

    prec = round(float(precision_score(y_true_binary, y_pred_binary, zero_division=0)), 4)
    rec = round(float(recall_score(y_true_binary, y_pred_binary, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true_binary, y_pred_binary, zero_division=0)), 4)
    acc = round(float(accuracy_score(y_true_binary, y_pred_binary)), 4)
    b_acc = round(float(balanced_accuracy_score(y_true_binary, y_pred_binary)), 4)

    roc_auc = round(float(roc_auc_score(y_true_binary, y_scores)), 4) if y_scores is not None and len(np.unique(y_true_binary)) > 1 else np.nan
    pr_auc = round(float(average_precision_score(y_true_binary, y_scores)), 4) if y_scores is not None and len(np.unique(y_true_binary)) > 1 else np.nan

    return {
        "Method": method_name,
        "Feature_Basis": feature_basis,
        "Threshold": threshold_desc,
        "Num_Flagged": num_flagged,
        "Flag_Percentage(%)": flag_pct,
        "Invalid_Precision": prec,
        "Invalid_Recall": rec,
        "Invalid_F1": f1,
        "Balanced_Accuracy": b_acc,
        "Accuracy": acc,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }


def analyze_false_positives(df: pd.DataFrame, preds: np.ndarray, method_name: str) -> pd.DataFrame:
    """Analyzes False Positives (Historically Valid but flagged Invalid)."""
    if TASK01_TARGET not in df.columns:
        return pd.DataFrame()

    fp_mask = (df[TASK01_TARGET] == "Valid") & (preds == 1)
    df_fp = df[fp_mask].copy()

    records = []
    for idx, row in df_fp.head(10).iterrows():
        c = row.get("Load_Current_A", np.nan)
        v = row.get("Applied_Voltage_kV", np.nan)
        regime = "Heavy HV Load" if c > 85 and v > 25 else ("Heavy Current" if c > 85 else ("High Voltage" if v > 25 else "Standard"))

        records.append({
            "Method": method_name,
            "Test_ID": str(row.get(ID_COLUMN, f"Row_{idx}")),
            "Operating_Regime": regime,
            "Load_Current_A": c,
            "Applied_Voltage_kV": v,
            "p1_max_abs_residual": row.get("p1_max_abs_residual", np.nan),
            "Validity_Label": "Valid",
            "FP_Analysis": f"Historically Valid test flagged by {method_name}. Likely high operating condition or unusual sensor fluctuation."
        })

    return pd.DataFrame(records)


def analyze_false_negatives(df: pd.DataFrame, preds: np.ndarray, method_name: str) -> pd.DataFrame:
    """Analyzes False Negatives (Historically Invalid but predicted Valid)."""
    if TASK01_TARGET not in df.columns:
        return pd.DataFrame()

    fn_mask = (df[TASK01_TARGET] == "Invalid") & (preds == 0)
    df_fn = df[fn_mask].copy()

    records = []
    for idx, row in df_fn.head(10).iterrows():
        c = row.get("Load_Current_A", np.nan)
        v = row.get("Applied_Voltage_kV", np.nan)

        records.append({
            "Method": method_name,
            "Test_ID": str(row.get(ID_COLUMN, f"Row_{idx}")),
            "Load_Current_A": c,
            "Applied_Voltage_kV": v,
            "p1_max_abs_residual": row.get("p1_max_abs_residual", np.nan),
            "Validity_Label": "Invalid",
            "FN_Analysis": f"Historically Invalid test missed by {method_name}. Requires contextual/residual feature modeling."
        })

    return pd.DataFrame(records)


def evaluate_regime_performance(df: pd.DataFrame, preds_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Evaluates baseline detector performance broken down across operating regimes."""
    if TASK01_TARGET not in df.columns:
        return pd.DataFrame()

    y_true = (df[TASK01_TARGET] == "Invalid").astype(int).values

    def assign_regime(row):
        c, v = row.get("Load_Current_A", 0), row.get("Applied_Voltage_kV", 0)
        if c > 85 and v > 25: return "Heavy HV Load"
        elif c > 85: return "Heavy Current"
        elif v > 25: return "High Voltage"
        else: return "Standard"

    regimes = df.apply(assign_regime, axis=1)

    records = []
    for r_name, idxs in regimes.groupby(regimes).groups.items():
        sub_y_true = y_true[idxs]
        if len(sub_y_true) < 5: continue

        for m_name, preds in preds_dict.items():
            sub_preds = preds[idxs]
            rec = recall_score(sub_y_true, sub_preds, zero_division=0)
            prec = precision_score(sub_y_true, sub_preds, zero_division=0)
            f1 = f1_score(sub_y_true, sub_preds, zero_division=0)

            records.append({
                "Operating_Regime": r_name,
                "Regime_Record_Count": len(sub_y_true),
                "Regime_Invalid_Count": int(sub_y_true.sum()),
                "Method": m_name,
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1_Score": round(float(f1), 4)
            })

    return pd.DataFrame(records)
