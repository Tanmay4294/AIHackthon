"""Final Person 1 detector for Task 01.

The detector deliberately keeps the validated Stage 7/8 Random Forest as the
official decision source.  Stage 9 evaluates quality, physical consistency,
and unsupervised evidence as OOF alternatives, but does not allow weak soft
signals to override legitimate unusual operating regimes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
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

from src.dataset_loader import ID_COLUMN, PHYSICAL_FEATURES, TASK01_TARGET, load_test_data, load_training_data
from src.stage5_features import create_stage5_features, prepare_stage5_feature_matrix
from src.stage7_ml_models import get_model_pipelines
from src.stage8_validation import (
    DEFAULT_FEATURE_SET_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_RANDOM_STATE,
    generate_oof_predictions,
    load_stage7_best_configuration,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKED_THRESHOLD = 0.40
HARD_QUALITY_COLUMNS = [
    "missing_critical_measurement",
    "non_finite_value",
    "invalid_voltage",
    "invalid_current",
    "invalid_duration",
    "invalid_ambient_temp",
    "malformed_numeric",
    "sensor_s2_negative",
    "sensor_zero_reading",
    "duplicate_measurement_pair",
]
SOFT_EVIDENCE_COLUMNS = [
    "p1_max_abs_residual",
    "p1_mean_abs_residual",
    "p1_consistency_index",
    "p1_sensor_disagreement_index",
]


def _binary_labels(values: pd.Series) -> np.ndarray:
    return (values.astype(str).str.strip() == "Invalid").astype(int).to_numpy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Invalid_Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Invalid_Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "Invalid_F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "Balanced_Accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "ROC_AUC": round(float(roc_auc_score(y_true, scores)), 4),
        "PR_AUC": round(float(average_precision_score(y_true, scores)), 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def _hard_quality_mask(df: pd.DataFrame) -> pd.Series:
    present = [c for c in HARD_QUALITY_COLUMNS if c in df.columns]
    if not present:
        return pd.Series(False, index=df.index)
    return df[present].fillna(False).astype(bool).any(axis=1)


def _physical_evidence_mask(df: pd.DataFrame) -> pd.Series:
    residual = pd.to_numeric(df.get("p1_max_abs_residual"), errors="coerce")
    consistency = pd.to_numeric(df.get("p1_consistency_index"), errors="coerce")
    disagreement = pd.to_numeric(df.get("p1_sensor_disagreement_index"), errors="coerce")
    return ((residual >= 10.0) | (consistency <= 0.10) | (disagreement >= 1.5)).fillna(False)


def build_oof_anomaly_flags(oof_df: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE) -> pd.Series:
    """Create fold-isolated Isolation Forest flags for Stage 9 comparison.

    The 95th percentile is calculated from each training fold, never from
    Test_Data and never from the validation fold being scored.
    """
    scores = np.full(len(oof_df), np.nan)
    raw = [c for c in PHYSICAL_FEATURES if c in oof_df.columns]
    for fold in sorted(oof_df["fold"].unique()):
        train_mask = oof_df["fold"] != fold
        val_mask = oof_df["fold"] == fold
        model = IsolationForest(n_estimators=200, contamination="auto", random_state=random_state)
        x_train = oof_df.loc[train_mask, raw].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        x_val = oof_df.loc[val_mask, raw].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        model.fit(x_train)
        train_scores = -model.score_samples(x_train)
        scores[val_mask.to_numpy()] = -model.score_samples(x_val)
        cutoff = float(np.quantile(train_scores, 0.95))
        if fold == sorted(oof_df["fold"].unique())[0]:
            flags = np.zeros(len(oof_df), dtype=bool)
        flags[val_mask.to_numpy()] = scores[val_mask.to_numpy()] >= cutoff
    return pd.Series(flags, index=oof_df.index, name="anomaly_flag")


def compare_hybrid_approaches(oof_df: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Compare the required hybrid candidates on historical OOF predictions."""
    y = _binary_labels(oof_df[TASK01_TARGET])
    probability = pd.to_numeric(oof_df["oof_probability_invalid"], errors="coerce").to_numpy()
    supervised = probability >= LOCKED_THRESHOLD
    quality = _hard_quality_mask(oof_df).to_numpy()
    consistency = _physical_evidence_mask(oof_df).to_numpy()
    anomaly = build_oof_anomaly_flags(oof_df, random_state=random_state).to_numpy()
    masks = {
        "Supervised_only": supervised,
        "Supervised_plus_quality": supervised | quality,
        "Supervised_plus_consistency": supervised | consistency,
        "Supervised_plus_anomaly": supervised | anomaly,
        "Supervised_plus_quality_consistency": supervised | quality | consistency,
        "Full_hybrid": supervised | quality | consistency | anomaly,
    }
    records = []
    for name, prediction in masks.items():
        records.append({"Approach": name, **_metrics(y, prediction.astype(int), probability),
                        "Quality_Evidence": name != "Supervised_only",
                        "Consistency_Evidence": "consistency" in name or name == "Full_hybrid",
                        "Anomaly_Evidence": "anomaly" in name or name == "Full_hybrid"})
    return pd.DataFrame(records)


class FinalHybridDetector:
    """Reusable production detector with the locked Stage 8 threshold."""

    def __init__(self, threshold: float = LOCKED_THRESHOLD, random_state: int = DEFAULT_RANDOM_STATE):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = float(threshold)
        self.random_state = int(random_state)
        self.model_name = DEFAULT_MODEL_NAME
        self.feature_set_name = DEFAULT_FEATURE_SET_NAME
        self.pipeline = None
        self.normal_models = None
        self.feature_names = []
        self.anomaly_model = None
        self.is_fitted = False

    def fit(self, training_df: pd.DataFrame) -> "FinalHybridDetector":
        if TASK01_TARGET not in training_df.columns:
            raise ValueError("training_df must contain Validity_Label")
        if ID_COLUMN not in training_df.columns:
            raise ValueError("training_df must contain Test_ID")
        config = load_stage7_best_configuration(training_df, prefer_cached=True)
        self.model_name = config.model_name
        self.feature_set_name = config.feature_set_name
        features, self.normal_models = create_stage5_features(training_df.copy(), fit_normal_models=True)
        self.feature_names = list(prepare_stage5_feature_matrix(features).columns)
        self.pipeline = clone(get_model_pipelines()[self.model_name])
        x = prepare_stage5_feature_matrix(features, feature_names=self.feature_names)
        y = _binary_labels(training_df[TASK01_TARGET])
        self.pipeline.fit(x, y)
        raw = training_df[[c for c in PHYSICAL_FEATURES if c in training_df.columns]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self.anomaly_model = IsolationForest(n_estimators=200, contamination="auto", random_state=self.random_state).fit(raw)
        self.is_fitted = True
        return self

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("fit must be called before inference")
        enriched, _ = create_stage5_features(df.copy(), fit_normal_models=False, normal_models=self.normal_models)
        missing = [c for c in self.feature_names if c not in enriched.columns]
        if missing:
            raise ValueError(f"Inference feature generation missing columns: {missing}")
        return enriched

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        enriched = self._features(df)
        x = prepare_stage5_feature_matrix(enriched, feature_names=self.feature_names)
        return np.asarray(self.pipeline.predict_proba(x)[:, 1], dtype=float)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(df)
        return np.where(probabilities >= self.threshold, "Invalid", "Valid")

    def predict_with_diagnostics(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = self._features(df)
        probability = self.predict_proba(df)
        model_prediction = probability >= self.threshold
        hard_quality = _hard_quality_mask(enriched).to_numpy()
        # Hard rules are a safety net only; they are already covered by the model
        # on historical OOF data and do not turn soft unusual regimes invalid.
        final_invalid = model_prediction | hard_quality
        raw = df[[c for c in PHYSICAL_FEATURES if c in df.columns]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        anomaly_score = -self.anomaly_model.score_samples(raw)
        evidence = np.where(hard_quality, "deterministic_corruption",
                            np.where(model_prediction, "supervised_probability", "model_valid"))
        result = pd.DataFrame({
            ID_COLUMN: df[ID_COLUMN].to_numpy() if ID_COLUMN in df.columns else np.arange(len(df)),
            "Invalid_Probability": probability,
            "Anomaly_Score": anomaly_score,
            "Model_Invalid": model_prediction,
            "Hard_Quality_Evidence": hard_quality,
            "Final_Validity_Label": np.where(final_invalid, "Invalid", "Valid"),
            "Reason_Evidence": evidence,
        }, index=df.index)
        for col in HARD_QUALITY_COLUMNS + SOFT_EVIDENCE_COLUMNS:
            if col in enriched.columns:
                result[col] = enriched[col].to_numpy()
        return result.reset_index(drop=True)


def fit_final_detector(training_df: Optional[pd.DataFrame] = None, **kwargs: Any) -> FinalHybridDetector:
    detector = FinalHybridDetector(**kwargs)
    return detector.fit(load_training_data() if training_df is None else training_df)

