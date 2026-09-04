"""
CPRI Hackathon — Task 01 — Person 2
P2-STAGE 3 — UNSUPERVISED ANOMALY DETECTION MODEL MODULE

Implements Isolation Forest and Local Outlier Factor (LOF) anomaly detectors
with standardized score directions (HIGHER score = MORE anomalous).

Guarantees:
- Model fitting operates strictly without access to Validity_Label.
- Score direction is explicitly standardized: HIGHER anomaly_score = MORE anomalous.
- Contamination is set to "auto" for primary baseline (no fitting to historical Invalid ratio).
- Preprocessing (SimpleImputer, StandardScaler) fitted ONLY on training feature data.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)


class IsolationForestAnomalyDetector:
    """
    Unsupervised Isolation Forest Anomaly Detector.
    Standardizes score direction: HIGHER anomaly_score = MORE anomalous.
    """
    def __init__(self, n_estimators: int = 200, contamination: str = "auto", random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        
        self.imputer = SimpleImputer(strategy="median")
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        X_imp = self.imputer.fit_transform(X)
        self.model.fit(X_imp)
        self.is_fitted = True
        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes continuous anomaly score where HIGHER = MORE anomalous.
        Inverts sklearn score_samples (which returns negative values for anomalies).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling score_samples!")
        X_imp = self.imputer.transform(X)
        raw_scores = self.model.score_samples(X_imp)
        # Sklearn returns lower (more negative) values for anomalies.
        # Negating ensures HIGHER score = MORE anomalous.
        return -raw_scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts binary anomaly flag: 1 = Anomalous, 0 = Normal.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict!")
        X_imp = self.imputer.transform(X)
        preds = self.model.predict(X_imp)
        # Sklearn predict returns -1 for outlier, 1 for inlier.
        # Map: -1 -> 1 (Anomalous), 1 -> 0 (Normal).
        return np.where(preds == -1, 1, 0)


class LOFAnomalyDetector:
    """
    Unsupervised Local Outlier Factor (LOF) Anomaly Detector with Novelty=True.
    Standardizes score direction: HIGHER anomaly_score = MORE anomalous.
    """
    def __init__(self, n_neighbors: int = 20, contamination: str = "auto"):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lof", LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination,
                novelty=True
            ))
        ])
        self.is_fitted = False

    def fit(self, X: pd.DataFrame):
        self.pipeline.fit(X)
        self.is_fitted = True
        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling score_samples!")
        # Access fitted LOF step inside pipeline
        imputer = self.pipeline.named_steps["imputer"]
        scaler = self.pipeline.named_steps["scaler"]
        lof = self.pipeline.named_steps["lof"]
        
        X_trans = scaler.transform(imputer.transform(X))
        raw_scores = lof.score_samples(X_trans)
        # Sklearn score_samples returns opposite direction -> negate it
        return -raw_scores

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict!")
        preds = self.pipeline.predict(X)
        return np.where(preds == -1, 1, 0)


def evaluate_unsupervised_anomaly(
    scores: np.ndarray,
    flags: np.ndarray,
    y_true: pd.Series,
    model_name: str,
    feature_set_name: str
) -> Dict[str, Any]:
    """
    Performs POST-HOC evaluation of unsupervised anomaly scores and flags
    against historical engineer Validity_Label (Invalid = 1 as positive target class).
    
    NOTE: These metrics are for post-hoc evaluation ONLY and are not used during model fitting.
    """
    roc_auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    
    prec = precision_score(y_true, flags, pos_label=1, zero_division=0)
    rec = recall_score(y_true, flags, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, flags, pos_label=1, zero_division=0)
    acc = accuracy_score(y_true, flags)
    bal_acc = balanced_accuracy_score(y_true, flags)
    cm = confusion_matrix(y_true, flags)
    
    num_flagged = int(np.sum(flags))
    pct_flagged = (num_flagged / len(flags)) * 100.0
    
    return {
        "model_name": model_name,
        "feature_set": feature_set_name,
        "num_flagged": num_flagged,
        "pct_flagged": pct_flagged,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "invalid_precision": float(prec),
        "invalid_recall": float(rec),
        "invalid_f1": float(f1),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "confusion_matrix": cm
    }
