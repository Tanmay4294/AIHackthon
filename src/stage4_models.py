"""
CPRI Hackathon — Task 01 — Person 2
P2-STAGE 4 — MODEL PIPELINE & LEAKAGE-FREE CROSS-VALIDATION MODULE

Defines fold-isolated anomaly transformers, leakage-free sklearn Pipelines for candidate models,
and executes 5-fold Stratified Cross-Validation on identical fold splits.

Guarantees:
- Preprocessing (imputation, scaling, anomaly feature extraction) occurs strictly inside each CV fold.
- StratifiedKFold (n_splits=5, shuffle=True, random_state=42) across identical fold splits.
- Invalid (1) is the positive target class.
- Computes Invalid Precision, Invalid Recall, Invalid F1, Macro F1, Accuracy, Balanced Accuracy, ROC-AUC, PR-AUC, and Confusion Matrix.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import IsolationForest
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


class FoldIsolatedAnomalyTransformer(BaseEstimator, TransformerMixin):
    """
    Fold-Isolated Anomaly Transformer.
    Fits IsolationForest strictly on the training portion of each CV fold during fit(),
    and appends normalized anomaly score (-score_samples) as a feature during transform().
    """
    def __init__(self, n_estimators: int = 100, contamination: str = "auto", random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.iforest = None
        
    def fit(self, X, y=None):
        self.iforest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state
        )
        # Convert X to array if DataFrame
        X_arr = np.asarray(X)
        self.iforest.fit(X_arr)
        return self
        
    def transform(self, X):
        X_arr = np.asarray(X)
        raw_scores = self.iforest.score_samples(X_arr)
        anom_col = (-raw_scores).reshape(-1, 1)
        return np.hstack([X_arr, anom_col])


def get_stage4_pipeline(model_name: str, config: str = "default", random_state: int = 42) -> Pipeline:
    """
    Constructs leakage-free sklearn Pipelines with imputation, optional scaling / anomaly transformer,
    and classification steps.
    """
    if model_name == "logistic_regression":
        if config == "balanced":
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000))
            ])
        else:
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(random_state=random_state, max_iter=1000))
            ])
            
    elif model_name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state))
        ])
        
    elif model_name == "random_forest_anomaly":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("anomaly_transformer", FoldIsolatedAnomalyTransformer(n_estimators=100, random_state=random_state)),
            ("classifier", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=random_state))
        ])
        
    elif model_name == "gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(random_state=random_state))
        ])
        
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def evaluate_stage4_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    feature_set_label: str,
    config: str = "default",
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes 5-fold Stratified Cross-Validation on a candidate model pipeline.
    Guarantees strict fold isolation for all learned preprocessing and models.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros(len(y), dtype=float)
    oof_folds = np.zeros(len(y), dtype=int)
    
    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline = get_stage4_pipeline(model_name=model_name, config=config, random_state=random_state)
        
        # Fit inside fold
        pipeline.fit(X_train, y_train)
        
        # Predict on validation fold
        val_preds = pipeline.predict(X_val)
        val_probs = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else val_preds.astype(float)
        
        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs
        oof_folds[val_idx] = fold + 1
        
        fold_f1 = f1_score(y_val, val_preds, pos_label=1, zero_division=0)
        fold_f1s.append(fold_f1)
        
    # Aggregate OOF metrics
    prec = precision_score(y, oof_preds, pos_label=1, zero_division=0)
    rec = recall_score(y, oof_preds, pos_label=1, zero_division=0)
    f1 = f1_score(y, oof_preds, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y, oof_preds, average="macro", zero_division=0)
    acc = accuracy_score(y, oof_preds)
    bal_acc = balanced_accuracy_score(y, oof_preds)
    roc_auc = roc_auc_score(y, oof_probs)
    pr_auc = average_precision_score(y, oof_probs)
    cm = confusion_matrix(y, oof_preds)
    f1_std = float(np.std(fold_f1s))
    
    display_model_name = {
        "logistic_regression": "Logistic Regression (Balanced)" if config == "balanced" else "Logistic Regression (Unweighted)",
        "random_forest": "Random Forest (Balanced)",
        "random_forest_anomaly": "Random Forest + Fold Anomaly Score",
        "gradient_boosting": "Gradient Boosting"
    }.get(model_name, model_name)
    
    return {
        "model_name": display_model_name,
        "feature_set": feature_set_label,
        "invalid_precision": prec,
        "invalid_recall": rec,
        "invalid_f1": f1,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "invalid_f1_std": f1_std,
        "confusion_matrix": cm,
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
        "oof_folds": oof_folds
    }
