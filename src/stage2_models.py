"""
CPRI Hackathon — Task 01 — Person 2
P2-STAGE 2 — MODEL PIPELINE & STRATIFIED CROSS-VALIDATION MODULE

Defines rule-based baselines, leakage-free sklearn Pipelines for Logistic Regression,
Random Forest, and Gradient Boosting, and executes 5-fold Stratified Cross-Validation.

Guarantees:
- Preprocessing (SimpleImputer, StandardScaler) strictly inside CV fold pipelines.
- StratifiedKFold (n_splits=5, shuffle=True, random_state=42) across identical fold splits.
- Invalid (1) is the positive target class.
- Evaluates Invalid Precision, Invalid Recall, Invalid F1, Macro F1, Accuracy, Balanced Accuracy.
- Generates complete out-of-fold (OOF) predictions and confusion matrices.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)


class RuleBasedClassifier:
    """
    Experimental rule-based classifier utilizing Stage 1 quality flags.
    Evaluated against historical engineer-verified Validity_Label.
    """
    def __init__(self, rule_type: str = "definite_issue"):
        self.rule_type = rule_type
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.rule_type == "definite_issue":
            if "data_quality_issue" in X.columns:
                return X["data_quality_issue"].astype(int).to_numpy()
            return np.zeros(len(X), dtype=int)
        elif self.rule_type == "definite_plus_duplicate":
            issue = X["data_quality_issue"] if "data_quality_issue" in X.columns else pd.Series(False, index=X.index)
            dup = X["duplicate_measurement_pair"] if "duplicate_measurement_pair" in X.columns else pd.Series(False, index=X.index)
            return (issue | dup).astype(int).to_numpy()
        else:
            raise ValueError(f"Unknown rule_type: {self.rule_type}")
            
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.predict(X)
        # Probability vector [1-p, p]
        probs = np.zeros((len(preds), 2))
        probs[:, 1] = preds.astype(float)
        probs[:, 0] = 1.0 - probs[:, 1]
        return probs


def get_model_pipeline(model_name: str, config: str = "default", random_state: int = 42) -> Pipeline:
    """
    Constructs leakage-free sklearn Pipelines with imputation, scaling, and classification steps.
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
            ("classifier", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state))
        ])
        
    elif model_name == "gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(random_state=random_state))
        ])
        
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def evaluate_rule_baseline(X: pd.DataFrame, y: pd.Series, rule_type: str) -> Dict[str, Any]:
    """
    Evaluates rule-based baseline on the complete dataset (rules require no statistical fitting).
    """
    clf = RuleBasedClassifier(rule_type=rule_type)
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    
    prec = precision_score(y, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y, y_pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    rule_name = "Rule-based (Definite Quality Issue)" if rule_type == "definite_issue" else "Rule-based (Definite Issue + Duplicate Pair)"
    
    return {
        "model_name": rule_name,
        "feature_set": "Stage 1 Quality Flags",
        "invalid_precision": prec,
        "invalid_recall": rec,
        "invalid_f1": f1,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "invalid_f1_std": 0.0,
        "confusion_matrix": cm,
        "oof_preds": y_pred,
        "oof_probs": y_prob
    }


def evaluate_model_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    feature_set_label: str,
    config: str = "default",
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes 5-fold Stratified Cross-Validation on a model pipeline.
    
    Guarantees strict leakage prevention: imputer & scaler fitted ONLY on training portion of each fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros(len(y), dtype=float)
    oof_folds = np.zeros(len(y), dtype=int)
    
    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Instantiate fresh pipeline for each fold
        pipeline = get_model_pipeline(model_name=model_name, config=config, random_state=random_state)
        
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
    cm = confusion_matrix(y, oof_preds)
    f1_std = float(np.std(fold_f1s))
    
    display_model_name = {
        "logistic_regression": "Logistic Regression (Balanced)" if config == "balanced" else "Logistic Regression (Unweighted)",
        "random_forest": "Random Forest (Balanced)",
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
        "invalid_f1_std": f1_std,
        "confusion_matrix": cm,
        "oof_preds": oof_preds,
        "oof_probs": oof_probs,
        "oof_folds": oof_folds
    }
