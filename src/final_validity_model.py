"""
CPRI Hackathon — Task 01 — Person 2
P2-STAGE 4 — FINAL VALIDITY CLASSIFIER INFERENCE MODULE

Provides the final reproducible FinalValidityClassifier class encapsulating feature selection,
pipeline model training, decision threshold application, and inference for new datasets.

Guarantees:
- Works on Test_Data without requiring Validity_Label or Reference_Parameter.
- Applies optimal decision threshold tuned during Stage 4 cross-validation.
- Reproducible execution with random_state=42.
"""

from typing import Dict, Any, List, Union
import numpy as np
import pandas as pd

from stage4_models import get_stage4_pipeline
from person1_adapter import get_person1_features
from quality_features import create_quality_flags, FEATURE_COLUMNS
from stage2_features import RAW_INPUT_COLUMNS, detect_constant_features


class FinalValidityClassifier:
    """
    Final Reproducible Validity Classifier for Task 01.
    Encapsulates feature extraction, pipeline estimator, and decision thresholding.
    """
    def __init__(
        self,
        model_name: str = "random_forest",
        feature_set_name: str = "Set_C",
        decision_threshold: float = 0.50,
        random_state: int = 42
    ):
        self.model_name = model_name
        self.feature_set_name = feature_set_name
        self.decision_threshold = decision_threshold
        self.random_state = random_state
        
        self.pipeline = None
        self.active_feature_names = []
        self.constant_feature_names = []
        self.is_fitted = False

    def _prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Extracts raw parameters, Stage 1 quality flags, and Person 1/fallback contextual features.
        Filters for active feature columns determined during fit().
        """
        df_flagged = create_quality_flags(df, feature_cols=FEATURE_COLUMNS)
        df_ctx, _, ctx_cols = get_person1_features(df_flagged)
        
        raw_cols = [c for c in RAW_INPUT_COLUMNS if c in df_flagged.columns]
        quality_cols = [
            "missing_any", "missing_critical_measurement", "non_finite_value",
            "duplicate_full_row", "duplicate_test_id", "duplicate_measurement_pair",
            "malformed_numeric", "invalid_voltage", "invalid_current",
            "invalid_duration", "invalid_ambient_temp", "sensor_s2_negative",
            "sensor_zero_reading", "data_quality_issue"
        ]
        avail_quality = [c for c in quality_cols if c in df_flagged.columns]
        
        df_combined = pd.concat([df_flagged[raw_cols], df_ctx, df_flagged[avail_quality]], axis=1)
        
        if is_training:
            active_cols, const_cols, _ = detect_constant_features(df_combined, list(df_combined.columns))
            self.constant_feature_names = const_cols
            
            if self.feature_set_name == "Set_A":
                self.active_feature_names = [c for c in (raw_cols + avail_quality) if c in active_cols]
            elif self.feature_set_name == "Set_B":
                self.active_feature_names = [c for c in (raw_cols + ctx_cols) if c in active_cols]
            elif self.feature_set_name == "Set_C":
                self.active_feature_names = [c for c in (raw_cols + avail_quality + ctx_cols) if c in active_cols]
            else:
                self.active_feature_names = [c for c in df_combined.columns if c in active_cols]
                
        return df_combined[self.active_feature_names].copy()

    def fit(self, df_train: pd.DataFrame, y_train: pd.Series = None):
        """
        Fits the final classifier pipeline on training data.
        """
        if y_train is None:
            if "Validity_Label" in df_train.columns:
                y_train = (df_train["Validity_Label"].str.strip() == "Invalid").astype(int)
            else:
                raise ValueError("Training data missing 'Validity_Label' target vector!")
                
        X_tr = self._prepare_features(df_train, is_training=True)
        self.pipeline = get_stage4_pipeline(self.model_name, config="balanced", random_state=self.random_state)
        self.pipeline.fit(X_tr, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts Invalid probability array [P(Valid), P(Invalid)].
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before calling predict_proba!")
        X_mat = self._prepare_features(df, is_training=False)
        return self.pipeline.predict_proba(X_mat)

    def predict_invalid_probability(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns 1D array of predicted Invalid probabilities P(Invalid).
        """
        return self.predict_proba(df)[:, 1]

    def predict(self, df: pd.DataFrame, return_str: bool = True) -> np.ndarray:
        """
        Predicts Valid/Invalid labels by applying decision_threshold to P(Invalid).
        """
        probs = self.predict_invalid_probability(df)
        preds_binary = (probs >= self.decision_threshold).astype(int)
        
        if return_str:
            return np.where(preds_binary == 1, "Invalid", "Valid")
        return preds_binary

    def predict_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        High-level inference API returning DataFrame with Test_ID, Probability_Invalid, and Predicted_Validity.
        """
        test_ids = df["Test_ID"] if "Test_ID" in df.columns else pd.Series(range(len(df)))
        probs = self.predict_invalid_probability(df)
        preds_str = self.predict(df, return_str=True)
        
        return pd.DataFrame({
            "Test_ID": test_ids,
            "Probability_Invalid": np.round(probs, 4),
            "Predicted_Validity": preds_str
        })
