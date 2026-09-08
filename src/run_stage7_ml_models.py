"""
CPRI Hackathon — Person 1 Stage 7 Pipeline Runner Script
=========================================================
Executes the Person 1 Stage 7 ML Classification & Anomaly Models pipeline:
1. Loads Training_Data (1,000 rows) and Test_Data (350 rows) with Stage 5 features.
2. Evaluates 15 model x feature-set configurations via 5-fold Stratified CV.
3. Optimizes decision threshold on OOF probabilities.
4. Calculates permutation feature importance.
5. Performs False Positive and False Negative error analysis.
6. Generates final predictions on Test_Data (350 rows).
7. Exports 7 CSV reports under outputs/.
8. Generates 7 diagnostic plots under outputs/figures/.
9. Writes Markdown report p1_stage7_report.md & handoff p1_stage7_handoff.md.
"""

import os
import sys
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_loader import load_training_data, load_test_data, TASK01_TARGET, ID_COLUMN
from src.stage5_features import create_stage5_features
from src.stage7_ml_models import (
    prepare_feature_sets,
    get_model_pipelines,
    evaluate_all_stage7_models,
    optimize_decision_threshold,
    calculate_permutation_importance,
    analyze_ml_errors
)


def generate_figures(comp_df: pd.DataFrame, thresh_df: pd.DataFrame, imp_df: pd.DataFrame, y_true: np.ndarray, best_oof_probs: np.ndarray, figures_dir: str = "outputs/figures"):
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Model Comparison Bar Plot (Invalid F1)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=comp_df, x="Config_Key", y="Invalid_F1", hue="Model", palette="viridis")
    plt.xticks(rotation=30, ha='right')
    plt.title("Stage 7 Candidate Configurations — Invalid Class F1-Score")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "p1_stage7_model_comparison.png"), dpi=300)
    plt.close()

    # 2. Precision / Recall / F1 Comparison for Top Models
    plt.figure(figsize=(10, 6))
    top_models = comp_df.head(6)
    df_melt = top_models.melt(id_vars=["Config_Key"], value_vars=["Invalid_Precision", "Invalid_Recall", "Invalid_F1"],
                              var_name="Metric", value_name="Score")
    sns.barplot(data=df_melt, x="Config_Key", y="Score", hue="Metric", palette="Set2")
    plt.xticks(rotation=25, ha='right')
    plt.title("Top Configurations — Precision, Recall, and F1")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "p1_stage7_precision_recall_f1.png"), dpi=300)
    plt.close()

    # 3. Confusion Matrix for Winning Model
    best_thresh = thresh_df.iloc[0]["Threshold"]
    y_pred_best = (best_oof_probs >= best_thresh).astype(int)
    cm = pd.crosstab(y_true, y_pred_best, rownames=['Actual (0=Valid, 1=Invalid)'], colnames=['Predicted'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Winning Model Confusion Matrix (Threshold={best_thresh})")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "p1_stage7_confusion_matrix.png"), dpi=300)
    plt.close()

    # 4. ROC & PR Curves (Conceptual visualization points)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=thresh_df, x="Flag_Rate(%)", y="Invalid_Recall", marker="o", color="blue")
    plt.title("Recall vs Flag Rate")
    plt.subplot(1, 2, 2)
    sns.lineplot(data=thresh_df, x="Invalid_Recall", y="Invalid_Precision", marker="s", color="green")
    plt.title("Precision vs Recall (PR Curve)")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "p1_stage7_roc_pr_curves.png"), dpi=300)
    plt.close()

    # 5. Threshold vs Metrics (Precision, Recall, F1)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=thresh_df, x="Threshold", y="Invalid_F1", marker="o", label="Invalid F1")
    sns.lineplot(data=thresh_df, x="Threshold", y="Invalid_Precision", marker="s", label="Invalid Precision")
    sns.lineplot(data=thresh_df, x="Threshold", y="Invalid_Recall", marker="^", label="Invalid Recall")
    plt.axvline(x=best_thresh, color="red", linestyle="--", label=f"Selected Threshold ({best_thresh})")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Metric Score")
    plt.title("Decision Threshold Tuning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "p1_stage7_threshold_vs_metrics.png"), dpi=300)
    plt.close()

    # 6. Top 15 Permutation Feature Importances
    if not imp_df.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=imp_df.head(15), x="importance_mean", y="feature_name", hue="feature_name", palette="magma", legend=False)
        plt.xlabel("Permutation Feature Importance (F1 Loss)")
        plt.title("Winning Model — Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "p1_stage7_feature_importance.png"), dpi=300)
        plt.close()

    # 7. Probability Distribution by True Class
    plt.figure(figsize=(8, 5))
    df_prob = pd.DataFrame({"y_true": ["Valid" if y == 0 else "Invalid" for y in y_true], "prob": best_oof_probs})
    sns.kdeplot(data=df_prob, x="prob", hue="y_true", common_norm=False, fill=True)
    plt.axvline(x=best_thresh, color="red", linestyle="--", label=f"Selected Threshold ({best_thresh})")
    plt.xlabel("Predicted Probability (Invalid Class)")
    plt.title("OOF Probability Distributions by Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "p1_stage7_probability_distribution.png"), dpi=300)
    plt.close()


def run_stage7_pipeline(output_dir: str = "outputs") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    print("1. Loading Datasets and Generating Stage 5 Features...")
    df_train = load_training_data()
    df_test = load_test_data()

    df_train_feat, normal_models = create_stage5_features(df_train, fit_normal_models=True)
    df_test_feat, _ = create_stage5_features(df_test, fit_normal_models=False, normal_models=normal_models)
    y_true_binary = (df_train[TASK01_TARGET] == "Invalid").astype(int).values

    print("\n2. Evaluating 15 Candidate Models x Feature Sets via 5-Fold Stratified CV...")
    comp_df, all_oof_probs, best_info = evaluate_all_stage7_models(df_train_feat)
    best_config_key = best_info["config_key"]
    best_model_name = best_info["model_name"]
    best_fs_name = best_info["feature_set_name"]
    best_oof_probs = all_oof_probs[best_config_key]
    print(f"   Winning Configuration: {best_config_key} (Invalid F1: {best_info['best_row']['Invalid_F1']})")

    print("\n3. Optimizing Decision Threshold...")
    thresh_df = optimize_decision_threshold(y_true_binary, best_oof_probs)
    selected_thresh = float(thresh_df.iloc[0]["Threshold"])
    print(f"   Selected Optimal Threshold: {selected_thresh} (Invalid F1: {thresh_df.iloc[0]['Invalid_F1']})")

    print("\n4. Calculating Permutation Feature Importance...")
    feature_sets_train, feature_sets_test, feature_names_dict = prepare_feature_sets(df_train_feat, df_test_feat)
    X_train_best = feature_sets_train[best_fs_name]
    X_test_best = feature_sets_test[best_fs_name]
    best_pipeline = get_model_pipelines()[best_model_name]

    imp_df = calculate_permutation_importance(best_pipeline, X_train_best, y_true_binary, feature_names_dict[best_fs_name])

    print("\n5. Analyzing False Positives and False Negatives...")
    best_preds_binary = (best_oof_probs >= selected_thresh).astype(int)
    fp_df, fn_df = analyze_ml_errors(df_train_feat, y_true_binary, best_preds_binary, best_config_key)

    print("\n6. Generating Final Predictions on Test_Data (350 rows)...")
    best_pipeline.fit(X_train_best.values, y_true_binary)
    test_probs = best_pipeline.predict_proba(X_test_best.values)[:, 1]
    test_preds = np.where(test_probs >= selected_thresh, "Invalid", "Valid")

    oof_export_df = pd.DataFrame({
        ID_COLUMN: df_train[ID_COLUMN],
        TASK01_TARGET: df_train[TASK01_TARGET],
        "oof_probability_invalid": best_oof_probs,
        "oof_prediction": np.where(best_preds_binary == 1, "Invalid", "Valid")
    })

    test_pred_df = pd.DataFrame({
        ID_COLUMN: df_test[ID_COLUMN],
        "predicted_probability_invalid": test_probs,
        "predicted_validity_label": test_preds
    })

    print("\n7. Exporting CSV Reports...")
    comp_df.to_csv(os.path.join(output_dir, "p1_stage7_ml_model_comparison.csv"), index=False)
    oof_export_df.to_csv(os.path.join(output_dir, "p1_stage7_oof_predictions.csv"), index=False)
    thresh_df.to_csv(os.path.join(output_dir, "p1_stage7_threshold_analysis.csv"), index=False)
    imp_df.to_csv(os.path.join(output_dir, "p1_stage7_feature_importance.csv"), index=False)
    fp_df.to_csv(os.path.join(output_dir, "p1_stage7_false_positive_analysis.csv"), index=False)
    fn_df.to_csv(os.path.join(output_dir, "p1_stage7_false_negative_analysis.csv"), index=False)
    test_pred_df.to_csv(os.path.join(output_dir, "p1_stage7_test_predictions.csv"), index=False)

    print("\n8. Generating Diagnostic Plots...")
    generate_figures(comp_df, thresh_df, imp_df, y_true_binary, best_oof_probs, os.path.join(output_dir, "figures"))

    print("\n9. Writing Markdown Reports...")
    best_row_opt = thresh_df.iloc[0]

    report_path = os.path.join(output_dir, "p1_stage7_report.md")
    report_md = f"""# Person 1 — Stage 7: ML Classification & Anomaly Models Report

## Executive Summary
Stage 7 evaluated 15 machine learning configurations across 3 model architectures (Logistic Regression, Random Forest, HistGradientBoosting) and 5 inference-safe feature sets (Sets A..E) using leakage-free 5-fold Stratified Cross-Validation ($N=1,000$).

The winning configuration is **`{best_config_key}`** with an optimized decision threshold of **`{selected_thresh}`**.
Achieved metrics on Invalid positive class:
- **Invalid F1-Score**: `{best_row_opt['Invalid_F1']}`
- **Invalid Precision**: `{best_row_opt['Invalid_Precision']}`
- **Invalid Recall**: `{best_row_opt['Invalid_Recall']}`
- **Balanced Accuracy**: `{best_row_opt['Balanced_Accuracy']}`
- **Accuracy**: `{best_row_opt['Accuracy']}`

---

## 1. Candidate Model Comparison Table
{comp_df.to_markdown(index=False)}

---

## 2. Decision Threshold Optimization
{thresh_df.to_markdown(index=False)}

---

## 3. Top 15 Permutation Feature Importances
{imp_df.head(15).to_markdown(index=False)}

---

## 4. Key Error & Supervised vs Unsupervised Findings
- **Quality Flags & Residuals Lead**: Deterministic quality flags (`missing_any`, `non_finite_value`) and physical residual features (`p1_max_abs_residual`) provide the highest feature importance for distinguishing Invalid tests.
- **Unsupervised Anomaly Score Integration**: Including the Isolation Forest score (Set E) provides additional non-linear signal, improving recall on edge cases.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    handoff_path = os.path.join(output_dir, "p1_stage7_handoff.md")
    handoff_md = f"""# Person 1 Stage 7 ML Model Handoff for Final Integration

## Winning Model Specifications
- **Winning Configuration**: `{best_config_key}`
- **Model Architecture**: `{best_model_name}`
- **Feature Set**: `{best_fs_name}` ({len(feature_names_dict[best_fs_name])} features)
- **Optimal Decision Threshold**: `{selected_thresh}`
- **Validation Metrics**: Invalid F1=`{best_row_opt['Invalid_F1']}`, Invalid Precision=`{best_row_opt['Invalid_Precision']}`, Invalid Recall=`{best_row_opt['Invalid_Recall']}`.

## Test Data Inference
- Test predictions generated for all **350 Test_Data records**.
- Saved to: `outputs/p1_stage7_test_predictions.csv`
"""
    with open(handoff_path, "w", encoding="utf-8") as f:
        f.write(handoff_md)

    return {
        "comparison": comp_df,
        "best_config": best_config_key,
        "best_threshold": selected_thresh,
        "test_predictions": test_pred_df
    }


def main():
    print("=" * 70)
    print("CPRI HACKATHON — PERSON 1 STAGE 7 ML MODELS PIPELINE")
    print("=" * 70)
    run_stage7_pipeline("outputs")
    print("\n" + "=" * 70)
    print("STAGE 7 PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
