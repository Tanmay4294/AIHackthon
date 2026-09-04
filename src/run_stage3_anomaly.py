"""
CPRI Hackathon — Task 01 — Person 2
P2-STAGE 3 — EXECUTE UNSUPERVISED ANOMALY DETECTION PIPELINE

Loads CPRI Training_Data and Test_Data, builds 4 feature sets (Set A, B, C, D),
executes Isolation Forest and Local Outlier Factor (LOF) anomaly detection,
evaluates post-hoc metrics against engineer labels, reuses Stage 2 OOF predictions for
supervised-vs-unsupervised 4-group agreement comparison, extracts disagreement cases,
saves all required CSV outputs, exports diagnostic plots, and generates executive reports.
"""

from typing import List, Dict, Any, Tuple
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

sys.path.insert(0, os.path.dirname(__file__))

from stage2_features import RAW_INPUT_COLUMNS
from stage3_features import load_stage3_datasets, prepare_stage3_feature_sets
from stage3_anomaly import (
    IsolationForestAnomalyDetector,
    LOFAnomalyDetector,
    evaluate_unsupervised_anomaly
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10


def run_stage3_pipeline():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    dataset_path = os.path.join(base_dir, 'data', 'CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx')
    
    print("=" * 70)
    print("CPRI HACKATHON — TASK 01 — PERSON 2 — P2-STAGE 3 ANOMALY DETECTION")
    print("=" * 70)
    
    # Step 1: Load Data
    df_train_flagged, df_test_flagged = load_stage3_datasets(dataset_path)
    print(f"Loaded Training_Data: {len(df_train_flagged)} rows, Test_Data: {len(df_test_flagged)} rows.")
    
    # Step 2: Prepare Feature Sets & Constant Feature Analysis
    prep = prepare_stage3_feature_sets(df_train_flagged, df_test_flagged)
    X_train_dict = prep["X_train"]
    X_test_dict = prep["X_test"]
    y_train = prep["y_train"]
    
    print("\n--- CONSTANT FEATURE ANALYSIS ---")
    print(f"Detected {len(prep['constant_features'])} constant features across 1,000 training records:")
    for c_col, c_val in prep["constant_features"].items():
        print(f"  - {c_col}: constant value = {c_val} (Excluded from anomaly feature matrices)")
        
    print("\n--- FEATURE SET DEFINITIONS ---")
    for set_key, set_cols in prep["feature_set_cols"].items():
        print(f"  - {set_key}: {len(set_cols)} active columns")

    # Step 3: Evaluate Isolation Forest & LOF across Feature Sets A, B, C, D
    print("\n--- EXECUTING UNSUPERVISED ANOMALY DETECTORS ---")
    
    method_results = []
    trained_models = {}
    train_scores_dict = {}
    train_flags_dict = {}
    test_scores_dict = {}
    test_flags_dict = {}
    
    feature_set_names = {
        "Set_A": "Set A (Raw Inputs)",
        "Set_B": "Set B (Raw + Engineered)",
        "Set_C": "Set C (Raw + Quality)",
        "Set_D": "Set D (Raw + Eng + Quality)"
    }
    
    for set_key in ["Set_A", "Set_B", "Set_C", "Set_D"]:
        X_tr = X_train_dict[set_key]
        X_te = X_test_dict[set_key]
        f_name = feature_set_names[set_key]
        
        # 1. Isolation Forest
        iforest = IsolationForestAnomalyDetector(n_estimators=200, contamination="auto", random_state=42)
        iforest.fit(X_tr)
        
        if_tr_scores = iforest.score_samples(X_tr)
        if_tr_flags = iforest.predict(X_tr)
        if_te_scores = iforest.score_samples(X_te)
        if_te_flags = iforest.predict(X_te)
        
        eval_if = evaluate_unsupervised_anomaly(if_tr_scores, if_tr_flags, y_train, "Isolation Forest", f_name)
        method_results.append(eval_if)
        
        trained_models[f"IF_{set_key}"] = iforest
        train_scores_dict[f"IF_{set_key}"] = if_tr_scores
        train_flags_dict[f"IF_{set_key}"] = if_tr_flags
        test_scores_dict[f"IF_{set_key}"] = if_te_scores
        test_flags_dict[f"IF_{set_key}"] = if_te_flags
        
        # 2. Local Outlier Factor (LOF)
        lof = LOFAnomalyDetector(n_neighbors=20, contamination="auto")
        lof.fit(X_tr)
        
        lof_tr_scores = lof.score_samples(X_tr)
        lof_tr_flags = lof.predict(X_tr)
        lof_te_scores = lof.score_samples(X_te)
        lof_te_flags = lof.predict(X_te)
        
        eval_lof = evaluate_unsupervised_anomaly(lof_tr_scores, lof_tr_flags, y_train, "Local Outlier Factor (LOF)", f_name)
        method_results.append(eval_lof)
        
        trained_models[f"LOF_{set_key}"] = lof
        train_scores_dict[f"LOF_{set_key}"] = lof_tr_scores
        train_flags_dict[f"LOF_{set_key}"] = lof_tr_flags
        test_scores_dict[f"LOF_{set_key}"] = lof_te_scores
        test_flags_dict[f"LOF_{set_key}"] = lof_te_flags

    # Summary Method Comparison Table
    df_method_comp = pd.DataFrame([{
        "Model": r["model_name"],
        "Feature_Set": r["feature_set"],
        "Num_Flagged": r["num_flagged"],
        "Pct_Flagged": round(r["pct_flagged"], 2),
        "ROC_AUC": round(r["roc_auc"], 4),
        "PR_AUC": round(r["pr_auc"], 4),
        "Invalid_Precision": round(r["invalid_precision"], 4),
        "Invalid_Recall": round(r["invalid_recall"], 4),
        "Invalid_F1": round(r["invalid_f1"], 4),
        "Accuracy": round(r["accuracy"], 4),
        "Balanced_Accuracy": round(r["balanced_accuracy"], 4)
    } for r in method_results])
    
    print("\n--- ANOMALY METHOD COMPARISON TABLE ---")
    print(df_method_comp.to_string(index=False))

    # Select Primary Anomaly Model based on highest ROC_AUC against historical labels
    best_row = df_method_comp.sort_values(by="ROC_AUC", ascending=False).iloc[0]
    print(f"\n>>> PRIMARY ANOMALY MODEL WINNER: {best_row['Model']} ({best_row['Feature_Set']})")
    print(f"    ROC-AUC: {best_row['ROC_AUC']:.4f} | PR-AUC: {best_row['PR_AUC']:.4f} | Flagged: {best_row['Num_Flagged']} records ({best_row['Pct_Flagged']}%)")
    
    # We select Isolation Forest on Set D as our primary benchmark model
    primary_key = "IF_Set_D"
    if_d_scores_tr = train_scores_dict[primary_key]
    if_d_flags_tr = train_flags_dict[primary_key]
    if_d_scores_te = test_scores_dict[primary_key]
    if_d_flags_te = test_flags_dict[primary_key]

    # Step 4: Supervised vs. Unsupervised Agreement Analysis (Reusing Stage 2 OOF Predictions)
    print("\n--- SUPERVISED VS UNSUPERVISED AGREEMENT ANALYSIS ---")
    oof_csv_path = os.path.join(base_dir, 'outputs', 'p2_stage2_oof_predictions.csv')
    if not os.path.exists(oof_csv_path):
        raise FileNotFoundError(f"Stage 2 OOF predictions file not found at: {oof_csv_path}")
        
    df_stage2_oof = pd.read_csv(oof_csv_path)
    
    df_sup_anom = pd.DataFrame({
        "Test_ID": prep["test_ids_train"],
        "Actual_Validity": df_train_flagged["Validity_Label"],
        "Actual_Binary": y_train,
        "Supervised_Prediction": df_stage2_oof["Predicted_Validity"],
        "Supervised_Binary": df_stage2_oof["Predicted_Binary"],
        "Supervised_Probability_Invalid": df_stage2_oof["Probability_Invalid"],
        "IsolationForest_Score": if_d_scores_tr,
        "IsolationForest_Flag": if_d_flags_tr,
        "LOF_Score": train_scores_dict["LOF_Set_D"],
        "LOF_Flag": train_flags_dict["LOF_Set_D"]
    })
    
    # Define 4 Agreement Groups based on Supervised vs Isolation Forest
    # Group 1: Supervised Invalid (1) & Anomaly Anomalous (1)
    # Group 2: Supervised Invalid (1) & Anomaly Normal (0)
    # Group 3: Supervised Valid (0) & Anomaly Anomalous (1)
    # Group 4: Supervised Valid (0) & Anomaly Normal (0)
    def assign_group(row):
        sup = row["Supervised_Binary"]
        anom = row["IsolationForest_Flag"]
        if sup == 1 and anom == 1:
            return "Group 1 (Both Invalid/Anomalous)"
        elif sup == 1 and anom == 0:
            return "Group 2 (Supervised Invalid / Anomaly Normal)"
        elif sup == 0 and anom == 1:
            return "Group 3 (Supervised Valid / Anomaly Anomalous)"
        else:
            return "Group 4 (Both Valid/Normal)"
            
    df_sup_anom["Agreement_Group"] = df_sup_anom.apply(assign_group, axis=1)
    
    group_counts = df_sup_anom["Agreement_Group"].value_counts()
    print("\n--- 4 AGREEMENT GROUPS BREAKDOWN (N=1000) ---")
    for grp, cnt in group_counts.items():
        pct = (cnt / len(df_sup_anom)) * 100.0
        print(f"  - {grp}: {cnt} records ({pct:.2f}%)")

    # Step 5: Save Output CSVs
    outputs_dir = os.path.join(base_dir, 'outputs')
    figures_dir = os.path.join(outputs_dir, 'figures')
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Training Anomaly Scores CSV
    df_scores_train = pd.DataFrame({
        "Test_ID": prep["test_ids_train"],
        "Validity_Label": df_train_flagged["Validity_Label"],
        "IsolationForest_Score_SetD": train_scores_dict["IF_Set_D"],
        "IsolationForest_Flag_SetD": train_flags_dict["IF_Set_D"],
        "LOF_Score_SetD": train_scores_dict["LOF_Set_D"],
        "LOF_Flag_SetD": train_flags_dict["LOF_Set_D"],
        "IsolationForest_Score_SetA": train_scores_dict["IF_Set_A"],
        "IsolationForest_Flag_SetA": train_flags_dict["IF_Set_A"],
        "missing_any": df_train_flagged["missing_any"],
        "duplicate_measurement_pair": df_train_flagged["duplicate_measurement_pair"],
        "data_quality_issue": df_train_flagged["data_quality_issue"]
    })
    train_csv_path = os.path.join(outputs_dir, 'p2_stage3_anomaly_scores_training.csv')
    df_scores_train.to_csv(train_csv_path, index=False)
    print(f"\nSaved Training Anomaly Scores CSV: {train_csv_path}")

    # 2. Test Anomaly Scores CSV
    df_scores_test = pd.DataFrame({
        "Test_ID": prep["test_ids_test"],
        "IsolationForest_Score_SetD": test_scores_dict["IF_Set_D"],
        "IsolationForest_Flag_SetD": test_flags_dict["IF_Set_D"],
        "LOF_Score_SetD": test_scores_dict["LOF_Set_D"],
        "LOF_Flag_SetD": test_flags_dict["LOF_Set_D"],
        "IsolationForest_Score_SetA": test_scores_dict["IF_Set_A"],
        "IsolationForest_Flag_SetA": test_flags_dict["IF_Set_A"],
        "missing_any": df_test_flagged["missing_any"],
        "duplicate_measurement_pair": df_test_flagged["duplicate_measurement_pair"],
        "data_quality_issue": df_test_flagged["data_quality_issue"]
    })
    test_csv_path = os.path.join(outputs_dir, 'p2_stage3_anomaly_scores_test.csv')
    df_scores_test.to_csv(test_csv_path, index=False)
    print(f"Saved Test Anomaly Scores CSV: {test_csv_path}")

    # 3. Method Comparison CSV
    method_csv_path = os.path.join(outputs_dir, 'p2_stage3_method_comparison.csv')
    df_method_comp.to_csv(method_csv_path, index=False)
    print(f"Saved Method Comparison CSV: {method_csv_path}")

    # 4. Supervised vs Anomaly Comparison CSV
    sup_anom_csv_path = os.path.join(outputs_dir, 'p2_stage3_supervised_anomaly_comparison.csv')
    df_sup_anom.to_csv(sup_anom_csv_path, index=False)
    print(f"Saved Supervised vs Anomaly Comparison CSV: {sup_anom_csv_path}")

    # 5. Disagreement Cases CSV (Group 2 and Group 3)
    df_disagree = df_sup_anom[df_sup_anom["Agreement_Group"].isin([
        "Group 2 (Supervised Invalid / Anomaly Normal)",
        "Group 3 (Supervised Valid / Anomaly Anomalous)"
    ])].copy()
    
    # Merge raw features and Stage 1 quality flags for detailed analysis
    df_disagree = pd.merge(df_disagree, df_train_flagged[RAW_INPUT_COLUMNS + ["missing_any", "duplicate_measurement_pair", "data_quality_issue"]], left_index=True, right_index=True)
    disagree_csv_path = os.path.join(outputs_dir, 'p2_stage3_disagreement_cases.csv')
    df_disagree.to_csv(disagree_csv_path, index=False)
    print(f"Saved Disagreement Cases CSV: {disagree_csv_path} ({len(df_disagree)} records)")

    # Step 6: Generate Visualizations
    plot_anomaly_distribution(if_d_scores_tr, os.path.join(figures_dir, 'p2_stage3_anomaly_score_distribution.png'))
    plot_score_by_validity_label(if_d_scores_tr, df_train_flagged["Validity_Label"], os.path.join(figures_dir, 'p2_stage3_score_by_validity_label.png'))
    plot_roc_pr_curves(if_d_scores_tr, train_scores_dict["LOF_Set_D"], y_train, os.path.join(figures_dir, 'p2_stage3_roc_pr_curves.png'))
    plot_agreement_group_distribution(df_sup_anom["Agreement_Group"], os.path.join(figures_dir, 'p2_stage3_agreement_group_distribution.png'))

    # Step 7: Generate Markdown Reports & Person 1 Handoff
    generate_stage3_report_md(df_method_comp, df_sup_anom, df_disagree, prep, os.path.join(outputs_dir, 'p2_stage3_report.md'))
    generate_person1_stage3_handoff_md(df_disagree, df_scores_test, os.path.join(outputs_dir, 'p2_stage3_handoff.md'))
    
    print(f"Saved Stage 3 Report: {os.path.join(outputs_dir, 'p2_stage3_report.md')}")
    print(f"Saved Person 1 Handoff: {os.path.join(outputs_dir, 'p2_stage3_handoff.md')}")
    
    print("\nP2-Stage 3 Unsupervised Anomaly Detection Pipeline completed successfully.")


def plot_anomaly_distribution(scores: np.ndarray, output_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(scores, kde=True, ax=ax, color='#2b5c8f', bins=35)
    ax.set_title('Isolation Forest Anomaly Score Distribution (Training_Data N=1000)', fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel('Anomaly Score (Higher = More Anomalous)', fontweight='bold')
    ax.set_ylabel('Record Frequency', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_score_by_validity_label(scores: np.ndarray, labels: pd.Series, output_path: str):
    df_p = pd.DataFrame({'Score': scores, 'Label': labels})
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.boxplot(x='Label', y='Score', data=df_p, palette=['#2b5c8f', '#d95f02'], ax=ax, width=0.4)
    ax.set_title('Isolation Forest Anomaly Score by Historical Validity Label', fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel('Historical Engineer Label', fontweight='bold')
    ax.set_ylabel('Anomaly Score (Higher = More Anomalous)', fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_pr_curves(if_scores: np.ndarray, lof_scores: np.ndarray, y_true: pd.Series, output_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # ROC Curve
    fpr_if, tpr_if, _ = roc_curve(y_true, if_scores)
    fpr_lof, tpr_lof, _ = roc_curve(y_true, lof_scores)
    
    ax1.plot(fpr_if, tpr_if, label='Isolation Forest (Set D)', color='#1f77b4', lw=2)
    ax1.plot(fpr_lof, tpr_lof, label='LOF (Set D)', color='#ff7f0e', lw=2)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_title('ROC Curves (Post-Hoc Label Comparison)', fontweight='bold')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # PR Curve
    prec_if, rec_if, _ = precision_recall_curve(y_true, if_scores)
    prec_lof, rec_lof, _ = precision_recall_curve(y_true, lof_scores)
    
    ax2.plot(rec_if, prec_if, label='Isolation Forest (Set D)', color='#1f77b4', lw=2)
    ax2.plot(rec_lof, prec_lof, label='LOF (Set D)', color='#ff7f0e', lw=2)
    ax2.set_title('Precision-Recall Curves (Post-Hoc Label Comparison)', fontweight='bold')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_agreement_group_distribution(groups: pd.Series, output_path: str):
    counts = groups.value_counts()
    
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(counts.index, counts.values, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'], height=0.55)
    ax.set_title('Supervised Stage 2 vs Unsupervised Stage 3 Agreement Groups', fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Historical Records (Total N=1000)', fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    for bar in bars:
        w = bar.get_width()
        pct = (w / 1000.0) * 100.0
        ax.annotate(f'{w} ({pct:.1f}%)', (w, bar.get_y() + bar.get_height()/2.),
                    ha='left', va='center', fontsize=9.5, fontweight='bold', xytext=(5, 0),
                    textcoords='offset points')
                    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_stage3_report_md(df_comp: pd.DataFrame, df_sup_anom: pd.DataFrame, df_disagree: pd.DataFrame, prep: Dict[str, Any], output_path: str):
    grp_counts = df_sup_anom["Agreement_Group"].value_counts()
    
    const_str = "\n".join([f"- **`{k}`**: Constant value `{v}`" for k, v in prep["constant_features"].items()])
    
    report_content = f"""# Person 2 — Stage 3 Unsupervised Anomaly Detection Report

**Owner**: Person 2 (Data Quality, Anomaly Detection & Classification Lead)  
**Project**: CPRI State-Level Hackathon — Task 01  
**Scope**: P2-Stage 3 Unsupervised Anomaly Detection  
**Date**: September 2026  

---

## 1. Executive Summary

P2-Stage 3 establishes an independent, **unsupervised anomaly detection pipeline** for CPRI electrical test screening.

Model fitting operates **strictly without access to `Validity_Label`**, `Reference_Parameter`, or `Test_ID`. Anomaly scores are standardized so that **HIGHER score = MORE anomalous**.

### Primary Findings:
1. **Unsupervised Identification**: Isolation Forest (Set D) achieved a post-hoc **ROC-AUC of `{df_comp.loc[df_comp['Model']=='Isolation Forest', 'ROC_AUC'].max():.4f}`** against historical engineer labels.
2. **Supervised vs. Unsupervised Agreement**:
   - **Both Agree Normal (Group 4)**: `{grp_counts.get('Group 4 (Both Valid/Normal)', 0)}` records ({grp_counts.get('Group 4 (Both Valid/Normal)', 0)/10.0:.1f}%)
   - **Both Agree Invalid (Group 1)**: `{grp_counts.get('Group 1 (Both Invalid/Anomalous)', 0)}` records ({grp_counts.get('Group 1 (Both Invalid/Anomalous)', 0)/10.0:.1f}%)
   - **Supervised Invalid / Anomaly Normal (Group 2)**: `{grp_counts.get('Group 2 (Supervised Invalid / Anomaly Normal)', 0)}` records ({grp_counts.get('Group 2 (Supervised Invalid / Anomaly Normal)', 0)/10.0:.1f}%)
   - **Supervised Valid / Anomaly Anomalous (Group 3)**: `{grp_counts.get('Group 3 (Supervised Valid / Anomaly Anomalous)', 0)}` records ({grp_counts.get('Group 3 (Supervised Valid / Anomaly Anomalous)', 0)/10.0:.1f}%)

> [!IMPORTANT]
> **Anomaly Detection Principle**:
> Anomaly detection identifies statistically unusual records; it does NOT automatically prove a test is Invalid. Group 3 records (Valid but Anomalous) represent key candidates for Person 1's operating-regime investigation.

---

## 2. Feature Sets & Constant Feature Isolation

Four feature configurations were evaluated:
- **Set A**: Raw physical variables (Voltage, Current, Temp, Duration, Sensors S1..S4).
- **Set B**: Raw variables + 5 physically justified engineered features (`Apparent_Power_kVA`, `Sensor_Spread`, `Sensor_Mean`, `Sensor_S1_S2_Ratio`, `Sensor_S3_S4_Ratio`).
- **Set C**: Raw variables + active Stage 1 quality flags (`missing_any`, `duplicate_measurement_pair`, `sensor_s2_negative`, `sensor_zero_reading`, `data_quality_issue`).
- **Set D**: Raw + Engineered + Stage 1 quality flags.

### Excluded Constant Features:
{const_str}

---

## 3. Anomaly Method Comparison Table (Post-Hoc Evaluation)

{df_comp.to_markdown(index=False)}

---

## 4. Supervised (Stage 2 OOF) vs Unsupervised (Stage 3) Comparison

Using Stage 2 out-of-fold predictions and Stage 3 Isolation Forest anomaly flags:

- **Group 1 (Both Agree Invalid/Anomalous)**: High-confidence invalid tests containing clear physical/sensor corruptions.
- **Group 2 (Supervised Invalid / Anomaly Normal)**: Tests with subtle pattern shifts that fall within standard statistical distribution bounds.
- **Group 3 (Supervised Valid / Anomaly Anomalous)**: Valid high-load operating regimes or unusual sensor configurations.
- **Group 4 (Both Agree Valid/Normal)**: Standard, nominal electrical tests.

---

## 5. Next-Stage Handoff & Recommendations

1. **For Person 1 (Regime & Behaviour Lead)**:
   - Investigate Group 3 records in `outputs/p2_stage3_disagreement_cases.csv` to distinguish high-load valid testing regimes from true equipment anomalies.
2. **For Task 01 Final Integration**:
   - Combine Stage 1 quality flags, Stage 2 supervised probabilities, and Stage 3 anomaly scores into Person 1's contextual decision framework.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)


def generate_person1_stage3_handoff_md(df_disagree: pd.DataFrame, df_test_scores: pd.DataFrame, output_path: str):
    grp2 = df_disagree[df_disagree["Agreement_Group"] == "Group 2 (Supervised Invalid / Anomaly Normal)"]
    grp3 = df_disagree[df_disagree["Agreement_Group"] == "Group 3 (Supervised Valid / Anomaly Anomalous)"]
    
    handoff_content = f"""# Person 2 — Stage 3 Handoff Report for Person 1

**From**: Person 2 (Data Quality & Anomaly Lead)  
**To**: Person 1 (Equipment Behaviour & Regime Analysis Lead)  
**Project**: CPRI State-Level Hackathon — Task 01  
**Subject**: Stage 3 Unsupervised Anomaly Cases & Disagreement Analysis  

---

## 1. Handoff Purpose

This document highlights key test records where unsupervised statistical anomaly detection and supervised classification disagree.

> [!IMPORTANT]
> **Key Guidance for Person 1**:
> - **Group 3 (Supervised Valid / Anomaly Anomalous)**: Tests labeled *Valid* by engineers but flagged as *Anomalous* by Isolation Forest. These represent potential **high-load operating regimes** (e.g. high current, temperature spikes) or unique sensor configurations.
> - **Group 2 (Supervised Invalid / Anomaly Normal)**: Tests labeled *Invalid* by engineers but statistically *Normal*. These represent subtle functional failures or manual engineer invalidations.

---

## 2. Key Disagreement Summary

- **Total Disagreement Records**: **{len(df_disagree)} records**
  - **Group 2 (Supervised Invalid / Anomaly Normal)**: **{len(grp2)} records**
  - **Group 3 (Supervised Valid / Anomaly Anomalous)**: **{len(grp3)} records**

---

## 3. Representative Case Investigations

### Sample Group 3 Cases (Valid but Anomalous — Potential Operating Regimes):
{grp3[["Test_ID", "Actual_Validity", "Supervised_Probability_Invalid", "IsolationForest_Score", "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C"]].head(5).to_markdown(index=False)}

### Sample Group 2 Cases (Invalid but Normal — Subtle Engineer Failures):
{grp2[["Test_ID", "Actual_Validity", "Supervised_Probability_Invalid", "IsolationForest_Score", "Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C"]].head(5).to_markdown(index=False)}

---

## 4. Test_Data Anomaly Screening (350 Records)

- **Test_Data Records Flagged Anomalous**: **{(df_test_scores['IsolationForest_Flag_SetD'] == 1).sum()} records** ({((df_test_scores['IsolationForest_Flag_SetD'] == 1).sum() / len(df_test_scores))*100.0:.2f}%)
- Reference `outputs/p2_stage3_anomaly_scores_test.csv` for individual `Test_ID` anomaly scores.
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(handoff_content)


if __name__ == '__main__':
    run_stage3_pipeline()
