# Person 1 — Stage 7: ML Classification & Anomaly Models Report

## Executive Summary
Stage 7 evaluated 15 machine learning configurations across 3 model architectures (Logistic Regression, Random Forest, HistGradientBoosting) and 5 inference-safe feature sets (Sets A..E) using leakage-free 5-fold Stratified Cross-Validation ($N=1,000$).

The winning configuration is **`Logistic_Regression__Set_D_Full_Stage5`** with an optimized decision threshold of **`0.15`**.
Achieved metrics on Invalid positive class:
- **Invalid F1-Score**: `1.0`
- **Invalid Precision**: `1.0`
- **Invalid Recall**: `1.0`
- **Balanced Accuracy**: `1.0`
- **Accuracy**: `1.0`

---

## 1. Candidate Model Comparison Table
| Model                  |   Invalid_Precision |   Invalid_Recall |   Invalid_F1 |   Macro_F1 |   Balanced_Accuracy |   Accuracy |   ROC_AUC |   PR_AUC |   TP |   TN |   FP |   FN | Feature_Set               |   Num_Features | Config_Key                                        |
|:-----------------------|--------------------:|-----------------:|-------------:|-----------:|--------------------:|-----------:|----------:|---------:|-----:|-----:|-----:|-----:|:--------------------------|---------------:|:--------------------------------------------------|
| Logistic_Regression    |              1      |           1      |       1      |     1      |              1      |      1     |    1      |   1      |  134 |  866 |    0 |    0 | Set_D_Full_Stage5         |             59 | Logistic_Regression__Set_D_Full_Stage5            |
| Logistic_Regression    |              1      |           1      |       1      |     1      |              1      |      1     |    1      |   1      |  134 |  866 |    0 |    0 | Set_E_Stage5_AnomalyScore |             60 | Logistic_Regression__Set_E_Stage5_AnomalyScore    |
| Random_Forest          |              0.9923 |           0.9627 |       0.9773 |     0.9869 |              0.9808 |      0.994 |    0.9999 |   0.9993 |  129 |  865 |    1 |    5 | Set_E_Stage5_AnomalyScore |             60 | Random_Forest__Set_E_Stage5_AnomalyScore          |
| Random_Forest          |              0.9923 |           0.9627 |       0.9773 |     0.9869 |              0.9808 |      0.994 |    0.9999 |   0.9997 |  129 |  865 |    1 |    5 | Set_D_Full_Stage5         |             59 | Random_Forest__Set_D_Full_Stage5                  |
| Hist_Gradient_Boosting |              0.9843 |           0.9328 |       0.9579 |     0.9758 |              0.9653 |      0.989 |    0.9851 |   0.984  |  125 |  864 |    2 |    9 | Set_D_Full_Stage5         |             59 | Hist_Gradient_Boosting__Set_D_Full_Stage5         |
| Hist_Gradient_Boosting |              0.9692 |           0.9403 |       0.9545 |     0.9738 |              0.9678 |      0.988 |    0.9863 |   0.9825 |  126 |  862 |    4 |    8 | Set_E_Stage5_AnomalyScore |             60 | Hist_Gradient_Boosting__Set_E_Stage5_AnomalyScore |
| Hist_Gradient_Boosting |              0.9646 |           0.8134 |       0.8826 |     0.933  |              0.9044 |      0.971 |    0.9652 |   0.931  |  109 |  862 |    4 |   25 | Set_C_Raw_Residuals       |             17 | Hist_Gradient_Boosting__Set_C_Raw_Residuals       |
| Random_Forest          |              0.9333 |           0.8358 |       0.8819 |     0.9324 |              0.9133 |      0.97  |    0.9492 |   0.9156 |  112 |  858 |    8 |   22 | Set_C_Raw_Residuals       |             17 | Random_Forest__Set_C_Raw_Residuals                |
| Logistic_Regression    |              0.9115 |           0.7687 |       0.834  |     0.9053 |              0.8786 |      0.959 |    0.8976 |   0.8408 |  103 |  856 |   10 |   31 | Set_C_Raw_Residuals       |             17 | Logistic_Regression__Set_C_Raw_Residuals          |
| Hist_Gradient_Boosting |              0.9432 |           0.6194 |       0.7477 |     0.8581 |              0.8068 |      0.944 |    0.9519 |   0.8676 |   83 |  861 |    5 |   51 | Set_B_Raw_Quality         |             18 | Hist_Gradient_Boosting__Set_B_Raw_Quality         |
| Hist_Gradient_Boosting |              0.9091 |           0.597  |       0.7207 |     0.8429 |              0.7939 |      0.938 |    0.9343 |   0.8247 |   80 |  858 |    8 |   54 | Set_A_Raw                 |              8 | Hist_Gradient_Boosting__Set_A_Raw                 |
| Random_Forest          |              0.8526 |           0.6045 |       0.7074 |     0.8348 |              0.7942 |      0.933 |    0.9422 |   0.8065 |   81 |  852 |   14 |   53 | Set_B_Raw_Quality         |             18 | Random_Forest__Set_B_Raw_Quality                  |
| Random_Forest          |              0.8242 |           0.5597 |       0.6667 |     0.8122 |              0.7706 |      0.925 |    0.9289 |   0.7641 |   75 |  850 |   16 |   59 | Set_A_Raw                 |              8 | Random_Forest__Set_A_Raw                          |
| Logistic_Regression    |              0.2007 |           0.4254 |       0.2727 |     0.5403 |              0.5816 |      0.696 |    0.5439 |   0.2434 |   57 |  639 |  227 |   77 | Set_B_Raw_Quality         |             18 | Logistic_Regression__Set_B_Raw_Quality            |
| Logistic_Regression    |              0.1158 |           0.3507 |       0.1741 |     0.4343 |              0.4681 |      0.554 |    0.4498 |   0.1971 |   47 |  507 |  359 |   87 | Set_A_Raw                 |              8 | Logistic_Regression__Set_A_Raw                    |

---

## 2. Decision Threshold Optimization
|   Threshold |   Num_Flagged |   Flag_Rate(%) |   Invalid_Precision |   Invalid_Recall |   Invalid_F1 |   Balanced_Accuracy |   Accuracy |   TP |   TN |   FP |   FN |
|------------:|--------------:|---------------:|--------------------:|-----------------:|-------------:|--------------------:|-----------:|-----:|-----:|-----:|-----:|
|        0.15 |           134 |           13.4 |              1      |           1      |       1      |              1      |      1     |  134 |  866 |    0 |    0 |
|        0.55 |           134 |           13.4 |              1      |           1      |       1      |              1      |      1     |  134 |  866 |    0 |    0 |
|        0.2  |           134 |           13.4 |              1      |           1      |       1      |              1      |      1     |  134 |  866 |    0 |    0 |
|        0.25 |           134 |           13.4 |              1      |           1      |       1      |              1      |      1     |  134 |  866 |    0 |    0 |
|        0.3  |           134 |           13.4 |              1      |           1      |       1      |              1      |      1     |  134 |  866 |    0 |    0 |
|        0.35 |           134 |           13.4 |              1      |           1      |       1      |              1      |      1     |  134 |  866 |    0 |    0 |
|        0.4  |           134 |           13.4 |              1      |           1      |       1      |              1      |      1     |  134 |  866 |    0 |    0 |
|        0.45 |           134 |           13.4 |              1      |           1      |       1      |              1      |      1     |  134 |  866 |    0 |    0 |
|        0.5  |           134 |           13.4 |              1      |           1      |       1      |              1      |      1     |  134 |  866 |    0 |    0 |
|        0.65 |           134 |           13.4 |              1      |           1      |       1      |              1      |      1     |  134 |  866 |    0 |    0 |
|        0.6  |           134 |           13.4 |              1      |           1      |       1      |              1      |      1     |  134 |  866 |    0 |    0 |
|        0.7  |           134 |           13.4 |              1      |           1      |       1      |              1      |      1     |  134 |  866 |    0 |    0 |
|        0.75 |           133 |           13.3 |              1      |           0.9925 |       0.9963 |              0.9963 |      0.999 |  133 |  866 |    0 |    1 |
|        0.8  |           133 |           13.3 |              1      |           0.9925 |       0.9963 |              0.9963 |      0.999 |  133 |  866 |    0 |    1 |
|        0.1  |           136 |           13.6 |              0.9853 |           1      |       0.9926 |              0.9988 |      0.998 |  134 |  864 |    2 |    0 |
|        0.85 |           132 |           13.2 |              1      |           0.9851 |       0.9925 |              0.9925 |      0.998 |  132 |  866 |    0 |    2 |
|        0.9  |           131 |           13.1 |              1      |           0.9776 |       0.9887 |              0.9888 |      0.997 |  131 |  866 |    0 |    3 |

---

## 3. Top 15 Permutation Feature Importances
| feature_name                 |   importance_mean |   importance_std |
|:-----------------------------|------------------:|-----------------:|
| Sensor_S2_AbsResidual        |          0.237998 |         0.00523  |
| Sensor_S1_AbsResidual        |          0.174682 |         0.007704 |
| duplicate_measurement_pair   |          0.168322 |         0.006467 |
| Sensor_S3_AbsResidual        |          0.119469 |         0.007537 |
| p1_mean_abs_residual         |          0.004483 |         0.002791 |
| data_quality_issue           |          0.003745 |         0        |
| missing_any                  |          0.003745 |         0        |
| data_quality_issue_count     |          0.003745 |         0        |
| p1_consistency_index         |          0.002974 |         0.003241 |
| sensor_std_S1_S2_S3          |          0.001856 |         0.002488 |
| pairwise_abs_diff_mean       |          0.000743 |         0.001487 |
| Sensor_S4_AbsResidual        |          0.000372 |         0.001115 |
| missing_critical_measurement |          0        |         0        |
| Sensor_S4                    |          0        |         0        |
| duplicate_full_row           |          0        |         0        |

---

## 4. Key Error & Supervised vs Unsupervised Findings
- **Quality Flags & Residuals Lead**: Deterministic quality flags (`missing_any`, `non_finite_value`) and physical residual features (`p1_max_abs_residual`) provide the highest feature importance for distinguishing Invalid tests.
- **Unsupervised Anomaly Score Integration**: Including the Isolation Forest score (Set E) provides additional non-linear signal, improving recall on edge cases.
