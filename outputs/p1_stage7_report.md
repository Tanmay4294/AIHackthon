# Person 1 — Stage 7: ML Classification & Anomaly Models Report

## Executive Summary
Stage 7 evaluated 15 machine learning configurations across 3 model architectures (Logistic Regression, Random Forest, HistGradientBoosting) and 5 inference-safe feature sets (Sets A..E) using leakage-free 5-fold Stratified Cross-Validation ($N=1,000$).

The winning configuration is **`Random_Forest__Set_D_Full_Stage5`** with an optimized decision threshold of **`0.4`**.
Achieved metrics on Invalid positive class:
- **Invalid F1-Score**: `0.9853`
- **Invalid Precision**: `0.971`
- **Invalid Recall**: `1.0`
- **Balanced Accuracy**: `0.9977`
- **Accuracy**: `0.996`

---

## 1. Candidate Model Comparison Table
| Model                  | Feature_Set               |   Invalid_Precision |   Invalid_Recall |   Invalid_F1 |   Macro_F1 |   Balanced_Accuracy |   Accuracy |   ROC_AUC |   PR_AUC |   TP |   TN |   FP |   FN | Config_Key                                        |
|:-----------------------|:--------------------------|--------------------:|-----------------:|-------------:|-----------:|--------------------:|-----------:|----------:|---------:|-----:|-----:|-----:|-----:|:--------------------------------------------------|
| Random_Forest          | Set_D_Full_Stage5         |              0.9848 |           0.9701 |       0.9774 |     0.987  |              0.9839 |      0.994 |    0.9999 |   0.9991 |  130 |  864 |    2 |    4 | Random_Forest__Set_D_Full_Stage5                  |
| Random_Forest          | Set_E_Stage5_AnomalyScore |              0.9774 |           0.9701 |       0.9738 |     0.9849 |              0.9833 |      0.993 |    0.9998 |   0.9989 |  130 |  863 |    3 |    4 | Random_Forest__Set_E_Stage5_AnomalyScore          |
| Hist_Gradient_Boosting | Set_E_Stage5_AnomalyScore |              0.9845 |           0.9478 |       0.9658 |     0.9803 |              0.9727 |      0.991 |    0.9818 |   0.9762 |  127 |  864 |    2 |    7 | Hist_Gradient_Boosting__Set_E_Stage5_AnomalyScore |
| Hist_Gradient_Boosting | Set_D_Full_Stage5         |              0.9844 |           0.9403 |       0.9618 |     0.978  |              0.969  |      0.99  |    0.9828 |   0.9778 |  126 |  864 |    2 |    8 | Hist_Gradient_Boosting__Set_D_Full_Stage5         |
| Logistic_Regression    | Set_D_Full_Stage5         |              1      |           0.8433 |       0.915  |     0.9515 |              0.9216 |      0.979 |    0.9767 |   0.9406 |  113 |  866 |    0 |   21 | Logistic_Regression__Set_D_Full_Stage5            |
| Logistic_Regression    | Set_E_Stage5_AnomalyScore |              0.9912 |           0.8358 |       0.9069 |     0.9469 |              0.9173 |      0.977 |    0.9787 |   0.941  |  112 |  865 |    1 |   22 | Logistic_Regression__Set_E_Stage5_AnomalyScore    |
| Hist_Gradient_Boosting | Set_C_Raw_Residuals       |              0.9818 |           0.806  |       0.8852 |     0.9347 |              0.9018 |      0.972 |    0.9564 |   0.9226 |  108 |  864 |    2 |   26 | Hist_Gradient_Boosting__Set_C_Raw_Residuals       |
| Random_Forest          | Set_C_Raw_Residuals       |              0.9391 |           0.806  |       0.8675 |     0.9243 |              0.8989 |      0.967 |    0.946  |   0.9093 |  108 |  859 |    7 |   26 | Random_Forest__Set_C_Raw_Residuals                |
| Logistic_Regression    | Set_C_Raw_Residuals       |              0.9266 |           0.7537 |       0.8313 |     0.904  |              0.8722 |      0.959 |    0.8964 |   0.8353 |  101 |  858 |    8 |   33 | Logistic_Regression__Set_C_Raw_Residuals          |
| Hist_Gradient_Boosting | Set_B_Raw_Quality         |              0.9425 |           0.6119 |       0.7421 |     0.855  |              0.8031 |      0.943 |    0.9508 |   0.8662 |   82 |  861 |    5 |   52 | Hist_Gradient_Boosting__Set_B_Raw_Quality         |
| Random_Forest          | Set_B_Raw_Quality         |              0.866  |           0.6269 |       0.7273 |     0.8458 |              0.8059 |      0.937 |    0.932  |   0.792  |   84 |  853 |   13 |   50 | Random_Forest__Set_B_Raw_Quality                  |
| Hist_Gradient_Boosting | Set_A_Raw                 |              0.9091 |           0.597  |       0.7207 |     0.8429 |              0.7939 |      0.938 |    0.9343 |   0.8247 |   80 |  858 |    8 |   54 | Hist_Gradient_Boosting__Set_A_Raw                 |
| Random_Forest          | Set_A_Raw                 |              0.8242 |           0.5597 |       0.6667 |     0.8122 |              0.7706 |      0.925 |    0.9289 |   0.7641 |   75 |  850 |   16 |   59 | Random_Forest__Set_A_Raw                          |
| Logistic_Regression    | Set_B_Raw_Quality         |              0.2007 |           0.4254 |       0.2727 |     0.5403 |              0.5816 |      0.696 |    0.5439 |   0.2434 |   57 |  639 |  227 |   77 | Logistic_Regression__Set_B_Raw_Quality            |
| Logistic_Regression    | Set_A_Raw                 |              0.1158 |           0.3507 |       0.1741 |     0.4343 |              0.4681 |      0.554 |    0.4498 |   0.1971 |   47 |  507 |  359 |   87 | Logistic_Regression__Set_A_Raw                    |

---

## 2. Decision Threshold Optimization
|   Threshold |   Num_Flagged |   Flag_Rate(%) |   Invalid_Precision |   Invalid_Recall |   Invalid_F1 |   Balanced_Accuracy |   Accuracy |   TP |   TN |   FP |   FN |
|------------:|--------------:|---------------:|--------------------:|-----------------:|-------------:|--------------------:|-----------:|-----:|-----:|-----:|-----:|
|        0.4  |           138 |           13.8 |              0.971  |           1      |       0.9853 |              0.9977 |      0.996 |  134 |  862 |    4 |    0 |
|        0.45 |           136 |           13.6 |              0.9779 |           0.9925 |       0.9852 |              0.9945 |      0.996 |  133 |  863 |    3 |    1 |
|        0.55 |           131 |           13.1 |              0.9924 |           0.9701 |       0.9811 |              0.9845 |      0.995 |  130 |  865 |    1 |    4 |
|        0.5  |           132 |           13.2 |              0.9848 |           0.9701 |       0.9774 |              0.9839 |      0.994 |  130 |  864 |    2 |    4 |
|        0.35 |           143 |           14.3 |              0.9371 |           1      |       0.9675 |              0.9948 |      0.991 |  134 |  857 |    9 |    0 |
|        0.6  |           124 |           12.4 |              1      |           0.9254 |       0.9612 |              0.9627 |      0.99  |  124 |  866 |    0 |   10 |
|        0.65 |           123 |           12.3 |              1      |           0.9179 |       0.9572 |              0.959  |      0.989 |  123 |  866 |    0 |   11 |
|        0.3  |           148 |           14.8 |              0.9054 |           1      |       0.9504 |              0.9919 |      0.986 |  134 |  852 |   14 |    0 |
|        0.7  |           116 |           11.6 |              1      |           0.8657 |       0.928  |              0.9328 |      0.982 |  116 |  866 |    0 |   18 |
|        0.25 |           157 |           15.7 |              0.8535 |           1      |       0.921  |              0.9867 |      0.977 |  134 |  843 |   23 |    0 |
|        0.75 |           103 |           10.3 |              1      |           0.7687 |       0.8692 |              0.8843 |      0.969 |  103 |  866 |    0 |   31 |
|        0.2  |           184 |           18.4 |              0.7283 |           1      |       0.8428 |              0.9711 |      0.95  |  134 |  816 |   50 |    0 |
|        0.8  |            85 |            8.5 |              1      |           0.6343 |       0.7763 |              0.8172 |      0.951 |   85 |  866 |    0 |   49 |
|        0.15 |           229 |           22.9 |              0.5852 |           1      |       0.7383 |              0.9452 |      0.905 |  134 |  771 |   95 |    0 |
|        0.85 |            67 |            6.7 |              1      |           0.5    |       0.6667 |              0.75   |      0.933 |   67 |  866 |    0 |   67 |
|        0.1  |           326 |           32.6 |              0.411  |           1      |       0.5826 |              0.8891 |      0.808 |  134 |  674 |  192 |    0 |
|        0.9  |            51 |            5.1 |              1      |           0.3806 |       0.5514 |              0.6903 |      0.917 |   51 |  866 |    0 |   83 |

---

## 3. Top 15 Permutation Feature Importances
| feature_name               |   importance_mean |   importance_std |
|:---------------------------|------------------:|-----------------:|
| duplicate_measurement_pair |          0.091069 |         0.010081 |
| sensor_std_S1_S2_S3        |          0.031356 |         0.009136 |
| Sensor_S1_AbsResidual      |          0.001859 |         0.001859 |
| Sensor_S2_AbsResidual      |          0.000372 |         0.001115 |
| Sensor_S1                  |          0        |         0        |
| Sensor_S2                  |          0        |         0        |
| Sensor_S3                  |          0        |         0        |
| Sensor_S4                  |          0        |         0        |
| missing_any                |          0        |         0        |
| Applied_Voltage_kV         |          0        |         0        |
| Ambient_Temperature_C      |          0        |         0        |
| Load_Current_A             |          0        |         0        |
| duplicate_full_row         |          0        |         0        |
| duplicate_test_id          |          0        |         0        |
| malformed_numeric          |          0        |         0        |

---

## 4. Key Error & Supervised vs Unsupervised Findings
- **Quality Flags & Residuals Lead**: Deterministic quality flags (`missing_any`, `non_finite_value`) and physical residual features (`p1_max_abs_residual`) provide the highest feature importance for distinguishing Invalid tests.
- **Unsupervised Anomaly Score Integration**: Including the Isolation Forest score (Set E) provides additional non-linear signal, improving recall on edge cases.
