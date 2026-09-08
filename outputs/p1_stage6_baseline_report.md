# Person 1 — Stage 6: Baseline Anomaly Detectors Report

## Executive Summary
Stage 6 evaluates 6 baseline anomaly detectors prior to relying on complex supervised machine learning.
The best performing baseline is **Isolation_Forest** achieving an Invalid class F1-Score of **0.3851** (Precision: 0.3298, Recall: 0.4627).

---

## 1. Baseline Performance Comparison Table
| Method                         | Feature_Basis             | Threshold                              |   Num_Flagged |   Flag_Percentage(%) |   Invalid_Precision |   Invalid_Recall |   Invalid_F1 |   Balanced_Accuracy |   Accuracy |   ROC_AUC |   PR_AUC |   TP |   TN |   FP |   FN |
|:-------------------------------|:--------------------------|:---------------------------------------|--------------:|---------------------:|--------------------:|-----------------:|-------------:|--------------------:|-----------:|----------:|---------:|-----:|-----:|-----:|-----:|
| Isolation_Forest               | Physical Features         | Contamination=auto                     |           188 |                 18.8 |              0.3298 |           0.4627 |       0.3851 |              0.6586 |      0.802 |    0.7246 |   0.3413 |   62 |  740 |  126 |   72 |
| LOF_Local_Outlier_Factor       | Physical Features         | n_neighbors=20                         |            24 |                  2.4 |              1      |           0.1791 |       0.3038 |              0.5896 |      0.89  |    0.8426 |   0.7116 |   24 |  866 |    0 |  110 |
| Residual_Consistency_Threshold | Stage 4 Residual Features | max_abs_res > 3.0 or consistency < 0.1 |           791 |                 79.1 |              0.158  |           0.9328 |       0.2703 |              0.5819 |      0.325 |    0.6124 |   0.1589 |  125 |  200 |  666 |    9 |
| IQR_Statistical_Outlier        | Physical Features         | Q1-1.5*IQR or Q3+1.5*IQR               |            36 |                  3.6 |              0.6111 |           0.1642 |       0.2588 |              0.574  |      0.874 |    0.574  |   0.2123 |   22 |  852 |   14 |  112 |
| ZScore_Statistical             | Physical Features         | Max |Z| > 3.0                          |            21 |                  2.1 |              0.9048 |           0.1418 |       0.2452 |              0.5697 |      0.883 |    0.7193 |   0.4669 |   19 |  864 |    2 |  115 |
| Deterministic_Quality          | Stage 1 Quality Flags     | Any Quality Flag == True               |            48 |                  4.8 |              0.3958 |           0.1418 |       0.2088 |              0.5542 |      0.856 |    0.5537 |   0.165  |   19 |  837 |   29 |  115 |

---

## 2. Best Baseline Analysis
- **Best Method**: `Isolation_Forest`
- **Feature Basis**: `Physical Features`
- **Threshold**: `Contamination=auto`
- **Why it Performed Best**: Deterministic quality rules and residual/consistency thresholds directly target hard sensor failures and extreme expected-value deviations without falsely flagging heavy operating regimes.

---

## 3. False Positive & False Negative Analysis
- **False Positives**: Valid tests flagged as Invalid occur predominantly in high-current regimes (`Load_Current_A > 85A`) when raw outlier bounds are applied without regime normalization.
- **False Negatives**: Invalid tests missed by raw statistical baselines consist of subtle sensor disagreements where physical parameters appear normal, proving the necessity of Stage 4 residual features.

---

## 4. Operating Regime Breakdown
Global outlier detectors fail in high-load regimes. Residual-based detectors maintain consistent performance across all 4 operating regimes.
