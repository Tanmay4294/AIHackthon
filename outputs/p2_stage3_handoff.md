# Person 2 — Stage 3 Handoff Report for Person 1

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

- **Total Disagreement Records**: **109 records**
  - **Group 2 (Supervised Invalid / Anomaly Normal)**: **57 records**
  - **Group 3 (Supervised Valid / Anomaly Anomalous)**: **52 records**

---

## 3. Representative Case Investigations

### Sample Group 3 Cases (Valid but Anomalous — Potential Operating Regimes):
| Test_ID   | Actual_Validity   |   Supervised_Probability_Invalid |   IsolationForest_Score |   Applied_Voltage_kV |   Load_Current_A |   Ambient_Temperature_C |
|:----------|:------------------|---------------------------------:|------------------------:|---------------------:|-----------------:|------------------------:|
| TRN-0140  | Valid             |                             0.32 |                0.596399 |              13.4841 |          75.9719 |                 25.2769 |
| TRN-0405  | Valid             |                             0.29 |                0.593995 |              24.6143 |          33.6779 |                 42.1612 |
| TRN-0257  | Valid             |                             0.16 |                0.603664 |              30.6035 |          29.4846 |                 36.2652 |
| TRN-0434  | Valid             |                             0.23 |                0.596047 |              18.3304 |          91.2825 |                 54.8353 |
| TRN-0920  | Valid             |                             0.38 |                0.604191 |              28.0007 |          43.5856 |                 29.5635 |

### Sample Group 2 Cases (Invalid but Normal — Subtle Engineer Failures):
| Test_ID   | Actual_Validity   |   Supervised_Probability_Invalid |   IsolationForest_Score |   Applied_Voltage_kV |   Load_Current_A |   Ambient_Temperature_C |
|:----------|:------------------|---------------------------------:|------------------------:|---------------------:|-----------------:|------------------------:|
| TRN-0135  | Invalid           |                             0.68 |                0.450723 |              23.8796 |          37.0768 |                 27.7869 |
| TRN-0087  | Invalid           |                             0.83 |                0.448606 |              17.6142 |          31.0902 |                 47.58   |
| TRN-0323  | Invalid           |                             0.67 |                0.441606 |              23.8319 |          94.5641 |                 40.9548 |
| TRN-0124  | Invalid           |                             0.82 |                0.450916 |              28.7059 |          44.4546 |                 28.3929 |
| TRN-0097  | Invalid           |                             0.61 |                0.470628 |              24.0983 |          15.5749 |                 32.5565 |

---

## 4. Test_Data Anomaly Screening (350 Records)

- **Test_Data Records Flagged Anomalous**: **28 records** (8.00%)
- Reference `outputs/p2_stage3_anomaly_scores_test.csv` for individual `Test_ID` anomaly scores.
