# Task 01 Methodology Note

**Problem and data.** Task 01 classifies each electrical test record as Valid or Invalid. The historical Training_Data has 1,000 labeled records (866 Valid, 134 Invalid); Test_Data has 350 unlabeled records.

**Data quality and features.** We preserve every input row and generate inference-safe flags for missing, malformed, non-finite, duplicate, and physically impossible values. Feature engineering combines the eight raw measurements with sensor differences, sensor aggregates, disagreement metrics, apparent-power/thermal/impedance interactions, and normal-behaviour residual features. Expected sensor behaviour is modelled relative to voltage, current, temperature, and duration, so unusual but internally consistent regimes are not rejected merely for being extreme.

**Model and validation.** We evaluated fold-isolated 5-fold Stratified CV configurations and retained a class-balanced Random Forest using the full Stage 5 feature set. Its `n_estimators=200`, `max_depth=12`, and `min_samples_split=5` configuration uses median imputation in the model pipeline. `Test_ID`, `Validity_Label`, and `Reference_Parameter` are excluded from predictors.

**Decision threshold and final logic.** A threshold sweep on historical out-of-fold probabilities selected `0.40`, optimizing Invalid F1 with recall-aware tie breaking. Stage 9 tested deterministic, consistency, and anomaly combinations; soft combinations added false positives without improving recall. The final decision is therefore the locked supervised probability, with a narrow safety override for deterministic hard corruption evidence. No Test_Data labels, row-specific rules, or manual edits are used.

**Results and inference.** Historical OOF validation achieved Invalid precision 0.9710, recall 1.0000, F1 0.9853, balanced accuracy 0.9977, accuracy 0.9960, ROC-AUC 0.9999, and PR-AUC 0.9991 (TP=134, TN=862, FP=4, FN=0). The production pipeline retrains only on Training_Data and predicts all 350 Test_Data rows. Test labels are hidden, so no Test_Data accuracy claim is made.
