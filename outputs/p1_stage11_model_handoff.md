# Person 1 Task 01 Model Handoff

## Locked Configuration

- Model: `Random_Forest` (`RandomForestClassifier`)
- Feature set: `Set_D_Full_Stage5` (60 features; exact ordered list in `p1_stage11_final_config.json`)
- Decision threshold: `0.40` for Invalid probability
- Random state: `42`
- Training data: 1,000 historical Training_Data rows (866 Valid, 134 Invalid)
- Class balance: `class_weight="balanced"`
- Hyperparameters: `n_estimators=200`, `max_depth=12`, `min_samples_split=5`, `criterion="gini"`, `max_features="sqrt"`
- Preprocessing: inference-safe quality flags; Stage 4 normal-behaviour residuals; Stage 5 differences, aggregates, disagreement, interactions, and residual features; median imputation inside the sklearn pipeline.

## Validation

Five-fold StratifiedKFold (`shuffle=True`, `random_state=42`) produced fold-isolated probabilities. Threshold selection used historical OOF predictions only.

| Metric | Value |
|---|---:|
| Invalid precision | 0.9710 |
| Invalid recall | 1.0000 |
| Invalid F1 | 0.9853 |
| Balanced accuracy | 0.9977 |
| Accuracy | 0.9960 |
| ROC-AUC | 0.9999 |
| PR-AUC | 0.9991 |
| TP / TN / FP / FN | 134 / 862 / 4 / 0 |

## Decision and Safety Rules

1. Inference-safe hard corruption flags may mark a record Invalid: missing critical measurement, non-finite or malformed numeric input, impossible physical input, zero/negative sensor anomaly, or duplicate measurement pair.
2. Otherwise, Invalid probability at or above `0.40` is Invalid; below is Valid.
3. Residual, consistency, anomaly, and regime evidence are diagnostic only. They do not override the probability decision because Stage 9 OOF tests showed added false positives without recall improvement.

`Test_ID`, `Validity_Label`, and `Reference_Parameter` are never model features. No Test_ID-specific exceptions exist.

## Reproducible Task 01 Inference

From repository root:

```powershell
python src/build_task1_submission.py
```

This reads the raw workbook, retrains on historical Training_Data, predicts raw Test_Data, and writes exactly:

```csv
Test_ID,Validity_Label
```

to `outputs/task1_predictions.csv`. The build fails on row-count, ID, missing-label, or invalid-label violations.

## Task 02 Merge Contract

Task 01 intermediate: `Test_ID,Validity_Label`.

Task 02 intermediate: `Test_ID,Reference_Parameter` (or `Test_ID,Predicted_Reference_Parameter`). Merge strictly by `Test_ID`, never row position. The workbook Sample_Submission establishes the final schema and order:

```csv
Test_ID,Predicted_Reference_Parameter,Validity_Label
```

After the Task 02 teammate provides their CSV, run:

```powershell
python src/build_final_team_submission.py --task2 path\to\task2_predictions.csv
```

The merge rejects missing/duplicate IDs, ID-set mismatches, missing values, non-binary Task 01 labels, and any output other than 350 rows. It does not create or estimate Task 02 predictions.
