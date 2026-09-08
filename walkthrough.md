# Person 1 Stage 11 Walkthrough

## Audit and Reuse

Stage 11 reuses the validated Stage 1 loader, Stage 4/5 feature generation, Stage 7 Random Forest pipeline, Stage 8 OOF threshold selection, and Stage 9/10 `FinalHybridDetector`. No model architecture, feature set, preprocessing step, or threshold was changed. The existing Task 01 file was audited as valid before the integration wrapper was added.

## Final Task 01 Detector

- Model: class-balanced Random Forest (`n_estimators=200`, `max_depth=12`, `min_samples_split=5`, `random_state=42`)
- Feature set: `Set_D_Full_Stage5` (60 inference-safe physical, quality, interaction, sensor-consistency, and residual features)
- Threshold: `0.40` Invalid probability
- OOF validation: precision `0.9710`, recall `1.0000`, F1 `0.9853`, balanced accuracy `0.9977`, accuracy `0.9960`, ROC-AUC `0.9999`, PR-AUC `0.9991`; TP/TN/FP/FN `134/862/4/0`

Hard deterministic corruption evidence is a narrow safety override. Soft residual, consistency, anomaly, and regime evidence remains diagnostic because OOF hybrid tests increased false positives. There are no Test_ID-specific rules; raw extreme operating values are not automatically Invalid.

## Build and Output

Run from the repository root:

```powershell
python src/build_task1_submission.py
```

The command loads the raw workbook, fits the locked detector on the 1,000-row Training_Data, predicts all 350 Test_Data rows, validates integrity, and writes [task1_predictions.csv](/C:/Hackthon/aihackthon/outputs/task1_predictions.csv) with exactly:

```csv
Test_ID,Validity_Label
```

The verified output contains 304 Valid and 46 Invalid predictions. Test_Data labels remain hidden, so no Test_Data performance metric is claimed.

## Task 02 Integration

Provide a Task 02 CSV with `Test_ID,Reference_Parameter` or `Test_ID,Predicted_Reference_Parameter`. Then merge by ID only:

```powershell
python src/build_final_team_submission.py --task2 path\to\task2_predictions.csv
```

The final schema follows `Sample_Submission`: `Test_ID,Predicted_Reference_Parameter,Validity_Label`. The merge validates 350 unique IDs, exact ID-set match, no missing Task 01/Task 02 values, and no diagnostic columns.

## Deliverables

- [Model handoff](/C:/Hackthon/aihackthon/outputs/p1_stage11_model_handoff.md)
- [Machine-readable configuration](/C:/Hackthon/aihackthon/outputs/p1_stage11_final_config.json)
- [Feature methodology](/C:/Hackthon/aihackthon/outputs/p1_stage11_feature_methodology.md)
- [Competition methodology note](/C:/Hackthon/aihackthon/outputs/p1_stage11_methodology_note.md)
- [Stage 11 gap analysis](/C:/Hackthon/aihackthon/outputs/p1_stage11_gap_analysis.md)

## Environment and Verification

Python `>=3.10` is declared in `requirements.txt`; the final pipeline requires pandas, numpy, scipy, scikit-learn, openpyxl, matplotlib, seaborn, and tabulate. Stage 11 adds integration tests while preserving all prior tests. The raw workbook is never modified.
