# Person 1 Stage 10 Test Prediction Report

- Test rows: `350`
- Predicted Valid: `304` (86.86%)
- Predicted Invalid: `46` (13.14%)
- Model: `Random_Forest`
- Feature set: `Set_D_Full_Stage5`
- Locked threshold: `0.40`
- Hybrid components used for official decisions: supervised probability plus validated hard-corruption safety rules; physical consistency and anomaly scores are diagnostics only.
- Precedence: hard deterministic corruption, then locked supervised probability, with soft evidence non-overriding.
- Hidden Test_Data labels were never accessed or inferred.
- Every Test_ID received exactly one prediction; IDs are unique and match the input set.
- Historical validation metrics are reported only in `p1_stage9_hybrid_comparison.csv` and Stage 8 reports. No Test_Data accuracy, precision, recall, or F1 is claimed.

Official submission: `outputs/task1_predictions.csv`
Diagnostics: `outputs/p1_stage10_test_predictions_diagnostics.csv`
Distribution checks: `outputs/p1_stage10_distribution_check.md`
