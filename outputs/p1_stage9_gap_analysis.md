# Person 1 Stage 9 Gap Analysis

| Requirement | Already Implemented | Missing | Action Taken |
|---|---|---|---|
| Deterministic quality rules | Stage 1/2 quality flags and inference-safe rules | Stage 9 precedence decision | Reused hard corruption flags; validated them OOF before allowing precedence. |
| Stage 4 residual and consistency features | Stage 4/5 residual, consistency, disagreement, and regime features | Evidence comparison in one final experiment | Reused Stage 5 transformations and evaluated physical evidence OOF. |
| Stage 5 engineered features | `create_stage5_features` and leakage exclusions | Final production wrapper | Reused the exact feature generator in `FinalHybridDetector`. |
| Stage 6 anomaly scores | Isolation Forest and LOF baseline infrastructure | Fold-isolated final comparison | Added fold-isolated Isolation Forest evidence for Stage 9 comparison and diagnostics only. |
| Stage 7 supervised model | Validated Random Forest configuration | Production fit wrapper | Reused the winning model and feature set. |
| Stage 8 OOF predictions | 5-fold OOF probabilities and locked threshold | Stage 9 candidate comparison | Compared six required approaches using the OOF probabilities. |
| Stage 8 selected threshold | Threshold `0.40` with Invalid F1 `0.9853` | None | Locked `0.40`; never tuned on Test_Data. |
| Person 2 final validity model | Existing P2 model and reports | Not competitive with P1 baseline | Audited existing P2 artifacts; retained P1 because its validated performance is stronger. |
| Raw-to-prediction production path | Existing loader and feature modules | One official output and diagnostics | Added reusable detector and Stage 10 runner. |

## Selection

The supervised-only approach is retained. Quality evidence produces no historical metric improvement because the locked model already catches every hard-quality Invalid OOF row. Physical consistency and anomaly evidence add false positives without improving recall. Soft evidence is therefore diagnostic only.
