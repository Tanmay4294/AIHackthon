# Person 1 Stage 4 Feature Handoff for Person 2

## Available Features
Person 1 Stage 4 exposes the following 18 residual and physical consistency features:
1. `Sensor_S1_Expected`, `Sensor_S1_Residual`, `Sensor_S1_AbsResidual`
2. `Sensor_S2_Expected`, `Sensor_S2_Residual`, `Sensor_S2_AbsResidual`
3. `Sensor_S3_Expected`, `Sensor_S3_Residual`, `Sensor_S3_AbsResidual`
4. `Sensor_S4_Expected`, `Sensor_S4_Residual`, `Sensor_S4_AbsResidual`
5. `p1_max_abs_residual`: Maximum absolute prediction error across all sensors.
6. `p1_mean_abs_residual`: Average absolute prediction error across all sensors.
7. `p1_residual_error`: Alias for `p1_max_abs_residual`.
8. `p1_consistency_index`: $1 / (1 + \text{p1\_mean\_abs\_residual})$.
9. `p1_sensor_disagreement_index`: $(\max(S) - \min(S)) / (|\text{mean}(S)| + 1e-5)$.
10. `p1_regime_cluster`: Categorical operating regime code (1..4).

## How to Load Features
In Person 2 Stage 4 script or adapter (`src/person1_adapter.py`), call:
```python
from src.person1_adapter import get_person1_features
df_p1_features, is_genuine, feature_names = get_person1_features(df)
```
If genuine `p1_` features exist in `df`, `is_genuine` will evaluate to `True` and return the engineered residual features.

## Validation Status
- 100% reproducible on Training_Data (1,000 rows) and Test_Data (350 rows).
- Zero missing values generated (imputed medians for predictor features).
- Trained STRICTLY on Valid training records to prevent target leakage.
