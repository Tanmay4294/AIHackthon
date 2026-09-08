# Person 1 - Stage 8: Misclassification Analysis

## Selected Threshold

`0.40`

## False Positives

- Count: `4`
- Most common regimes: `{'High Voltage': 2, 'Standard': 1, 'Heavy Current': 1}`
- Mean current: `56.1458`
- Mean voltage: `21.9619`
- Mean p1_max_abs_residual: `19.0825`

## False Negatives

- Count: `0`
- Most common regimes: `None`
- Mean current: `N/A`
- Mean voltage: `N/A`
- Mean p1_max_abs_residual: `N/A`

## Systematic Pattern Table

| Pattern_Category   | Pattern                   |   Count |   Rate(%) | Notes                                            | Error_Type   |
|:-------------------|:--------------------------|--------:|----------:|:-------------------------------------------------|:-------------|
| Operating Regime   | High Voltage              |       2 |        50 | 2 / 4 FP cases occur in this regime.             | FP           |
| Operating Regime   | Standard                  |       1 |        25 | 1 / 4 FP cases occur in this regime.             | FP           |
| Operating Regime   | Heavy Current             |       1 |        25 | 1 / 4 FP cases occur in this regime.             | FP           |
| Residual Severity  | p1_max_abs_residual >= 10 |       4 |       100 | Counts cases with large fold-isolated residuals. | FP           |
| Load               | Load_Current_A > 85       |       1 |        25 | High-current operating regime.                   | FP           |
| Voltage            | Applied_Voltage_kV > 25   |       2 |        50 | High-voltage operating regime.                   | FP           |
| FN Count           | None                      |       0 |         0 | No FN records at the selected threshold.         | FN           |

## Interpretation

False positives are reviewed for legitimate unusual operating regimes, high load, high voltage, temperature, duration, and elevated residual or sensor disagreement patterns.
False negatives, if present, are reviewed for subtle residual abnormalities, missing values, or cases that look physically normal but remain inconsistent in the engineered features.
