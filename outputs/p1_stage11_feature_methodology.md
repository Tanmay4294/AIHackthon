# Person 1 Task 01 Feature Methodology

The final `Set_D_Full_Stage5` matrix contains the 60 ordered features in `p1_stage11_final_config.json`. It is generated identically for historical validation and raw Test_Data inference.

## A. Raw Physical Measurements

`Applied_Voltage_kV`, `Load_Current_A`, `Ambient_Temperature_C`, `Test_Duration_min`, and `Sensor_S1` through `Sensor_S4` describe operating load, environmental condition, duration, and observed sensors.

## B. Data-Quality Evidence

The matrix includes `missing_any`, `missing_critical_measurement`, `non_finite_value`, `duplicate_full_row`, `duplicate_test_id`, `duplicate_measurement_pair`, `malformed_numeric`, `invalid_voltage`, `invalid_current`, `invalid_duration`, `invalid_ambient_temp`, `sensor_s2_negative`, `sensor_zero_reading`, `data_quality_issue_count`, and `data_quality_issue`. These are calculated from the raw record at inference; they do not use labels, IDs as predictors, or manual changes.

## C. Sensor Consistency and Physical Behaviour

Sensor differences (`S1_minus_S2` through `S3_minus_S4`), sensor means/medians/standard deviations, `max_minus_min_sensors`, and pairwise absolute differences quantify agreement. Stage 4 expected values and residuals (`Sensor_S*_Expected`, residuals, absolute residuals, `p1_max_abs_residual`, `p1_mean_abs_residual`, `p1_residual_error`, `p1_consistency_index`, `p1_sensor_disagreement_index`, and `p1_regime_cluster`) judge sensors relative to load, voltage, temperature, and duration rather than against a global average.

## D. Other Stage 5 Features

`Apparent_Power_kVA`, `Thermal_Loading_Index`, and `Apparent_Impedance_Proxy` encode physically meaningful operating interactions.

## Interpretation Boundaries

High load, high voltage, high temperature, long duration, rare regime, or large raw sensor value is not an automatic Invalid decision. A physically consistent unusual regime remains eligible for Valid classification. Residual and consistency evidence matters because it identifies departures from behaviour expected under that record's operating conditions, but Stage 9 found it should not independently override the validated supervised probability rule. `Test_ID` is strictly an output key, and `Reference_Parameter` is excluded because it belongs to Task 02 rather than Task 01 validity classification.
