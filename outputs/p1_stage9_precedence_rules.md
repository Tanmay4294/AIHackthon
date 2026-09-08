# Person 1 Stage 9 Precedence Rules

1. Generate quality, Stage 4/5 physical, and diagnostic anomaly evidence using the same inference-safe code used for historical validation.
2. If an inference-available hard corruption flag is present (`missing_critical_measurement`, non-finite or malformed value, impossible physical value, negative/zero sensor reading, or duplicate measurement pair), mark Invalid. These rules are deterministic and were Invalid-only in historical Training_Data; the locked model already identified all such OOF rows.
3. Otherwise, use the locked Random Forest Invalid probability with threshold `0.40`.
4. Residual, consistency, disagreement, operating-regime, and anomaly evidence never override the supervised decision. Historical OOF comparison showed that doing so increases false positives and would risk rejecting legitimate unusual regimes.
5. No Test_ID-specific exception or manual row edit is permitted.

High current, high voltage, high temperature, long duration, extreme sensor values, and rare regimes are not deterministic Invalid rules by themselves.
