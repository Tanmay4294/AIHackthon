"""Run the production Stage 9 detector and Stage 10 Test_Data inference."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Make ``python src/run_final_detector.py`` work from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import ID_COLUMN, TASK01_TARGET, load_test_data, load_training_data
from src.final_hybrid_detector import (
    DEFAULT_FEATURE_SET_NAME,
    DEFAULT_MODEL_NAME,
    LOCKED_THRESHOLD,
    FinalHybridDetector,
    compare_hybrid_approaches,
)
from src.stage8_validation import generate_oof_predictions, load_stage7_best_configuration

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"


def _distribution_table(train: pd.DataFrame, test: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["Applied_Voltage_kV", "Load_Current_A", "Ambient_Temperature_C", "Test_Duration_min",
                "Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4", "Invalid_Probability", "Anomaly_Score"]:
        if col in train and col in test:
            a = pd.to_numeric(train[col], errors="coerce")
            b = pd.to_numeric(test[col], errors="coerce")
        elif col in diagnostics:
            a = pd.Series(dtype=float)
            b = pd.to_numeric(diagnostics[col], errors="coerce")
        else:
            continue
        rows.append({"Feature": col, "Training_Missing": int(a.isna().sum()), "Test_Missing": int(b.isna().sum()),
                     "Training_Min": a.min(), "Test_Min": b.min(), "Training_Max": a.max(), "Test_Max": b.max(),
                     "Training_Mean": a.mean(), "Test_Mean": b.mean()})
    for col in ["missing_any", "data_quality_issue", "sensor_s2_negative", "sensor_zero_reading"]:
        if col in diagnostics:
            rows.append({"Feature": col, "Training_Missing": np.nan, "Test_Missing": int(diagnostics[col].isna().sum()),
                         "Training_Min": np.nan, "Test_Min": np.nan, "Training_Max": np.nan, "Test_Max": np.nan,
                         "Training_Mean": np.nan, "Test_Mean": float(pd.Series(diagnostics[col]).fillna(False).astype(bool).mean())})
    return pd.DataFrame(rows)


def _write_stage9_reports(comparison: pd.DataFrame, output_dir: Path, config: object) -> None:
    comparison.to_csv(output_dir / "p1_stage9_hybrid_comparison.csv", index=False)
    selected = comparison.loc[comparison["Approach"] == "Supervised_only"].iloc[0]
    output_dir.joinpath("p1_stage9_gap_analysis.md").write_text(
        """# Person 1 Stage 9 Gap Analysis\n\n| Requirement | Already Implemented | Missing | Action Taken |\n|---|---|---|---|\n| Deterministic quality rules | Stage 1/2 quality flags and inference-safe rules | Stage 9 precedence decision | Reused hard corruption flags; validated them OOF before allowing precedence. |\n| Stage 4 residual and consistency features | Stage 4/5 residual, consistency, disagreement, and regime features | Evidence comparison in one final experiment | Reused Stage 5 transformations and evaluated physical evidence OOF. |\n| Stage 5 engineered features | `create_stage5_features` and leakage exclusions | Final production wrapper | Reused the exact feature generator in `FinalHybridDetector`. |\n| Stage 6 anomaly scores | Isolation Forest and LOF baseline infrastructure | Fold-isolated final comparison | Added fold-isolated Isolation Forest evidence for Stage 9 comparison and diagnostics only. |\n| Stage 7 supervised model | Validated Random Forest configuration | Production fit wrapper | Reused the winning model and feature set. |\n| Stage 8 OOF predictions | 5-fold OOF probabilities and locked threshold | Stage 9 candidate comparison | Compared six required approaches using the OOF probabilities. |\n| Stage 8 selected threshold | Threshold `0.40` with Invalid F1 `0.9853` | None | Locked `0.40`; never tuned on Test_Data. |\n| Person 2 final validity model | Existing P2 model and reports | Not competitive with P1 baseline | Audited existing P2 artifacts; retained P1 because its validated performance is stronger. |\n| Raw-to-prediction production path | Existing loader and feature modules | One official output and diagnostics | Added reusable detector and Stage 10 runner. |\n\n## Selection\n\nThe supervised-only approach is retained. Quality evidence produces no historical metric improvement because the locked model already catches every hard-quality Invalid OOF row. Physical consistency and anomaly evidence add false positives without improving recall. Soft evidence is therefore diagnostic only.\n""", encoding="utf-8")
    output_dir.joinpath("p1_stage9_precedence_rules.md").write_text(
        """# Person 1 Stage 9 Precedence Rules\n\n1. Generate quality, Stage 4/5 physical, and diagnostic anomaly evidence using the same inference-safe code used for historical validation.\n2. If an inference-available hard corruption flag is present (`missing_critical_measurement`, non-finite or malformed value, impossible physical value, negative/zero sensor reading, or duplicate measurement pair), mark Invalid. These rules are deterministic and were Invalid-only in historical Training_Data; the locked model already identified all such OOF rows.\n3. Otherwise, use the locked Random Forest Invalid probability with threshold `0.40`.\n4. Residual, consistency, disagreement, operating-regime, and anomaly evidence never override the supervised decision. Historical OOF comparison showed that doing so increases false positives and would risk rejecting legitimate unusual regimes.\n5. No Test_ID-specific exception or manual row edit is permitted.\n\nHigh current, high voltage, high temperature, long duration, extreme sensor values, and rare regimes are not deterministic Invalid rules by themselves.\n""", encoding="utf-8")


def run_final_detector() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    training = load_training_data()
    test = load_test_data()
    if len(test) != 350:
        raise AssertionError(f"Expected 350 Test_Data rows, got {len(test)}")
    if TASK01_TARGET in test.columns or "Reference_Parameter" in test.columns:
        raise AssertionError("Test_Data must be used without hidden target columns")

    config = load_stage7_best_configuration(training, prefer_cached=True)
    oof = generate_oof_predictions(training, model_name=config.model_name, feature_set_name=config.feature_set_name,
                                    n_splits=5, random_state=42)
    comparison = compare_hybrid_approaches(oof, random_state=42)
    _write_stage9_reports(comparison, OUTPUTS, config)

    detector = FinalHybridDetector(threshold=LOCKED_THRESHOLD, random_state=42).fit(training)
    diagnostics = detector.predict_with_diagnostics(test)
    diagnostics.to_csv(OUTPUTS / "p1_stage10_test_predictions_diagnostics.csv", index=False)
    official = diagnostics[[ID_COLUMN, "Final_Validity_Label"]].rename(columns={"Final_Validity_Label": "Validity_Label"})
    if list(official.columns) != ["Test_ID", "Validity_Label"]:
        raise AssertionError("Official submission schema is incorrect")
    if len(official) != 350 or official[ID_COLUMN].nunique() != 350:
        raise AssertionError("Official submission row or ID count is incorrect")
    if set(official[ID_COLUMN]) != set(test[ID_COLUMN]):
        raise AssertionError("Official submission IDs do not match Test_Data")
    if official["Validity_Label"].isna().any() or not set(official["Validity_Label"]).issubset({"Valid", "Invalid"}):
        raise AssertionError("Official submission contains invalid labels")
    official.to_csv(OUTPUTS / "task1_predictions.csv", index=False)

    distribution = _distribution_table(training, test, diagnostics)
    distribution.to_csv(OUTPUTS / "p1_stage10_distribution_check.csv", index=False)
    distribution.to_markdown(OUTPUTS / "p1_stage10_distribution_check.md", index=False)
    valid_count = int((official["Validity_Label"] == "Valid").sum())
    invalid_count = int((official["Validity_Label"] == "Invalid").sum())
    OUTPUTS.joinpath("p1_stage10_test_prediction_report.md").write_text(
        f"""# Person 1 Stage 10 Test Prediction Report\n\n- Test rows: `350`\n- Predicted Valid: `{valid_count}` ({valid_count / 350:.2%})\n- Predicted Invalid: `{invalid_count}` ({invalid_count / 350:.2%})\n- Model: `{config.model_name}`\n- Feature set: `{config.feature_set_name}`\n- Locked threshold: `{LOCKED_THRESHOLD:.2f}`\n- Hybrid components used for official decisions: supervised probability plus validated hard-corruption safety rules; physical consistency and anomaly scores are diagnostics only.\n- Precedence: hard deterministic corruption, then locked supervised probability, with soft evidence non-overriding.\n- Hidden Test_Data labels were never accessed or inferred.\n- Every Test_ID received exactly one prediction; IDs are unique and match the input set.\n- Historical validation metrics are reported only in `p1_stage9_hybrid_comparison.csv` and Stage 8 reports. No Test_Data accuracy, precision, recall, or F1 is claimed.\n\nOfficial submission: `outputs/task1_predictions.csv`\nDiagnostics: `outputs/p1_stage10_test_predictions_diagnostics.csv`\nDistribution checks: `outputs/p1_stage10_distribution_check.md`\n""", encoding="utf-8")
    return {"config": config, "comparison": comparison, "official": official, "diagnostics": diagnostics}


if __name__ == "__main__":
    result = run_final_detector()
    print(result["comparison"].to_string(index=False))
    print(f"Saved {len(result['official'])} predictions to outputs/task1_predictions.csv")
