"""
CPRI Hackathon -- Person 1 Stage 8 Pipeline Runner
==================================================

Executes the Stage 8 validation workflow and writes all required artifacts
to the outputs directory.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stage8_validation import run_stage8_pipeline


def main() -> None:
    print("=" * 70)
    print("CPRI HACKATHON -- PERSON 1 STAGE 8 VALIDATION PIPELINE")
    print("=" * 70)
    results = run_stage8_pipeline()

    selection = results["selection"]
    overall_threshold = float(selection["selected_threshold"])
    oof_df = results["oof_predictions"]
    fold_metrics = results["fold_metrics"]
    fp_df = results["false_positives"]
    fn_df = results["false_negatives"]

    print(f"Selected threshold: {overall_threshold:.2f}")
    print(f"OOF rows: {len(oof_df)}")
    print(f"False positives: {len(fp_df)}")
    print(f"False negatives: {len(fn_df)}")
    print(f"Fold Invalid F1 mean: {fold_metrics['Invalid_F1'].mean():.4f}")
    print("=" * 70)
    print("STAGE 8 PIPELINE EXECUTED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

