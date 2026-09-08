"""Merge Task 01 and Task 02 outputs by Test_ID without fabricating Task 02 values."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import ID_COLUMN, load_sample_submission, load_test_data

ROOT = Path(__file__).resolve().parent.parent


def build_final_team_submission(task1_path: Path | str, task2_path: Path | str, output_path: Path | str) -> pd.DataFrame:
    """Validate and merge Task 01 labels with a teammate-provided Task 02 CSV.

    Task 02 may provide either ``Reference_Parameter`` (the integration input
    contract) or the sample-submission name ``Predicted_Reference_Parameter``.
    The final file always follows the workbook's official sample schema.
    """
    task1 = pd.read_csv(task1_path)
    task2 = pd.read_csv(task2_path)
    official_columns = list(load_sample_submission().columns)
    task2_output_column = "Predicted_Reference_Parameter"
    if list(task1.columns) != [ID_COLUMN, "Validity_Label"]:
        raise ValueError("Task 01 file must contain exactly Test_ID,Validity_Label")
    if task2_output_column not in task2.columns:
        if "Reference_Parameter" not in task2.columns:
            raise ValueError("Task 02 file must contain Test_ID and Reference_Parameter or Predicted_Reference_Parameter")
        task2 = task2.rename(columns={"Reference_Parameter": task2_output_column})
    if set(task2.columns) != {ID_COLUMN, task2_output_column}:
        raise ValueError("Task 02 file must contain only Test_ID and its prediction column")
    for name, frame in (("Task 01", task1), ("Task 02", task2)):
        if ID_COLUMN not in frame.columns or frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"{name} has missing or duplicate Test_ID values")
    test_data = load_test_data()
    expected_ids = set(test_data[ID_COLUMN])
    if len(task1) != 350 or len(task2) != 350 or set(task1[ID_COLUMN]) != expected_ids or set(task2[ID_COLUMN]) != expected_ids:
        raise ValueError("Both Task 01 and Task 02 must contain exactly the 350 Test_Data IDs")
    if task1["Validity_Label"].isna().any() or not set(task1["Validity_Label"]).issubset({"Valid", "Invalid"}):
        raise ValueError("Task 01 labels are missing or invalid")
    if task2[task2_output_column].isna().any():
        raise ValueError("Task 02 predictions contain missing values")
    merged = test_data[[ID_COLUMN]].merge(task2, on=ID_COLUMN, validate="one_to_one").merge(task1, on=ID_COLUMN, validate="one_to_one")
    merged = merged[official_columns]
    if len(merged) != 350 or merged[ID_COLUMN].nunique() != 350 or merged.isna().any().any():
        raise ValueError("Final submission validation failed")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, index=False)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge validated Task 01 and Task 02 predictions by Test_ID.")
    parser.add_argument("--task1", default=str(ROOT / "outputs" / "task1_predictions.csv"))
    parser.add_argument("--task2", required=True, help="Teammate Task 02 CSV; no predictions are fabricated.")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "final_team_submission.csv"))
    args = parser.parse_args()
    final = build_final_team_submission(args.task1, args.task2, args.output)
    print(f"Saved {len(final)} validated final-team rows to {args.output}")
