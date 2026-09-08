"""Build the official Task 01 submission from the supplied raw workbook.

This is intentionally a thin integration entry point: feature generation and
prediction remain owned by ``FinalHybridDetector`` so the locked Stage 9/10
pipeline has one authoritative implementation.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_loader import ID_COLUMN, load_test_data, load_training_data
from src.final_hybrid_detector import FinalHybridDetector, LOCKED_THRESHOLD

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = ROOT / "outputs" / "task1_predictions.csv"


def validate_task1_submission(predictions: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Raise a clear error if a Task 01 submission violates the official contract."""
    required_columns = ["Test_ID", "Validity_Label"]
    if list(predictions.columns) != required_columns:
        raise ValueError(f"Expected exactly {required_columns}; received {list(predictions.columns)}")
    if len(test_data) != 350 or len(predictions) != len(test_data):
        raise ValueError(f"Expected 350 input and output rows; got {len(test_data)} and {len(predictions)}")
    if predictions[ID_COLUMN].isna().any() or predictions[ID_COLUMN].duplicated().any():
        raise ValueError("Task 01 predictions contain missing or duplicate Test_ID values")
    if set(predictions[ID_COLUMN]) != set(test_data[ID_COLUMN]):
        raise ValueError("Task 01 Test_ID values do not exactly match Test_Data")
    if predictions["Validity_Label"].isna().any():
        raise ValueError("Task 01 predictions contain missing Validity_Label values")
    if not set(predictions["Validity_Label"]).issubset({"Valid", "Invalid"}):
        raise ValueError("Task 01 labels must be exactly Valid or Invalid")


def build_task1_submission(output_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Fit the locked detector on Training_Data and write validated Test_Data labels."""
    training = load_training_data()
    test_data = load_test_data()
    detector = FinalHybridDetector(threshold=LOCKED_THRESHOLD, random_state=42).fit(training)
    labels = detector.predict(test_data)
    predictions = pd.DataFrame({ID_COLUMN: test_data[ID_COLUMN].to_numpy(), "Validity_Label": labels})
    validate_task1_submission(predictions, test_data)

    destination = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    return predictions


if __name__ == "__main__":
    submission = build_task1_submission()
    print(f"Saved {len(submission)} validated Task 01 predictions to {DEFAULT_OUTPUT_PATH}")
