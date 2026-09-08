"""
CPRI Hackathon — Person 1 Stage 1 Dataset Loader Module
======================================================
Provides reusable, non-destructive, relative-path functions for loading
the CPRI Hackathon dataset across all 4 sheets:
- README
- Training_Data
- Test_Data
- Sample_Submission

Defines strict schema separation rules isolating physical features from identifiers
and target variables.
"""

from pathlib import Path
from typing import Dict, List, Union
import pandas as pd

# Relative dataset path resolution
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_FILENAME = "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx"
DATASET_PATH = DATA_DIR / DATASET_FILENAME

# Column definitions
PHYSICAL_FEATURES: List[str] = [
    "Applied_Voltage_kV",
    "Load_Current_A",
    "Ambient_Temperature_C",
    "Test_Duration_min",
    "Sensor_S1",
    "Sensor_S2",
    "Sensor_S3",
    "Sensor_S4",
]

ID_COLUMN: str = "Test_ID"
TASK01_TARGET: str = "Validity_Label"
TASK02_TARGET: str = "Reference_Parameter"
TARGET_COLUMNS: List[str] = [TASK02_TARGET, TASK01_TARGET]


def get_dataset_path() -> Path:
    """Returns absolute path to the dataset file using relative resolution."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found at: {DATASET_PATH}")
    return DATASET_PATH


def load_sheet(sheet_name: str) -> pd.DataFrame:
    """Loads a specific sheet from the Excel workbook without modifying raw data.

    Args:
        sheet_name: Name of the sheet ('README', 'Training_Data', 'Test_Data', 'Sample_Submission').

    Returns:
        pd.DataFrame containing sheet content.
    """
    path = get_dataset_path()
    return pd.read_excel(path, sheet_name=sheet_name)


def load_training_data() -> pd.DataFrame:
    """Loads the Training_Data sheet (1,000 rows x 11 columns).

    Returns:
        pd.DataFrame: Unmodified historical training records.
    """
    return load_sheet("Training_Data")


def load_test_data() -> pd.DataFrame:
    """Loads the Test_Data sheet (350 rows x 9 columns).

    Returns:
        pd.DataFrame: Unmodified test records without ground truth targets.
    """
    return load_sheet("Test_Data")


def load_readme() -> pd.DataFrame:
    """Loads the README sheet containing participant instructions and metadata.

    Returns:
        pd.DataFrame: Overview metadata sheet.
    """
    return load_sheet("README")


def load_sample_submission() -> pd.DataFrame:
    """Loads the Sample_Submission sheet (350 rows x 3 columns).

    Returns:
        pd.DataFrame: Format specification for test predictions.
    """
    return load_sheet("Sample_Submission")


def get_feature_schema() -> Dict[str, Union[List[str], str]]:
    """Returns the programmatically verified feature schema dictionary.

    Isolates physical input features from ID columns and target variables.

    Returns:
        Dict: Mapping schema categories to column names.
    """
    return {
        "physical_features": list(PHYSICAL_FEATURES),
        "id_column": ID_COLUMN,
        "task01_target": TASK01_TARGET,
        "task02_target": TASK02_TARGET,
        "target_columns": list(TARGET_COLUMNS),
    }


def extract_physical_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts strictly physical input features from a dataset DataFrame,

    guaranteeing exclusion of Test_ID, Validity_Label, and Reference_Parameter.

    Args:
        df: Input DataFrame (Training or Test).

    Returns:
        pd.DataFrame: DataFrame containing only physical input columns.
    """
    available = [c for c in PHYSICAL_FEATURES if c in df.columns]
    return df[available].copy()
