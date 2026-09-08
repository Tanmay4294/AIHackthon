"""Integration contract tests for the competition-ready Task 01 deliverable."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.build_task1_submission import build_task1_submission, validate_task1_submission
from src.dataset_loader import load_test_data, load_training_data
from src.final_hybrid_detector import FinalHybridDetector, LOCKED_THRESHOLD

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def official_submission():
    return pd.read_csv(ROOT / "outputs" / "task1_predictions.csv")


def test_task1_predictions_exists(official_submission):
    assert len(official_submission) == 350


def test_task1_prediction_ids_match_test_data(official_submission):
    test_data = load_test_data()
    assert official_submission["Test_ID"].is_unique
    assert set(official_submission["Test_ID"]) == set(test_data["Test_ID"])


def test_task1_labels_and_schema_are_official(official_submission):
    assert list(official_submission.columns) == ["Test_ID", "Validity_Label"]
    assert official_submission["Validity_Label"].notna().all()
    assert set(official_submission["Validity_Label"]).issubset({"Valid", "Invalid"})


def test_final_features_exclude_id_and_targets():
    detector = FinalHybridDetector().fit(load_training_data())
    assert "Test_ID" not in detector.feature_names
    assert "Reference_Parameter" not in detector.feature_names
    assert "Validity_Label" not in detector.feature_names


def test_locked_configuration_matches_documented_config():
    config = json.loads((ROOT / "outputs" / "p1_stage11_final_config.json").read_text(encoding="utf-8"))
    assert config["model"] == "Random_Forest"
    assert config["feature_set"] == "Set_D_Full_Stage5"
    assert config["decision_threshold"] == LOCKED_THRESHOLD == 0.40
    assert config["classifier"]["n_estimators"] == 200
    assert config["classifier"]["class_weight"] == "balanced"


def test_builder_is_reproducible_without_manual_editing(tmp_path, official_submission):
    first = build_task1_submission(tmp_path / "first.csv")
    second = build_task1_submission(tmp_path / "second.csv")
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, official_submission)


def test_submission_validator_rejects_invalid_labels():
    test_data = load_test_data()
    bad = pd.DataFrame({"Test_ID": test_data["Test_ID"], "Validity_Label": ["Unknown"] * len(test_data)})
    with pytest.raises(ValueError, match="labels"):
        validate_task1_submission(bad, test_data)


def test_raw_workbook_and_no_id_specific_logic():
    assert (ROOT / "data" / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx").exists()
    source = (ROOT / "src" / "final_hybrid_detector.py").read_text(encoding="utf-8")
    assert "if Test_ID" not in source
    assert "TST-" not in source


def test_previous_stage_tests_remain_present():
    assert (ROOT / "tests" / "test_person1_stage9_10_final_detector.py").exists()
    assert (ROOT / "tests" / "test_stage4_final_validity.py").exists()
