"""Regression tests for the final detector and official Test_Data output."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset_loader import load_test_data, load_training_data
from src.final_hybrid_detector import FinalHybridDetector, LOCKED_THRESHOLD

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def datasets():
    return load_training_data(), load_test_data()


@pytest.fixture(scope="module")
def fitted_detector(datasets):
    train, _ = datasets
    return FinalHybridDetector(random_state=42).fit(train)


@pytest.fixture(scope="module")
def test_diagnostics(fitted_detector, datasets):
    _, test = datasets
    return fitted_detector.predict_with_diagnostics(test)


def test_detector_can_be_instantiated():
    detector = FinalHybridDetector(random_state=42)
    assert detector.threshold == LOCKED_THRESHOLD


def test_training_data_can_be_fitted(fitted_detector):
    assert fitted_detector.is_fitted
    assert fitted_detector.feature_names


def test_required_feature_generation_succeeds(fitted_detector, datasets):
    _, test = datasets
    probabilities = fitted_detector.predict_proba(test.head(5))
    assert len(probabilities) == 5
    assert np.isfinite(probabilities).all()


def test_predictive_features_exclude_identifiers_and_targets(fitted_detector):
    assert "Test_ID" not in fitted_detector.feature_names
    assert "Validity_Label" not in fitted_detector.feature_names
    assert "Reference_Parameter" not in fitted_detector.feature_names


def test_final_threshold_is_reproducible():
    assert FinalHybridDetector(random_state=42).threshold == FinalHybridDetector(random_state=42).threshold == 0.40


def test_final_predictions_are_binary(test_diagnostics):
    assert set(test_diagnostics["Final_Validity_Label"]).issubset({"Valid", "Invalid"})


def test_no_individual_test_id_exceptions():
    source = (ROOT / "src" / "final_hybrid_detector.py").read_text(encoding="utf-8")
    assert "if Test_ID" not in source
    assert "TST-" not in source


def test_deterministic_quality_rules_are_inference_safe(fitted_detector, datasets):
    _, test = datasets
    assert "Validity_Label" not in test.columns
    assert "Reference_Parameter" not in test.columns
    diagnostics = fitted_detector.predict_with_diagnostics(test.head(3))
    assert "Hard_Quality_Evidence" in diagnostics


def test_extreme_operating_regimes_are_not_direct_rules():
    source = (ROOT / "src" / "final_hybrid_detector.py").read_text(encoding="utf-8")
    assert "Load_Current_A >" not in source
    assert "Applied_Voltage_kV >" not in source
    assert "Ambient_Temperature_C >" not in source


def test_final_detector_reproducible(datasets):
    train, test = datasets
    one = FinalHybridDetector(random_state=42).fit(train).predict_proba(test.head(20))
    two = FinalHybridDetector(random_state=42).fit(train).predict_proba(test.head(20))
    np.testing.assert_array_equal(one, two)


def test_test_data_has_exactly_350_rows(datasets):
    assert len(datasets[1]) == 350


def test_final_output_has_exactly_350_rows(test_diagnostics):
    assert len(test_diagnostics) == 350


def test_test_id_is_unique(test_diagnostics):
    assert test_diagnostics["Test_ID"].nunique() == 350


def test_output_ids_match_input(datasets, test_diagnostics):
    assert set(test_diagnostics["Test_ID"]) == set(datasets[1]["Test_ID"])


def test_no_prediction_is_missing(test_diagnostics):
    assert test_diagnostics["Final_Validity_Label"].notna().all()


def test_labels_are_only_valid_or_invalid(test_diagnostics):
    assert set(test_diagnostics["Final_Validity_Label"]).issubset({"Valid", "Invalid"})


def test_test_data_never_used_for_threshold_optimization():
    source = (ROOT / "src" / "run_final_detector.py").read_text(encoding="utf-8")
    assert "evaluate_thresholds(test" not in source
    assert "select_threshold(test" not in source
    assert "Validity_Label" not in load_test_data().columns


def test_official_submission_has_exact_columns():
    path = ROOT / "outputs" / "task1_predictions.csv"
    assert path.exists()
    assert list(pd.read_csv(path).columns) == ["Test_ID", "Validity_Label"]


def test_existing_stage1_to_stage8_tests_are_present():
    for stage in range(1, 9):
        assert list((ROOT / "tests").glob(f"test_person1_stage{stage}_*.py"))


def test_existing_person2_tests_are_present():
    assert (ROOT / "tests" / "test_quality_features.py").exists()
    assert (ROOT / "tests" / "test_stage2_classification.py").exists()
    assert (ROOT / "tests" / "test_stage3_anomaly_detection.py").exists()
    assert (ROOT / "tests" / "test_stage4_final_validity.py").exists()


def test_raw_workbook_is_not_modified():
    assert (ROOT / "data" / "CPRI_Hackathon_Screening_Dataset_PARTICIPANT.xlsx").exists()

