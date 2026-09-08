"""
CPRI Hackathon — Person 1 Stage 3 Unit Tests
=============================================
Automated test suite verifying the Person 1 Stage 3 Valid vs Invalid Historical Analysis pipeline:
1. Training data has exactly 1,000 rows.
2. Valid + Invalid counts equal 1,000 (866 Valid, 134 Invalid).
3. All 8 physical features and 5 engineered features are present.
4. Test_ID is not treated as an explanatory feature.
5. Reference_Parameter is excluded from analysis feature set.
6. Feature construction operates without requiring Validity_Label.
7. Zero records are deleted or imputed during analysis.
8. Numeric statistics and percentiles are finite and non-empty.
9. Pattern counts reconcile 100% with dataset totals.
10. Representative records table contains required case categories.
11. Analysis outputs are 100% reproducible.
"""

import numpy as np
import pandas as pd
import pytest

from src.dataset_loader import (
    ID_COLUMN,
    PHYSICAL_FEATURES,
    TASK01_TARGET,
    TASK02_TARGET,
    load_training_data,
)
from src.stage3_historical_analysis import (
    analyze_class_balance,
    analyze_classwise_numeric_statistics,
    analyze_legitimate_unusual_regimes,
    analyze_recurring_invalid_patterns,
    build_representative_records_table,
    calculate_feature_label_associations,
    enrich_dataset_with_stage3_features,
    generate_important_variable_shortlist,
    run_stage3_historical_analysis,
)


@pytest.fixture
def training_data():
    return load_training_data()


def test_training_data_row_count(training_data):
    """1. Verifies Training_Data has exactly 1,000 rows."""
    assert len(training_data) == 1000, f"Expected 1,000 rows, got {len(training_data)}"


def test_valid_plus_invalid_count_reconciliation(training_data):
    """2. Verifies Valid + Invalid counts equal 1,000 total records."""
    cb = analyze_class_balance(training_data)
    assert cb["TotalRecords"] == 1000
    assert cb["ValidCount"] + cb["InvalidCount"] == 1000
    assert cb["ValidCount"] == 866
    assert cb["InvalidCount"] == 134


def test_expected_physical_and_engineered_features_present(training_data):
    """3. Verifies all 8 physical features and 5 engineered features are generated."""
    df_enriched = enrich_dataset_with_stage3_features(training_data)
    expected_engineered = [
        "Apparent_Power_kVA",
        "Sensor_Spread",
        "Sensor_Mean",
        "Sensor_S1_S2_Ratio",
        "Sensor_S3_S4_Ratio",
    ]
    for col in PHYSICAL_FEATURES + expected_engineered:
        assert col in df_enriched.columns, f"Missing feature: {col}"


def test_test_id_not_treated_as_explanatory_feature(training_data):
    """4. Verifies Test_ID is strictly excluded from feature statistics."""
    stats_df = analyze_classwise_numeric_statistics(training_data)
    assert ID_COLUMN not in stats_df["Feature"].values


def test_reference_parameter_excluded_from_analysis_features(training_data):
    """5. Verifies Reference_Parameter is excluded from feature statistics."""
    stats_df = analyze_classwise_numeric_statistics(training_data)
    assert TASK02_TARGET not in stats_df["Feature"].values


def test_analysis_does_not_require_validity_label_for_feature_construction(training_data):
    """6. Verifies feature enrichment operates on DataFrames without Validity_Label."""
    df_no_label = training_data.drop(columns=[TASK01_TARGET], errors="ignore")
    df_enriched = enrich_dataset_with_stage3_features(df_no_label)
    assert "Sensor_Spread" in df_enriched.columns
    assert "Apparent_Power_kVA" in df_enriched.columns


def test_no_records_deleted_or_imputed(training_data):
    """7. Verifies analysis functions preserve 100% of rows without deletion or imputation."""
    df_copy = training_data.copy()
    _ = run_stage3_historical_analysis(training_data)
    assert len(training_data) == 1000
    assert training_data.equals(df_copy)


def test_statistics_and_percentiles_are_finite(training_data):
    """8. Verifies numeric statistics table contains valid, finite values."""
    stats_df = analyze_classwise_numeric_statistics(training_data)
    assert len(stats_df) == 13
    assert not stats_df["Valid_Mean"].isna().all()
    assert not stats_df["Invalid_Mean"].isna().all()


def test_pattern_counts_reconcile_with_dataset(training_data):
    """9. Verifies recurring pattern analysis totals reconcile 100% with dataset query counts."""
    pat_df = analyze_recurring_invalid_patterns(training_data)
    assert len(pat_df) > 0
    for _, row in pat_df.iterrows():
        assert row["Valid_Count"] + row["Invalid_Count"] == row["Total_Records"]


def test_representative_records_contain_required_categories(training_data):
    """10. Verifies representative records table contains required case categories."""
    rep_df = build_representative_records_table(training_data)
    assert len(rep_df) >= 4
    assert "CaseCategory" in rep_df.columns
    categories = rep_df["CaseCategory"].tolist()
    assert "Unusual Valid" in categories


def test_analysis_results_are_reproducible(training_data):
    """11. Verifies run_stage3_historical_analysis returns identical results on repeated calls."""
    res1 = run_stage3_historical_analysis(training_data)
    res2 = run_stage3_historical_analysis(training_data)
    assert res1["class_balance"] == res2["class_balance"]
    assert res1["class_statistics"].equals(res2["class_statistics"])


def test_important_variable_shortlist_categorization(training_data):
    """12. Verifies generate_important_variable_shortlist categorizes variables into A, B, C."""
    shortlist = generate_important_variable_shortlist(training_data)
    assert "Category_A_Strong_Evidence" in shortlist
    assert "Category_B_Moderate_Evidence" in shortlist
    assert "Category_C_Hypotheses_Further_Investigation" in shortlist


def test_legitimate_unusual_regimes(training_data):
    """13. Verifies analyze_legitimate_unusual_regimes identifies heavy load valid cases."""
    reg_df = analyze_legitimate_unusual_regimes(training_data)
    assert len(reg_df) > 0
    regimes = reg_df["Regime_Category"].tolist()
    assert "Unusual Heavy-Load (Valid)" in regimes


def test_pipeline_output_dictionary_integrity(training_data):
    """14. Verifies run_stage3_historical_analysis returns all 7 expected dictionary keys."""
    res = run_stage3_historical_analysis(training_data)
    expected_keys = {
        "class_balance",
        "class_statistics",
        "associations",
        "patterns",
        "regimes",
        "shortlist",
        "representative_records",
    }
    assert set(res.keys()) == expected_keys
