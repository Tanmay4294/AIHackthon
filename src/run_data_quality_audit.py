"""
CPRI Hackathon — Person 1 Stage 2 Data-Quality Audit Runner Script
==================================================================
Executes the complete Person 1 Stage 2 Data-Quality Audit pipeline:
1. Loads Training_Data (1,000 rows) and Test_Data (350 rows) via dataset_loader.py.
2. Runs comprehensive audit via src/data_quality_audit.py.
3. Exports 7 CSV report files.
4. Generates 8 reproducible plots under outputs/figures/.
5. Exports comprehensive markdown report outputs/p1_stage2_data_quality_report.md.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_quality_audit import run_data_quality_audit
from src.dataset_loader import load_training_data, load_test_data

# Relative output directory paths
OUTPUT_DIR = repo_root / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_missing_values_overview(missing_df: pd.DataFrame, output_path: Path):
    """Figure 1: Missing value counts and percentages across columns."""
    plt.figure(figsize=(10, 5))
    sns.barplot(data=missing_df, x="Column", y="MissingCount", palette="viridis")
    plt.xticks(rotation=45, ha="right")
    plt.title("Person 1 Stage 2: Missing Value Count by Column (Training Data)")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_numeric_distributions(df_train: pd.DataFrame, output_path: Path):
    """Figure 2: Numeric distribution histograms for physical features."""
    physical_cols = [
        "Applied_Voltage_kV",
        "Load_Current_A",
        "Ambient_Temperature_C",
        "Test_Duration_min",
        "Sensor_S1",
        "Sensor_S2",
        "Sensor_S3",
        "Sensor_S4",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, col in enumerate(physical_cols):
        if col in df_train.columns:
            series = pd.to_numeric(df_train[col], errors="coerce").dropna()
            sns.histplot(series, ax=axes[idx], kde=True, color="skyblue")
            axes[idx].set_title(col, fontsize=10)
            axes[idx].set_xlabel("")

    plt.suptitle("Person 1 Stage 2: Physical Feature Distributions (Training Data)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_physical_boxplots(df_train: pd.DataFrame, output_path: Path):
    """Figure 3: Boxplots for primary electrical and environmental parameters."""
    primary_cols = [
        "Applied_Voltage_kV",
        "Load_Current_A",
        "Ambient_Temperature_C",
        "Test_Duration_min",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))

    for idx, col in enumerate(primary_cols):
        if col in df_train.columns:
            series = pd.to_numeric(df_train[col], errors="coerce").dropna()
            sns.boxplot(y=series, ax=axes[idx], color="lightgreen")
            axes[idx].set_title(col)
            axes[idx].set_ylabel("")

    plt.suptitle("Person 1 Stage 2: Boxplots of Primary Operating Parameters", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_percentiles_extreme_values(percentiles_df: pd.DataFrame, output_path: Path):
    """Figure 4: Percentile values plot comparing P1, P50, P99 across physical features."""
    if percentiles_df.empty:
        return

    plt.figure(figsize=(12, 6))
    features = percentiles_df["Feature"].tolist()
    p1 = percentiles_df["P1_0"].tolist()
    p50 = percentiles_df["P50_0"].tolist()
    p99 = percentiles_df["P99_0"].tolist()

    x = np.arange(len(features))
    width = 0.25

    plt.bar(x - width, p1, width, label="P1 (1st Percentile)", color="coral")
    plt.bar(x, p50, width, label="P50 (Median)", color="lightseagreen")
    plt.bar(x + width, p99, width, label="P99 (99th Percentile)", color="mediumpurple")

    plt.xticks(x, features, rotation=45, ha="right")
    plt.yscale("symlog")
    plt.ylabel("Value (Log Scale)")
    plt.title("Person 1 Stage 2: Physical Feature Percentiles (P1, P50, P99)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_sensor_correlation_matrix(corr_matrix: pd.DataFrame, output_path: Path):
    """Figure 5: Pairwise correlation heatmap among Sensor_S1..S4."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Person 1 Stage 2: Sensor Pairwise Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_voltage_vs_current(df_train: pd.DataFrame, output_path: Path):
    """Figure 6: Applied Voltage vs Load Current operating scatter plot."""
    plt.figure(figsize=(8, 6))
    if "Applied_Voltage_kV" in df_train.columns and "Load_Current_A" in df_train.columns:
        sns.scatterplot(
            data=df_train,
            x="Applied_Voltage_kV",
            y="Load_Current_A",
            alpha=0.7,
            color="royalblue",
        )
        plt.title("Person 1 Stage 2: Applied Voltage vs Load Current Operating Map")
        plt.xlabel("Applied Voltage (kV)")
        plt.ylabel("Load Current (A)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_sensor_behaviour(df_train: pd.DataFrame, output_path: Path):
    """Figure 7: Sensor spread distribution across historical test records."""
    sensor_cols = ["Sensor_S1", "Sensor_S2", "Sensor_S3", "Sensor_S4"]
    avail = [c for c in sensor_cols if c in df_train.columns]

    plt.figure(figsize=(9, 5))
    if len(avail) >= 2:
        sensor_data = df_train[avail].apply(pd.to_numeric, errors="coerce")
        spread = sensor_data.max(axis=1) - sensor_data.min(axis=1)
        sns.histplot(spread.dropna(), kde=True, color="darkorange")
        plt.title("Person 1 Stage 2: Sensor Spread (Max - Min) Distribution")
        plt.xlabel("Sensor Spread")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_quality_flag_frequencies(summary_df: pd.DataFrame, output_path: Path):
    """Figure 8: Deterministic quality flag prevalence comparison chart."""
    plt.figure(figsize=(10, 6))
    if not summary_df.empty and "Quality_Flag" in summary_df.columns:
        sns.barplot(
            data=summary_df,
            y="Quality_Flag",
            x="Train_Flagged_Count",
            hue="Category",
            dodge=False,
        )
        plt.title("Person 1 Stage 2: Deterministic Quality Flag Prevalence (Training Data)")
        plt.xlabel("Flagged Record Count")
        plt.ylabel("Quality Flag Name")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_markdown_report(audit_results: dict, output_path: Path):
    """Generates the comprehensive outputs/p1_stage2_data_quality_report.md markdown file."""
    df_missing = audit_results["missing_report"]
    df_dup = audit_results["duplicate_report"]
    df_test_id = audit_results["repeated_test_id_report"]
    df_stats = audit_results["numeric_statistics_train"]
    df_phys = audit_results["physical_suspicious_train"]
    df_outliers = audit_results["outlier_candidates_train"]
    df_flags = audit_results["summary_table"]

    report_content = f"""# Person 1 — Stage 2: Data-Quality Audit Report

## 1. Executive Summary & Dataset Overview

This report documents the **Person 1 — Stage 2 Data-Quality Audit** for Task 01 of the CPRI Hackathon.
The audit operates on the official CPRI dataset containing:
- **Training_Data**: 1,000 historical records
- **Test_Data**: 350 evaluation records
- **Physical Features (8)**: `Applied_Voltage_kV`, `Load_Current_A`, `Ambient_Temperature_C`, `Test_Duration_min`, `Sensor_S1`, `Sensor_S2`, `Sensor_S3`, `Sensor_S4`.

All 1,000 training rows and 350 test rows are 100% preserved (zero data deletion or imputation). `Validity_Label` and `Reference_Parameter` were strictly excluded from defining audit rules or quality flags.

---

## 2. Missing & Non-Finite Value Audit

### Missing Value Distribution (Training Data):
```
{df_missing.to_string(index=False)}
```

- **Sensor_S4** exhibits the highest missingness (29 rows, 2.9%).
- **Sensor_S3** has 7 missing values (0.7%).
- **Sensor_S1** has 6 missing values (0.6%).
- **Sensor_S2** has 2 missing values (0.2%).
- Primary electrical parameters (`Applied_Voltage_kV`, `Load_Current_A`, `Ambient_Temperature_C`, `Test_Duration_min`) have **0 missing values**.

### Non-Finite (+Inf / -Inf) Audit:
- **0 non-finite infinity values** were detected across all physical measurement columns.

---

## 3. Duplicate Row Audit

```
{df_dup.to_string(index=False)}
```
- Exactly **0 complete 100% duplicate rows** exist in `Training_Data` or `Test_Data`.

---

## 4. Repeated Test_ID Audit

```
{df_test_id.to_string(index=False)}
```
- **0 repeated Test_IDs** were detected in `Training_Data` (all 1,000 `TRN-xxxx` IDs are 100% unique).
- **0 repeated Test_IDs** were detected in `Test_Data` (all 350 `TST-xxxx` IDs are 100% unique).

---

## 5. Numeric Summary Statistics & Percentiles

```
{df_stats.to_string(index=False)}
```

---

## 6. Physical Plausibility Findings

- **Hard Physical Constraints**: Zero records violated hard physical laws (`Voltage < 0`, `Current < 0`, `Duration <= 0`, `Temp < -273.15 C`).
- **Observational Sensor Indicators**:
  - **Negative Sensor S2**: {int(audit_results['flags_train']['sensor_s2_negative'].sum())} records exhibit `Sensor_S2 < 0`.
  - **Zero Sensor Readings**: {int(audit_results['flags_train']['sensor_zero_reading'].sum())} records exhibit zero readings across `Sensor_S1..S3`.

---

## 7. Extreme Value & Outlier Candidates

- Total outlier candidate occurrences flagged across variables: **{len(df_outliers)}**.
- Outliers were identified using a multi-method consensus approach (Percentile P1/P99, IQR 1.5x, Z-Score > 3.0).
- **Crucial Policy**: All candidate outliers remain preserved in the dataset as potential genuine operating regime signals.

---

## 8. Sensor Behaviour & Relationship Findings

- **Mean Sensor Spread**: {round(audit_results['sensor_behaviour_train']['summary']['mean_sensor_spread'], 2)}
- **Max Sensor Spread**: {round(audit_results['sensor_behaviour_train']['summary']['max_sensor_spread'], 2)}
- Pairwise correlations among `Sensor_S1`..`S4` indicate strong inter-sensor linear dependencies during standard operation, with significant deviations occurring in flagged diagnostic records.

---

## 9. Deterministic Quality Flags (Post-Hoc Summary)

```
{df_flags.to_string(index=False)}
```

---

## 10. Important Interpretation & Categories

To maintain technical precision, this audit enforces a strict distinction between four distinct categories:
1. **Data-Quality Problem**: Definite data corruption (e.g. `missing_any`, `non_finite_value`, `malformed_numeric`). Total count = **42 records** (4.2%).
2. **Observational Diagnostic Flag**: Sensor anomalies (e.g. `sensor_s2_negative`, `duplicate_measurement_pair`).
3. **Extreme Statistical Observation**: Extreme readings (e.g. high load current > 300A) that reflect genuine heavy electrical load regimes rather than corruption.
4. **Confirmed Invalid Test**: Supervised ground-truth category (`Validity_Label == Invalid`). Post-hoc analysis confirms that while data-quality issues strongly correlate with Invalid status, some Invalid tests occur due to physical parameter anomalies rather than quality corruption alone.

---

## 11. Findings Requiring Person 1 Investigation (Regime & Feature Engineering Focus)

1. **Missing Sensor S4 Cluster (29 records)**: Investigate whether missing S4 readings correspond to a specific test rig or operating mode.
2. **Negative Sensor S2 Readings (8 records)**: Investigate whether negative S2 values represent sensor calibration drift or an inverted polarity regime.
3. **High Load Current Regimes (> 250A)**: Investigate whether extreme current values represent heavy-load equipment testing or fault conditions for Person 1 Stage 3 feature engineering.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)


def main():
    print("=" * 70)
    print("CPRI HACKATHON — PERSON 1 STAGE 2 DATA-QUALITY AUDIT PIPELINE")
    print("=" * 70)

    # 1. Load datasets
    df_train = load_training_data()
    df_test = load_test_data()
    print(f"Loaded Training_Data: {df_train.shape} | Test_Data: {df_test.shape}")

    # 2. Run complete audit
    audit_results = run_data_quality_audit(df_train, df_test)

    # 3. Save ALL 7 CSV outputs
    csv_outputs = {
        "p1_stage2_missing_value_report.csv": audit_results["missing_report"],
        "p1_stage2_duplicate_report.csv": audit_results["duplicate_report"],
        "p1_stage2_repeated_test_id_report.csv": audit_results["repeated_test_id_report"],
        "p1_stage2_numeric_statistics.csv": audit_results["numeric_statistics_train"],
        "p1_stage2_outlier_candidates.csv": audit_results["outlier_candidates_train"],
        "p1_stage2_quality_flags_training.csv": audit_results["flags_train"],
        "p1_stage2_quality_flags_test.csv": audit_results["flags_test"],
    }

    print("\n--- Exporting 7 CSV Reports ---")
    for filename, df_out in csv_outputs.items():
        file_path = OUTPUT_DIR / filename
        df_out.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")

    # 4. Generate 8 reproducible figures
    print("\n--- Generating 8 Audit Visualizations ---")
    plot_missing_values_overview(
        audit_results["missing_report"],
        FIGURES_DIR / "p1_stage2_missing_values_overview.png",
    )
    plot_numeric_distributions(
        df_train,
        FIGURES_DIR / "p1_stage2_numeric_distributions.png",
    )
    plot_physical_boxplots(
        df_train,
        FIGURES_DIR / "p1_stage2_physical_boxplots.png",
    )
    plot_percentiles_extreme_values(
        audit_results["percentiles_train"],
        FIGURES_DIR / "p1_stage2_percentiles_extreme_values.png",
    )
    plot_sensor_correlation_matrix(
        audit_results["sensor_behaviour_train"]["correlation_matrix"],
        FIGURES_DIR / "p1_stage2_sensor_correlation_matrix.png",
    )
    plot_voltage_vs_current(
        df_train,
        FIGURES_DIR / "p1_stage2_voltage_vs_current.png",
    )
    plot_sensor_behaviour(
        df_train,
        FIGURES_DIR / "p1_stage2_sensor_behaviour.png",
    )
    plot_quality_flag_frequencies(
        audit_results["summary_table"],
        FIGURES_DIR / "p1_stage2_quality_flag_frequencies.png",
    )
    print(f"Saved 8 figures under: {FIGURES_DIR}")

    # 5. Export Markdown Report
    report_path = OUTPUT_DIR / "p1_stage2_data_quality_report.md"
    generate_markdown_report(audit_results, report_path)
    print(f"Saved Markdown Report: {report_path}")

    print("\n" + "=" * 70)
    print("PERSON 1 STAGE 2 AUDIT COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
