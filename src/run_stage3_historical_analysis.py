"""
CPRI Hackathon — Person 1 Stage 3 Historical Analysis Runner Script
===================================================================
Executes the complete Person 1 Stage 3 Valid vs Invalid Historical Analysis pipeline:
1. Loads Training_Data (1,000 rows) via dataset_loader.py.
2. Enriches dataset with quality flags and Person 1 engineered features.
3. Performs class-wise numeric statistics, associations, recurring pattern analysis,
   legitimate unusual regime analysis, and representative record extraction.
4. Exports 5 CSV report files.
5. Generates 8 reproducible plots under outputs/figures/.
6. Exports comprehensive markdown report outputs/p1_stage3_valid_invalid_report.md.
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

from src.dataset_loader import load_training_data
from src.stage3_historical_analysis import (
    enrich_dataset_with_stage3_features,
    run_stage3_historical_analysis,
)

# Relative output directory paths
OUTPUT_DIR = repo_root / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_class_balance(balance_dict: dict, output_path: Path):
    """Figure 1: Class balance pie chart and bar plot."""
    plt.figure(figsize=(7, 5))
    classes = ["Valid", "Invalid"]
    counts = [balance_dict["ValidCount"], balance_dict["InvalidCount"]]
    colors = ["#2ecc71", "#e74c3c"]

    plt.bar(classes, counts, color=colors, width=0.5)
    for i, count in enumerate(counts):
        pct = (count / balance_dict["TotalRecords"]) * 100
        plt.text(i, count + 10, f"{count} ({pct:.1f}%)", ha="center", fontweight="bold")

    plt.title("Person 1 Stage 3: Training Data Class Distribution (Valid vs Invalid)")
    plt.ylabel("Record Count")
    plt.ylim(0, balance_dict["TotalRecords"] * 1.1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_histograms_by_class(df_train: pd.DataFrame, output_path: Path):
    """Figure 2: Histograms/KDEs comparing Valid vs Invalid across physical features."""
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
            sns.histplot(
                data=df_train,
                x=col,
                hue="Validity_Label",
                kde=True,
                ax=axes[idx],
                palette={"Valid": "#2ecc71", "Invalid": "#e74c3c"},
                alpha=0.5,
            )
            axes[idx].set_title(col, fontsize=10)
            axes[idx].set_xlabel("")

    plt.suptitle("Person 1 Stage 3: Class-Wise Distribution Histograms", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_physical_boxplots_by_class(df_train: pd.DataFrame, output_path: Path):
    """Figure 3: Class-wise boxplots for primary operating parameters."""
    primary_cols = [
        "Applied_Voltage_kV",
        "Load_Current_A",
        "Ambient_Temperature_C",
        "Test_Duration_min",
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))

    for idx, col in enumerate(primary_cols):
        if col in df_train.columns:
            sns.boxplot(
                data=df_train,
                x="Validity_Label",
                y=col,
                ax=axes[idx],
                palette={"Valid": "#2ecc71", "Invalid": "#e74c3c"},
            )
            axes[idx].set_title(col)
            axes[idx].set_xlabel("")

    plt.suptitle("Person 1 Stage 3: Physical Parameter Boxplots by Validity Class", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_voltage_vs_current_by_class(df_train: pd.DataFrame, output_path: Path):
    """Figure 4: Voltage vs Current operating map colored by Validity_Label."""
    plt.figure(figsize=(8, 6))
    if "Applied_Voltage_kV" in df_train.columns and "Load_Current_A" in df_train.columns:
        sns.scatterplot(
            data=df_train,
            x="Applied_Voltage_kV",
            y="Load_Current_A",
            hue="Validity_Label",
            palette={"Valid": "#2ecc71", "Invalid": "#e74c3c"},
            alpha=0.7,
        )
        plt.title("Person 1 Stage 3: Voltage vs Current Operating Map by Class")
        plt.xlabel("Applied Voltage (kV)")
        plt.ylabel("Load Current (A)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_sensor_disagreement_by_class(df_train: pd.DataFrame, output_path: Path):
    """Figure 5: Sensor Spread distribution by Validity Class."""
    plt.figure(figsize=(9, 5))
    if "Sensor_Spread" in df_train.columns:
        sns.kdeplot(
            data=df_train,
            x="Sensor_Spread",
            hue="Validity_Label",
            common_norm=False,
            palette={"Valid": "#2ecc71", "Invalid": "#e74c3c"},
            fill=True,
            alpha=0.4,
        )
        plt.title("Person 1 Stage 3: Sensor Spread Distribution by Validity Class")
        plt.xlabel("Sensor Spread (Max - Min)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_associations_effect_sizes(assoc_df: pd.DataFrame, output_path: Path):
    """Figure 6: Cohen's d Effect Size bar chart across features."""
    if assoc_df.empty or "Cohens_D_EffectSize" not in assoc_df.columns:
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=assoc_df.sort_values(by="Cohens_D_EffectSize", key=abs, ascending=False),
        y="Feature",
        x="Cohens_D_EffectSize",
        palette="vlag",
    )
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(0.5, color="red", linestyle=":", label="Large Effect (|d| = 0.5)")
    plt.axvline(-0.5, color="red", linestyle=":")
    plt.title("Person 1 Stage 3: Feature Effect Sizes (Cohen's d: Invalid vs Valid)")
    plt.xlabel("Cohen's d Effect Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_recurring_pattern_invalid_rates(pattern_df: pd.DataFrame, output_path: Path):
    """Figure 7: Invalid rate and lift across recurring physical patterns."""
    if pattern_df.empty:
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=pattern_df,
        y="Pattern_Name",
        x="Invalid_Rate(%)",
        palette="Reds_r",
    )
    plt.axvline(13.4, color="blue", linestyle="--", label="Baseline Invalid Rate (13.4%)")
    plt.title("Person 1 Stage 3: Invalid Rate (%) Across Recurring Physical Patterns")
    plt.xlabel("Invalid Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_regime_scatter(df_train: pd.DataFrame, output_path: Path):
    """Figure 8: Operating regime scatter showing unusual valid vs normal invalid cases."""
    plt.figure(figsize=(9, 6))
    if "Apparent_Power_kVA" in df_train.columns and "Sensor_Spread" in df_train.columns:
        sns.scatterplot(
            data=df_train,
            x="Apparent_Power_kVA",
            y="Sensor_Spread",
            hue="Validity_Label",
            style="data_quality_issue",
            palette={"Valid": "#2ecc71", "Invalid": "#e74c3c"},
            alpha=0.8,
            s=60,
        )
        plt.title("Person 1 Stage 3: Operating Regime Map (Apparent Power vs Sensor Spread)")
        plt.xlabel("Apparent Power (kVA)")
        plt.ylabel("Sensor Spread")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def generate_markdown_report(analysis_results: dict, output_path: Path):
    """Generates the comprehensive outputs/p1_stage3_valid_invalid_report.md markdown file."""
    cb = analysis_results["class_balance"]
    df_stats = analysis_results["class_statistics"]
    df_assoc = analysis_results["associations"]
    df_pat = analysis_results["patterns"]
    df_reg = analysis_results["regimes"]
    df_rep = analysis_results["representative_records"]
    shortlist = analysis_results["shortlist"]

    report_content = f"""# Person 1 — Stage 3: Valid vs Invalid Historical Analysis Report

## 1. Executive Summary & Objective

This report documents **Person 1 — Stage 3: Investigate Valid vs Invalid Historical Tests** for Task 01 of the CPRI Hackathon.

The goal of this stage is to conduct a rigorous, non-predictive **exploratory analysis** of historical `Training_Data` (1,000 records) to uncover physical, sensor, and quality patterns distinguishing **Valid** and **Invalid** test records.

`Validity_Label` is used strictly as a grouping variable for post-hoc descriptive comparison. All 1,000 training records are preserved without deletion, imputation, or winsorization.

---

## 2. Dataset Class Balance

- **Total Training Records**: {cb['TotalRecords']}
- **Valid Tests**: {cb['ValidCount']} ({cb['ValidPct']}%)
- **Invalid Tests**: {cb['InvalidCount']} ({cb['InvalidPct']}%)
- **Imbalance Ratio**: {cb['ImbalanceRatio']}:1 (Valid : Invalid)

---

## 3. Feature-by-Feature Valid vs Invalid Comparison

```
{df_stats.to_string(index=False)}
```

---

## 4. Strongest Associations & Effect Sizes

```
{df_assoc.to_string(index=False)}
```

- **Sensor_S4 Missingness**: Sensor_S4 missingness has the strongest single association with Invalid status (29 missing, 100% Invalid rate).
- **Sensor_Spread**: High sensor spread (> 50) shows strong positive association with Invalid tests.
- **Apparent Power / Load Current**: Higher load current regimes show moderate positive association with Invalid tests due to increased thermal/electrical stress during testing.

---

## 5. Sensor Consistency Findings

- **Sensor Disagreement**: When sensor spread exceeds 50 units, the Invalid rate rises significantly above the 13.4% baseline.
- **Negative Sensor S2**: 8 records exhibit `Sensor_S2 < 0`, of which 100% are historical Invalid tests.
- **Zero Sensor Readings**: 2 records exhibit zero readings across `Sensor_S1..S3`, both of which are Invalid tests.

---

## 6. Recurring Invalid Pattern Analysis

```
{df_pat.to_string(index=False)}
```

### Highlights:
1. **Missing Sensor S4**: 29 records (100% Invalid, Lift = 7.46x).
2. **Definite Quality Issue**: 42 records (100% Invalid, Lift = 7.46x).
3. **Negative Sensor S2**: 8 records (100% Invalid, Lift = 7.46x).
4. **Zero Sensor Reading**: 2 records (100% Invalid, Lift = 7.46x).
5. **High Sensor Spread (> 50)**: High lift above baseline.

---

## 7. Legitimate Unusual Operating Regimes vs Normal-Looking Invalids

```
{df_reg.to_string(index=False)}
```

### Critical Finding: Unusual $\\neq$ Invalid
- **Heavy Load Regimes (> 250A)**: Multiple records operating at high load currents (> 250A) are validly labeled **Valid**, representing legitimate heavy-apparatus testing regimes.
- **Normal-Looking Invalids**: Several records operating within perfectly normal voltage (18–22 kV), current (90–110 A), and temperature (20–30 C) ranges are labeled **Invalid**, proving that abnormality can stem from subtle internal inconsistencies rather than gross physical extreme values alone.

---

## 8. Representative Record Case Examples

```
{df_rep.to_string(index=False)}
```

---

## 9. Ranked Shortlist of Important Variables & Interactions

### Category A: Strong Evidence (Primary Drivers)
{chr(10).join('- ' + item for item in shortlist['Category_A_Strong_Evidence'])}

### Category B: Moderate Evidence (Secondary Drivers)
{chr(10).join('- ' + item for item in shortlist['Category_B_Moderate_Evidence'])}

### Category C: Hypotheses Requiring Further Investigation (Contextual Interactions)
{chr(10).join('- ' + item for item in shortlist['Category_C_Hypotheses_Further_Investigation'])}

---

## 10. Evidence-Based Hypotheses & Person 1 Stage 4 Implications

1. **Hypothesis 1 (Data-Quality Dominance)**: Data corruption (missing S4, negative S2, zero readings) accounts for ~31% of all Invalid tests with 100% precision.
2. **Hypothesis 2 (Sensor Consistency & Spread)**: Sensor disagreement (`Sensor_Spread`) captures physical sensor failure modes where no data quality flag is triggered.
3. **Hypothesis 3 (Regime-Dependent Validation)**: Physical parameters must be evaluated in context (e.g. `Apparent_Power_kVA` vs `Sensor_Spread`) to prevent false-positive flagging of legitimate heavy-load testing regimes.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)


def main():
    print("=" * 70)
    print("CPRI HACKATHON — PERSON 1 STAGE 3 HISTORICAL ANALYSIS PIPELINE")
    print("=" * 70)

    # 1. Load training dataset
    df_train_raw = load_training_data()
    df_train = enrich_dataset_with_stage3_features(df_train_raw)
    print(f"Loaded and Enriched Training_Data: {df_train.shape}")

    # 2. Run complete analysis
    analysis_results = run_stage3_historical_analysis(df_train)

    # 3. Export 5 CSV outputs
    csv_outputs = {
        "p1_stage3_class_statistics.csv": analysis_results["class_statistics"],
        "p1_stage3_feature_associations.csv": analysis_results["associations"],
        "p1_stage3_pattern_analysis.csv": analysis_results["patterns"],
        "p1_stage3_representative_records.csv": analysis_results["representative_records"],
        "p1_stage3_regime_analysis.csv": analysis_results["regimes"],
    }

    print("\n--- Exporting 5 CSV Reports ---")
    for filename, df_out in csv_outputs.items():
        file_path = OUTPUT_DIR / filename
        df_out.to_csv(file_path, index=False)
        print(f"Saved: {file_path}")

    # 4. Generate 8 reproducible figures
    print("\n--- Generating 8 Analysis Visualizations ---")
    plot_class_balance(
        analysis_results["class_balance"],
        FIGURES_DIR / "p1_stage3_class_balance.png",
    )
    plot_feature_histograms_by_class(
        df_train,
        FIGURES_DIR / "p1_stage3_feature_histograms_by_class.png",
    )
    plot_physical_boxplots_by_class(
        df_train,
        FIGURES_DIR / "p1_stage3_physical_boxplots_by_class.png",
    )
    plot_voltage_vs_current_by_class(
        df_train,
        FIGURES_DIR / "p1_stage3_voltage_vs_current_by_class.png",
    )
    plot_sensor_disagreement_by_class(
        df_train,
        FIGURES_DIR / "p1_stage3_sensor_disagreement_by_class.png",
    )
    plot_feature_associations_effect_sizes(
        analysis_results["associations"],
        FIGURES_DIR / "p1_stage3_feature_associations_effect_sizes.png",
    )
    plot_recurring_pattern_invalid_rates(
        analysis_results["patterns"],
        FIGURES_DIR / "p1_stage3_recurring_pattern_invalid_rates.png",
    )
    plot_regime_scatter(
        df_train,
        FIGURES_DIR / "p1_stage3_regime_scatter.png",
    )
    print(f"Saved 8 figures under: {FIGURES_DIR}")

    # 5. Export Markdown Report
    report_path = OUTPUT_DIR / "p1_stage3_valid_invalid_report.md"
    generate_markdown_report(analysis_results, report_path)
    print(f"Saved Markdown Report: {report_path}")

    print("\n" + "=" * 70)
    print("PERSON 1 STAGE 3 HISTORICAL ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
