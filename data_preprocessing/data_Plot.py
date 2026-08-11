import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# File paths
# ============================================================
input_file = r"D:\WORK\number_final.xlsx"
main_output_folder = r"D:\WORK\All_Columns_Analysis_Final"

os.makedirs(main_output_folder, exist_ok=True)


# ============================================================
# Read Excel file
# ============================================================
# If real column names are in the second row, use header=1
df = pd.read_excel(input_file, header=1)

# If real column names are in the first row, use this instead:
# df = pd.read_excel(input_file)

df.columns = df.columns.astype(str).str.strip()


# ============================================================
# Limit analysis to rows up to a specific Excel row
# ============================================================
# If you want to analyze all rows, set this to None
# Example: if analyze_until_excel_row = 100, the code analyzes data up to Excel row 100
analyze_until_excel_row = 96

# Because header=1 means:
# Excel row 1 = ignored
# Excel row 2 = column names
# Excel row 3 = first data row
if analyze_until_excel_row is not None:
    number_of_data_rows = analyze_until_excel_row - 2

    if number_of_data_rows <= 0:
        raise ValueError("analyze_until_excel_row must be 3 or greater when header=1.")

    df = df.head(number_of_data_rows)

    print(f"Analysis limited to Excel row: {analyze_until_excel_row}")
    print(f"Number of data rows analyzed: {len(df)}")
else:
    print("Analysis will be performed on all rows.")


# ============================================================
# Columns to exclude from analysis
# ============================================================
exclude_columns = [
    "Rec_code"
]


# ============================================================
# Helper functions
# ============================================================
def safe_folder_name(name):
    """
    Create a safe folder name from column name.
    """
    name = str(name).strip()
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = name.replace(" ", "_")
    return name


def clean_numeric_series(series):
    """
    Convert a column to numeric.
    Non-numeric values and Excel errors become NaN.
    """
    cleaned = (
        series
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace({
            "": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "NAN": np.nan,
            "None": np.nan,
            "none": np.nan,
            "null": np.nan,
            "NULL": np.nan,
            "-": np.nan,
            "–": np.nan,
            "—": np.nan,
            "−": np.nan,
            "?": np.nan,
            "؟": np.nan,
            ".": np.nan,
            "..": np.nan,
            "...": np.nan,
            "#DIV/0!": np.nan,
            "#VALUE!": np.nan,
            "#REF!": np.nan,
            "#N/A": np.nan,
            "#NAME?": np.nan,
            "#NUM!": np.nan,
            "#NULL!": np.nan,
        })
    )

    return pd.to_numeric(cleaned, errors="coerce")


def save_plot(output_folder, filename):
    """
    Save current matplotlib figure.
    """
    plt.tight_layout()
    path = os.path.join(output_folder, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def analyze_numeric_column(df, target_col, main_output_folder):
    """
    Analyze one numeric column and save results in a separate folder.
    """

    folder_name = safe_folder_name(target_col)
    output_folder = os.path.join(main_output_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    output_excel = os.path.join(output_folder, f"{folder_name}_analysis_report.xlsx")

    analysis_df = df.copy()

    # Convert target column to numeric
    analysis_df[target_col] = clean_numeric_series(analysis_df[target_col])
    values = analysis_df[target_col].dropna()

    total_count = len(analysis_df)
    valid_count = analysis_df[target_col].notna().sum()
    missing_count = analysis_df[target_col].isna().sum()

    valid_percent = valid_count / total_count * 100 if total_count > 0 else 0
    missing_percent = missing_count / total_count * 100 if total_count > 0 else 0

    missing_summary = pd.DataFrame({
        "Status": ["Available / Non-missing", "Missing / NaN"],
        "Count": [valid_count, missing_count],
        "Percent": [valid_percent, missing_percent]
    })

    # If no valid numeric values exist, save missing plot and simple report
    if len(values) == 0:
        summary_stats = pd.DataFrame({
            "Metric": ["Valid count", "Missing count", "Note"],
            "Value": [0, missing_count, "No valid numeric values found"]
        })

        # Missing vs available plot
        plt.figure(figsize=(7, 5))
        plt.bar(missing_summary["Status"], missing_summary["Count"])
        plt.ylabel("Number of records")
        plt.title(f"Missing vs Available Data for {target_col}")
        for i, row in missing_summary.iterrows():
            plt.text(
                i,
                row["Count"],
                f'{row["Count"]} ({row["Percent"]:.1f}%)',
                ha="center",
                va="bottom"
            )
        plt.xticks(rotation=15, ha="right")
        save_plot(output_folder, "05_missing_vs_available.png")

        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            summary_stats.to_excel(writer, sheet_name="Summary_Statistics", index=False)
            missing_summary.to_excel(writer, sheet_name="Missing_Summary", index=False)

        print(f"Skipped {target_col}: no valid numeric values.")
        return

    # ============================================================
    # Descriptive statistics
    # ============================================================
    summary_stats = pd.DataFrame({
        "Metric": [
            "Valid count",
            "Missing count",
            "Mean",
            "Median",
            "Standard deviation",
            "Minimum",
            "Maximum",
            "Q1 / 25th percentile",
            "Q3 / 75th percentile",
            "IQR",
            "P5",
            "P10",
            "P20",
            "P30",
            "P40",
            "P50",
            "P60",
            "P70",
            "P80",
            "P90",
            "P95",
            "Skewness",
            "Kurtosis",
            "Number of unique values"
        ],
        "Value": [
            values.count(),
            missing_count,
            values.mean(),
            values.median(),
            values.std(),
            values.min(),
            values.max(),
            values.quantile(0.25),
            values.quantile(0.75),
            values.quantile(0.75) - values.quantile(0.25),
            values.quantile(0.05),
            values.quantile(0.10),
            values.quantile(0.20),
            values.quantile(0.30),
            values.quantile(0.40),
            values.quantile(0.50),
            values.quantile(0.60),
            values.quantile(0.70),
            values.quantile(0.80),
            values.quantile(0.90),
            values.quantile(0.95),
            values.skew(),
            values.kurtosis(),
            values.nunique()
        ]
    })

    # ============================================================
    # Frequency table
    # ============================================================
    value_counts = (
        values.value_counts()
        .sort_index()
        .reset_index()
    )

    value_counts.columns = [target_col, "Count"]
    value_counts["Percent"] = value_counts["Count"] / value_counts["Count"].sum() * 100
    value_counts["Cumulative_Count"] = value_counts["Count"].cumsum()
    value_counts["Cumulative_Percent"] = value_counts["Percent"].cumsum()

    # ============================================================
    # Outlier detection using IQR
    # ============================================================
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_col = f"{target_col}_is_outlier_IQR"

    analysis_df[outlier_col] = np.where(
        (analysis_df[target_col] < lower_bound) | (analysis_df[target_col] > upper_bound),
        1,
        0
    )

    # Do not mark missing values as outliers
    analysis_df.loc[analysis_df[target_col].isna(), outlier_col] = np.nan

    outlier_summary = pd.DataFrame({
        "Metric": [
            "Q1",
            "Q3",
            "IQR",
            "Lower outlier bound",
            "Upper outlier bound",
            "Number of outliers",
            "Percent of outliers"
        ],
        "Value": [
            q1,
            q3,
            iqr,
            lower_bound,
            upper_bound,
            analysis_df[outlier_col].sum(),
            analysis_df[outlier_col].sum() / values.count() * 100
        ]
    })

    outlier_rows = analysis_df[analysis_df[outlier_col] == 1].copy()

    # ============================================================
    # Percentile table
    # ============================================================
    percentile_points = list(range(0, 101, 5))
    percentile_values = np.percentile(values, percentile_points)

    percentile_table = pd.DataFrame({
        "Percentile": percentile_points,
        target_col: percentile_values
    })

    # ============================================================
    # Plot 1: Histogram
    # ============================================================
    plt.figure(figsize=(9, 6))
    plt.hist(values, bins=20, edgecolor="black")
    plt.xlabel(target_col)
    plt.ylabel("Number of patients")
    plt.title(f"Distribution of {target_col}")
    save_plot(output_folder, "01_histogram.png")

    # ============================================================
    # Plot 2: Histogram with mean and median
    # ============================================================
    plt.figure(figsize=(9, 6))
    plt.hist(values, bins=20, edgecolor="black")
    plt.axvline(values.mean(), linestyle="--", linewidth=2, label=f"Mean = {values.mean():.2f}")
    plt.axvline(values.median(), linestyle="-", linewidth=2, label=f"Median = {values.median():.2f}")
    plt.xlabel(target_col)
    plt.ylabel("Number of patients")
    plt.title(f"{target_col} Distribution with Mean and Median")
    plt.legend()
    save_plot(output_folder, "02_histogram_mean_median.png")

    # ============================================================
    # Plot 3: Boxplot
    # ============================================================
    plt.figure(figsize=(8, 5))
    plt.boxplot(values, vert=False)
    plt.xlabel(target_col)
    plt.title(f"Boxplot of {target_col}")
    save_plot(output_folder, "03_boxplot.png")

    # ============================================================
    # Plot 4: ECDF
    # ============================================================
    values_sorted = np.sort(values)
    ecdf = np.arange(1, len(values_sorted) + 1) / len(values_sorted)

    plt.figure(figsize=(9, 6))
    plt.plot(values_sorted, ecdf, marker=".", linestyle="none")
    plt.xlabel(target_col)
    plt.ylabel("Cumulative proportion")
    plt.title(f"Empirical Cumulative Distribution of {target_col}")
    save_plot(output_folder, "04_ecdf.png")

    # ============================================================
    # Plot 5: Missing vs Available Data
    # ============================================================
    plt.figure(figsize=(7, 5))
    plt.bar(missing_summary["Status"], missing_summary["Count"])
    plt.ylabel("Number of records")
    plt.title(f"Missing vs Available Data for {target_col}")

    for i, row in missing_summary.iterrows():
        plt.text(
            i,
            row["Count"],
            f'{row["Count"]} ({row["Percent"]:.1f}%)',
            ha="center",
            va="bottom"
        )

    plt.xticks(rotation=15, ha="right")
    save_plot(output_folder, "05_missing_vs_available.png")

    # ============================================================
    # Export Excel report
    # ============================================================
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        analysis_df.to_excel(writer, sheet_name="Data_with_Outlier_Flag", index=False)
        summary_stats.to_excel(writer, sheet_name="Summary_Statistics", index=False)
        value_counts.to_excel(writer, sheet_name="Value_Counts", index=False)
        percentile_table.to_excel(writer, sheet_name="Percentiles", index=False)
        outlier_summary.to_excel(writer, sheet_name="Outlier_Summary", index=False)
        outlier_rows.to_excel(writer, sheet_name="Outlier_Rows", index=False)
        missing_summary.to_excel(writer, sheet_name="Missing_Summary", index=False)

    print(f"Done: {target_col}")


# ============================================================
# Run analysis for all numeric columns
# ============================================================
for col in df.columns:
    if col in exclude_columns:
        continue

    numeric_series = clean_numeric_series(df[col])

    # Analyze only columns that have at least one valid numeric value
    if numeric_series.notna().sum() > 0:
        analyze_numeric_column(df, col, main_output_folder)
    else:
        print(f"Skipped {col}: no numeric values.")


print("\nAll analyses completed.")
print(f"Main output folder: {main_output_folder}")