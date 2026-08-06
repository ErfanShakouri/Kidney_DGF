import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# File paths
# ============================================================
input_file = r"D:\WORK\LOS.xlsx"
output_excel = r"D:\WORK\LOS_analysis_report.xlsx"
output_folder = r"D:\WORK\LOS_charts"

target_col = "LOS_Days"

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# Read Excel file
# ============================================================
# Use this if column names are in the first row
df = pd.read_excel(input_file, header=1)

# Use this instead if real column names are in the second row
# df = pd.read_excel(input_file, header=1)

df.columns = df.columns.astype(str).str.strip()

if target_col not in df.columns:
    raise ValueError(
        f"Column '{target_col}' not found. Available columns are: {df.columns.tolist()}"
    )


# ============================================================
# Clean LOS_Days
# ============================================================
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

los = df[target_col].dropna()

if len(los) == 0:
    raise ValueError("No valid numeric values found in LOS_Days.")


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
        "P95"
    ],
    "Value": [
        los.count(),
        df[target_col].isna().sum(),
        los.mean(),
        los.median(),
        los.std(),
        los.min(),
        los.max(),
        los.quantile(0.25),
        los.quantile(0.75),
        los.quantile(0.75) - los.quantile(0.25),
        los.quantile(0.05),
        los.quantile(0.10),
        los.quantile(0.20),
        los.quantile(0.30),
        los.quantile(0.40),
        los.quantile(0.50),
        los.quantile(0.60),
        los.quantile(0.70),
        los.quantile(0.80),
        los.quantile(0.90),
        los.quantile(0.95),
    ]
})


# ============================================================
# Frequency table for each LOS value
# ============================================================
value_counts = (
    los.value_counts()
    .sort_index()
    .reset_index()
)

value_counts.columns = ["LOS_Days", "Count"]
value_counts["Percent"] = value_counts["Count"] / value_counts["Count"].sum() * 100
value_counts["Cumulative_Count"] = value_counts["Count"].cumsum()
value_counts["Cumulative_Percent"] = value_counts["Percent"].cumsum()


# ============================================================
# Outlier detection using IQR
# ============================================================
q1 = los.quantile(0.25)
q3 = los.quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df["LOS_is_outlier_IQR"] = np.where(
    (df[target_col] < lower_bound) | (df[target_col] > upper_bound),
    1,
    0
)

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
        df["LOS_is_outlier_IQR"].sum(),
        df["LOS_is_outlier_IQR"].sum() / los.count() * 100
    ]
})


# ============================================================
# Suggested class definitions
# These are not final clinical labels; they are only data-driven proposals.
# ============================================================

# Proposal 1: clinically interpretable fixed ranges
clinical_bins = [-np.inf, 10, 14, 21, 30, np.inf]
clinical_labels = [
    "Class_1_<=10",
    "Class_2_11_14",
    "Class_3_15_21",
    "Class_4_22_30",
    "Class_5_>30"
]

df["LOS_Class_Clinical_5"] = pd.cut(
    df[target_col],
    bins=clinical_bins,
    labels=clinical_labels
)

clinical_class_summary = (
    df["LOS_Class_Clinical_5"]
    .value_counts(dropna=False)
    .reset_index()
)

clinical_class_summary.columns = ["LOS_Class_Clinical_5", "Count"]
clinical_class_summary["Percent"] = (
    clinical_class_summary["Count"] / clinical_class_summary["Count"].sum() * 100
)


# Proposal 2: quantile-based 3 classes
df["LOS_Class_Quantile_3"] = pd.qcut(
    df[target_col],
    q=3,
    labels=["Low_LOS", "Medium_LOS", "High_LOS"],
    duplicates="drop"
)

quantile_3_summary = (
    df["LOS_Class_Quantile_3"]
    .value_counts(dropna=False)
    .reset_index()
)

quantile_3_summary.columns = ["LOS_Class_Quantile_3", "Count"]
quantile_3_summary["Percent"] = (
    quantile_3_summary["Count"] / quantile_3_summary["Count"].sum() * 100
)


# Proposal 3: quantile-based 4 classes
df["LOS_Class_Quantile_4"] = pd.qcut(
    df[target_col],
    q=4,
    labels=["Q1_Short", "Q2_Moderate", "Q3_Long", "Q4_Very_Long"],
    duplicates="drop"
)

quantile_4_summary = (
    df["LOS_Class_Quantile_4"]
    .value_counts(dropna=False)
    .reset_index()
)

quantile_4_summary.columns = ["LOS_Class_Quantile_4", "Count"]
quantile_4_summary["Percent"] = (
    quantile_4_summary["Count"] / quantile_4_summary["Count"].sum() * 100
)


# Proposal 4: quantile-based 5 classes
df["LOS_Class_Quantile_5"] = pd.qcut(
    df[target_col],
    q=5,
    labels=[
        "Q1_Very_Short",
        "Q2_Short",
        "Q3_Moderate",
        "Q4_Long",
        "Q5_Very_Long"
    ],
    duplicates="drop"
)

quantile_5_summary = (
    df["LOS_Class_Quantile_5"]
    .value_counts(dropna=False)
    .reset_index()
)

quantile_5_summary.columns = ["LOS_Class_Quantile_5", "Count"]
quantile_5_summary["Percent"] = (
    quantile_5_summary["Count"] / quantile_5_summary["Count"].sum() * 100
)


# ============================================================
# Helper function for saving plots
# ============================================================
def save_plot(filename):
    plt.tight_layout()
    path = os.path.join(output_folder, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


# ============================================================
# Plot 1: Histogram
# ============================================================
plt.figure(figsize=(9, 6))
plt.hist(los, bins=20, edgecolor="black")
plt.xlabel("LOS_Days")
plt.ylabel("Number of patients")
plt.title("Distribution of LOS_Days")
save_plot("01_LOS_histogram.png")


# ============================================================
# Plot 2: Histogram with mean and median
# ============================================================
plt.figure(figsize=(9, 6))
plt.hist(los, bins=20, edgecolor="black")
plt.axvline(los.mean(), linestyle="--", linewidth=2, label=f"Mean = {los.mean():.1f}")
plt.axvline(los.median(), linestyle="-", linewidth=2, label=f"Median = {los.median():.1f}")
plt.xlabel("LOS_Days")
plt.ylabel("Number of patients")
plt.title("LOS_Days Distribution with Mean and Median")
plt.legend()
save_plot("02_LOS_histogram_mean_median.png")


# ============================================================
# Plot 3: Boxplot
# ============================================================
plt.figure(figsize=(8, 5))
plt.boxplot(los, vert=False)
plt.xlabel("LOS_Days")
plt.title("Boxplot of LOS_Days")
save_plot("03_LOS_boxplot.png")


# ============================================================
# Plot 4: Empirical cumulative distribution
# ============================================================
los_sorted = np.sort(los)
ecdf = np.arange(1, len(los_sorted) + 1) / len(los_sorted)

plt.figure(figsize=(9, 6))
plt.plot(los_sorted, ecdf, marker=".", linestyle="none")
plt.xlabel("LOS_Days")
plt.ylabel("Cumulative proportion")
plt.title("Empirical Cumulative Distribution of LOS_Days")
save_plot("04_LOS_ecdf.png")


# ============================================================
# Plot 5: Percentile curve
# ============================================================
percentile_points = list(range(0, 101, 5))
percentile_values = np.percentile(los, percentile_points)

plt.figure(figsize=(9, 6))
plt.plot(percentile_points, percentile_values, marker="o")
plt.xlabel("Percentile")
plt.ylabel("LOS_Days")
plt.title("LOS_Days Percentile Curve")
save_plot("05_LOS_percentile_curve.png")


# ============================================================
# Plot 6: Frequency by exact LOS day
# ============================================================
plt.figure(figsize=(12, 6))
plt.bar(value_counts["LOS_Days"], value_counts["Count"])
plt.xlabel("LOS_Days")
plt.ylabel("Number of patients")
plt.title("Frequency of Exact LOS_Days Values")
save_plot("06_LOS_exact_day_frequency.png")


# ============================================================
# Plot 7: Clinical 5-class proposal
# ============================================================
plt.figure(figsize=(10, 6))
plt.bar(
    clinical_class_summary["LOS_Class_Clinical_5"].astype(str),
    clinical_class_summary["Count"]
)
plt.xlabel("LOS class")
plt.ylabel("Number of patients")
plt.title("Suggested Clinical 5-Class LOS Distribution")
plt.xticks(rotation=30, ha="right")
save_plot("07_LOS_clinical_5class_distribution.png")


# ============================================================
# Plot 8: Quantile 3-class proposal
# ============================================================
plt.figure(figsize=(8, 5))
plt.bar(
    quantile_3_summary["LOS_Class_Quantile_3"].astype(str),
    quantile_3_summary["Count"]
)
plt.xlabel("LOS class")
plt.ylabel("Number of patients")
plt.title("Quantile-Based 3-Class LOS Distribution")
save_plot("08_LOS_quantile_3class_distribution.png")


# ============================================================
# Plot 9: Quantile 4-class proposal
# ============================================================
plt.figure(figsize=(8, 5))
plt.bar(
    quantile_4_summary["LOS_Class_Quantile_4"].astype(str),
    quantile_4_summary["Count"]
)
plt.xlabel("LOS class")
plt.ylabel("Number of patients")
plt.title("Quantile-Based 4-Class LOS Distribution")
plt.xticks(rotation=20, ha="right")
save_plot("09_LOS_quantile_4class_distribution.png")


# ============================================================
# Plot 10: Quantile 5-class proposal
# ============================================================
plt.figure(figsize=(10, 6))
plt.bar(
    quantile_5_summary["LOS_Class_Quantile_5"].astype(str),
    quantile_5_summary["Count"]
)
plt.xlabel("LOS class")
plt.ylabel("Number of patients")
plt.title("Quantile-Based 5-Class LOS Distribution")
plt.xticks(rotation=30, ha="right")
save_plot("10_LOS_quantile_5class_distribution.png")


# ============================================================
# Export Excel report
# ============================================================
percentile_table = pd.DataFrame({
    "Percentile": percentile_points,
    "LOS_Days": percentile_values
})

with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Data_with_LOS_Classes", index=False)
    summary_stats.to_excel(writer, sheet_name="Summary_Statistics", index=False)
    value_counts.to_excel(writer, sheet_name="Value_Counts", index=False)
    percentile_table.to_excel(writer, sheet_name="Percentiles", index=False)
    outlier_summary.to_excel(writer, sheet_name="Outlier_Summary", index=False)
    clinical_class_summary.to_excel(writer, sheet_name="Clinical_5Class", index=False)
    quantile_3_summary.to_excel(writer, sheet_name="Quantile_3Class", index=False)
    quantile_4_summary.to_excel(writer, sheet_name="Quantile_4Class", index=False)
    quantile_5_summary.to_excel(writer, sheet_name="Quantile_5Class", index=False)


# ============================================================
# Print final summary
# ============================================================
print("Done!")
print(f"Excel report saved as: {output_excel}")
print(f"Charts saved in folder: {output_folder}")

print("\nSummary statistics:")
print(summary_stats)

print("\nClinical 5-class proposal:")
print(clinical_class_summary)

print("\nQuantile 3-class proposal:")
print(quantile_3_summary)

print("\nQuantile 4-class proposal:")
print(quantile_4_summary)

print("\nQuantile 5-class proposal:")
print(quantile_5_summary)