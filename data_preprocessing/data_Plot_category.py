import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# File paths
# ============================================================
input_file = r"D:\WORK\MEDICINE\Project\Kidney\DGF\CODS\Group.xlsx"
main_output_folder = r"D:\WORK\MEDICINE\Project\Kidney\DGF\CODS\Categorical_Columns_Analysis"

os.makedirs(main_output_folder, exist_ok=True)


# ============================================================
# Read Excel file
# ============================================================
# If real column names are in the second row, use header=1
df = pd.read_excel(input_file, header=1)

# If real column names are in the first row, use this instead:
# df = pd.read_excel(input_file, dtype=object)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", "_", regex=True)
)


# ============================================================
# Optional: Limit analysis to rows up to a specific Excel row
# ============================================================
# If you want to analyze all rows, set this to None
analyze_until_excel_row = None

# Because header=1:
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
# Columns to exclude from categorical analysis
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
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("._ ")

    if name == "":
        name = "Unnamed_Column"

    return name


def clean_category_value(value):
    """
    Clean categorical values.
    Invalid symbols and Excel errors become NaN.
    Valid categories remain as clean text.
    """

    if pd.isna(value):
        return np.nan

    value_str = str(value).strip()

    # Remove invisible characters
    value_str = value_str.replace("\u200c", "")
    value_str = value_str.replace("\u200f", "")
    value_str = value_str.replace("\u200e", "")
    value_str = value_str.replace("\xa0", "")
    value_str = value_str.strip()

    # Convert Persian/Arabic digits to English digits
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    english_digits = "0123456789"

    for p, e in zip(persian_digits, english_digits):
        value_str = value_str.replace(p, e)

    for a, e in zip(arabic_digits, english_digits):
        value_str = value_str.replace(a, e)

    invalid_values = {
        "",
        "-",
        "–",
        "—",
        "−",
        "?",
        "؟",
        ".",
        "..",
        "...",
        "nan",
        "NaN",
        "NAN",
        "none",
        "None",
        "null",
        "NULL",
        "#DIV/0!",
        "#VALUE!",
        "#REF!",
        "#N/A",
        "#NAME?",
        "#NUM!",
        "#NULL!",
        "خالی",
        "نامشخص",
        "نامعلوم",
        "مشخص نیست",
        "ندارد",
    }

    if value_str in invalid_values:
        return np.nan

    # Convert 1.0 -> 1 and 0.0 -> 0
    if value_str.endswith(".0"):
        try:
            number_value = float(value_str)
            if number_value.is_integer():
                value_str = str(int(number_value))
        except ValueError:
            pass

    return value_str


def save_plot(output_folder, filename):
    """
    Save current matplotlib figure.
    """
    plt.tight_layout()
    path = os.path.join(output_folder, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def analyze_categorical_column(df, target_col, main_output_folder, max_categories_for_plot=20):
    """
    Analyze one categorical column and save outputs in a separate folder.
    """

    folder_name = safe_folder_name(target_col)
    output_folder = os.path.join(main_output_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    output_excel = os.path.join(
        output_folder,
        f"{folder_name}_categorical_analysis_report.xlsx"
    )

    analysis_df = df.copy()
    analysis_df[target_col] = analysis_df[target_col].apply(clean_category_value)

    total_count = len(analysis_df)
    valid_count = analysis_df[target_col].notna().sum()
    missing_count = analysis_df[target_col].isna().sum()

    valid_percent = valid_count / total_count * 100 if total_count > 0 else 0
    missing_percent = missing_count / total_count * 100 if total_count > 0 else 0

    values = analysis_df[target_col].dropna()

    missing_summary = pd.DataFrame({
        "Status": ["Available / Non-missing", "Missing / NaN"],
        "Count": [valid_count, missing_count],
        "Percent": [valid_percent, missing_percent]
    })

    # ============================================================
    # If column has no valid categorical values
    # ============================================================
    if len(values) == 0:
        basic_summary = pd.DataFrame({
            "Metric": [
                "Total count",
                "Valid count",
                "Missing count",
                "Missing percent",
                "Note"
            ],
            "Value": [
                total_count,
                valid_count,
                missing_count,
                missing_percent,
                "No valid categorical values found"
            ]
        })

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
        save_plot(output_folder, "01_missing_vs_available.png")

        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            basic_summary.to_excel(writer, sheet_name="Basic_Summary", index=False)
            missing_summary.to_excel(writer, sheet_name="Missing_Summary", index=False)

        print(f"Skipped {target_col}: no valid categorical values.")
        return

    # ============================================================
    # Category frequency table
    # ============================================================
    category_counts = (
        values
        .value_counts(dropna=False)
        .reset_index()
    )

    category_counts.columns = ["Category", "Count"]
    category_counts["Percent_Among_Valid"] = category_counts["Count"] / valid_count * 100
    category_counts["Percent_Among_All"] = category_counts["Count"] / total_count * 100
    category_counts["Cumulative_Count"] = category_counts["Count"].cumsum()
    category_counts["Cumulative_Percent_Among_Valid"] = (
        category_counts["Percent_Among_Valid"].cumsum()
    )

    category_counts_with_missing = category_counts.copy()

    if missing_count > 0:
        missing_row = pd.DataFrame({
            "Category": ["Missing / NaN"],
            "Count": [missing_count],
            "Percent_Among_Valid": [np.nan],
            "Percent_Among_All": [missing_count / total_count * 100],
            "Cumulative_Count": [np.nan],
            "Cumulative_Percent_Among_Valid": [np.nan]
        })

        category_counts_with_missing = pd.concat(
            [category_counts_with_missing, missing_row],
            ignore_index=True
        )

    # ============================================================
    # Basic categorical summary
    # ============================================================
    unique_count = values.nunique()
    mode_value = values.mode().iloc[0] if len(values.mode()) > 0 else np.nan
    mode_count = (values == mode_value).sum() if pd.notna(mode_value) else np.nan
    mode_percent = mode_count / valid_count * 100 if valid_count > 0 else np.nan

    if unique_count == 1:
        variable_type = "Single-value categorical"
    elif unique_count == 2:
        variable_type = "Binary categorical"
    else:
        variable_type = "Multiclass categorical"

    basic_summary = pd.DataFrame({
        "Metric": [
            "Variable type",
            "Total count",
            "Valid count",
            "Valid percent",
            "Missing count",
            "Missing percent",
            "Number of unique categories",
            "Most frequent category",
            "Most frequent category count",
            "Most frequent category percent among valid values"
        ],
        "Value": [
            variable_type,
            total_count,
            valid_count,
            valid_percent,
            missing_count,
            missing_percent,
            unique_count,
            mode_value,
            mode_count,
            mode_percent
        ]
    })

    # ============================================================
    # Rare category detection
    # ============================================================
    rare_threshold_percent = 5

    rare_categories = category_counts[
        category_counts["Percent_Among_Valid"] < rare_threshold_percent
    ].copy()

    rare_categories["Rare_Threshold_Percent"] = rare_threshold_percent

    # ============================================================
    # Binary summary
    # ============================================================
    if unique_count == 2:
        binary_summary = category_counts.copy()
    else:
        binary_summary = pd.DataFrame({
            "Note": [
                "This column is not binary because it has more or less than 2 non-missing categories."
            ]
        })

    # ============================================================
    # Plot data preparation
    # ============================================================
    plot_counts = category_counts.copy()

    if len(plot_counts) > max_categories_for_plot:
        top_part = plot_counts.head(max_categories_for_plot).copy()

        other_count = plot_counts.iloc[max_categories_for_plot:]["Count"].sum()
        other_percent_valid = plot_counts.iloc[max_categories_for_plot:]["Percent_Among_Valid"].sum()
        other_percent_all = plot_counts.iloc[max_categories_for_plot:]["Percent_Among_All"].sum()

        other_row = pd.DataFrame({
            "Category": ["Other_categories"],
            "Count": [other_count],
            "Percent_Among_Valid": [other_percent_valid],
            "Percent_Among_All": [other_percent_all],
            "Cumulative_Count": [np.nan],
            "Cumulative_Percent_Among_Valid": [np.nan]
        })

        plot_counts = pd.concat([top_part, other_row], ignore_index=True)

    # ============================================================
    # Plot 1: Category count bar plot
    # ============================================================
    plt.figure(figsize=(10, 6))
    plt.bar(plot_counts["Category"].astype(str), plot_counts["Count"])
    plt.xlabel(target_col)
    plt.ylabel("Number of records")
    plt.title(f"Category Count Distribution of {target_col}")
    plt.xticks(rotation=45, ha="right")

    for i, row in plot_counts.iterrows():
        plt.text(
            i,
            row["Count"],
            str(row["Count"]),
            ha="center",
            va="bottom"
        )

    save_plot(output_folder, "01_category_count_bar.png")

    # ============================================================
    # Plot 2: Category percent bar plot
    # ============================================================
    plt.figure(figsize=(10, 6))
    plt.bar(plot_counts["Category"].astype(str), plot_counts["Percent_Among_Valid"])
    plt.xlabel(target_col)
    plt.ylabel("Percent among valid values")
    plt.title(f"Category Percent Distribution of {target_col}")
    plt.xticks(rotation=45, ha="right")

    for i, row in plot_counts.iterrows():
        plt.text(
            i,
            row["Percent_Among_Valid"],
            f'{row["Percent_Among_Valid"]:.1f}%',
            ha="center",
            va="bottom"
        )

    save_plot(output_folder, "02_category_percent_bar.png")

    # ============================================================
    # Plot 3: Missing vs Available
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
    save_plot(output_folder, "03_missing_vs_available.png")

    # ============================================================
    # Plot 4: Pie chart
    # Only for columns with <= 10 categories
    # ============================================================
    if len(category_counts) <= 10:
        plt.figure(figsize=(8, 8))
        plt.pie(
            category_counts["Count"],
            labels=category_counts["Category"].astype(str),
            autopct="%1.1f%%",
            startangle=90
        )
        plt.title(f"Category Proportion of {target_col}")
        save_plot(output_folder, "04_category_pie_chart.png")

    # ============================================================
    # Export Excel report
    # ============================================================
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        analysis_df.to_excel(writer, sheet_name="Data_Cleaned_Category", index=False)
        basic_summary.to_excel(writer, sheet_name="Basic_Summary", index=False)
        category_counts.to_excel(writer, sheet_name="Category_Counts", index=False)
        category_counts_with_missing.to_excel(writer, sheet_name="Category_Counts_Missing", index=False)
        missing_summary.to_excel(writer, sheet_name="Missing_Summary", index=False)
        rare_categories.to_excel(writer, sheet_name="Rare_Categories", index=False)
        binary_summary.to_excel(writer, sheet_name="Binary_Summary", index=False)

    print(f"Done: {target_col}")


# ============================================================
# Run analysis for all categorical columns
# ============================================================
master_summary_rows = []

for col in df.columns:
    if col in exclude_columns:
        continue

    cleaned_col = df[col].apply(clean_category_value)
    valid_values = cleaned_col.dropna()

    if valid_values.shape[0] > 0:
        analyze_categorical_column(df, col, main_output_folder)

        unique_count = valid_values.nunique()
        missing_count = cleaned_col.isna().sum()
        valid_count = cleaned_col.notna().sum()
        total_count = len(cleaned_col)

        if unique_count == 1:
            variable_type = "Single-value categorical"
        elif unique_count == 2:
            variable_type = "Binary categorical"
        else:
            variable_type = "Multiclass categorical"

        mode_value = valid_values.mode().iloc[0] if len(valid_values.mode()) > 0 else np.nan
        mode_count = (valid_values == mode_value).sum() if pd.notna(mode_value) else np.nan
        mode_percent = mode_count / valid_count * 100 if valid_count > 0 else np.nan

        master_summary_rows.append({
            "Column_Name": col,
            "Variable_Type": variable_type,
            "Total_Count": total_count,
            "Valid_Count": valid_count,
            "Missing_Count": missing_count,
            "Missing_Percent": missing_count / total_count * 100 if total_count > 0 else np.nan,
            "Unique_Categories": unique_count,
            "Most_Frequent_Category": mode_value,
            "Most_Frequent_Count": mode_count,
            "Most_Frequent_Percent_Among_Valid": mode_percent
        })

    else:
        print(f"Skipped {col}: no valid categorical values.")


# ============================================================
# Save master summary
# ============================================================
master_summary = pd.DataFrame(master_summary_rows)

master_summary_file = os.path.join(
    main_output_folder,
    "All_Categorical_Columns_Master_Summary.xlsx"
)

master_summary.to_excel(master_summary_file, index=False)


print("\nAll categorical analyses completed.")
print(f"Main output folder: {main_output_folder}")
print(f"Master summary saved as: {master_summary_file}")