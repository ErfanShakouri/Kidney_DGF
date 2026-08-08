import pandas as pd


# ============================================================
# File paths
# ============================================================
input_file = r"D:\WORK\category.xlsx"
output_file = r"D:\WORK\category_report.xlsx"


# ============================================================
# Specify categorical columns here
# ============================================================
category_columns = [
    "Rec_Sex",
    "ESRD_Cause",
    "Dialysis_Type",
    "Rec_Blood_Type",
    "Donor_Sex",
    "Donor_Type",
    "Cause_of_Death",
    "Donor_Blood_Type",
    # Add more column names here
]


# ============================================================
# Read Excel file
# ============================================================
df = pd.read_excel(input_file, header=1)

df.columns = df.columns.astype(str).str.strip()


# ============================================================
# Check missing columns
# ============================================================
missing_columns = [col for col in category_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        f"The following columns were not found in the Excel file: {missing_columns}\n"
        f"Available columns are: {df.columns.tolist()}"
    )


# ============================================================
# Create category report
# ============================================================
category_report_rows = []

for col in category_columns:
    # Keep categories exactly as they are, except removing leading/trailing spaces
    temp = df[col].copy()

    # Convert to string only for non-missing values
    temp_clean = temp.apply(
        lambda x: str(x).strip() if pd.notna(x) else pd.NA
    )

    value_counts = temp_clean.value_counts(dropna=False)

    total_rows = len(temp_clean)

    for category, count in value_counts.items():
        percent = count / total_rows * 100

        category_report_rows.append({
            "Column_Name": col,
            "Category": category,
            "Count": count,
            "Percent": percent
        })


category_report = pd.DataFrame(category_report_rows)


# ============================================================
# Create summary per column
# ============================================================
summary_rows = []

for col in category_columns:
    temp = df[col].copy()

    temp_clean = temp.apply(
        lambda x: str(x).strip() if pd.notna(x) else pd.NA
    )

    unique_count = temp_clean.nunique(dropna=True)
    missing_count = temp_clean.isna().sum()
    total_count = len(temp_clean)

    summary_rows.append({
        "Column_Name": col,
        "Total_Rows": total_count,
        "Missing_Count": missing_count,
        "Non_Missing_Count": total_count - missing_count,
        "Number_of_Categories": unique_count
    })


summary_report = pd.DataFrame(summary_rows)


# ============================================================
# Save output Excel file
# ============================================================
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    category_report.to_excel(writer, sheet_name="Category_Details", index=False)
    summary_report.to_excel(writer, sheet_name="Column_Summary", index=False)


print("Done!")
print(f"Category report saved as: {output_file}")

print("\nColumn summary:")
print(summary_report)